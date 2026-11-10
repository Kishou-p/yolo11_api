import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from numpy.typing import NDArray
from pytest import MonkeyPatch

from app.core.config import Settings
from app.core.logging_config import setup_logging
from app.main import create_app
from app.schemas.inference_schema import BoundingBox, DetectionResponse
from app.services.model_registry_service import ModelRegistryService
from app.services.retention_service import RetentionService

AUTH_HEADERS = {"X-API-Key": "dev-secret-key"}


@pytest.fixture(autouse=True)
def fake_inference(monkeypatch: MonkeyPatch) -> None:
    def run_inference(
        self: ModelRegistryService,
        image: NDArray[np.uint8],
    ) -> list[DetectionResponse]:
        return [
            DetectionResponse(
                class_id=0,
                class_name="part",
                confidence=0.9,
                box=BoundingBox(x1=1, y1=1, x2=5, y2=5),
            )
        ]

    monkeypatch.setattr(ModelRegistryService, "run_inference", run_inference)


def test_settings_map_to_inference_params(tmp_path: Path) -> None:
    settings = _build_settings(
        tmp_path,
        yolo_conf=0.6,
        yolo_iou=0.3,
        yolo_imgsz=1280,
        yolo_max_det=50,
        yolo_device="cpu",
        yolo_half=True,
    )

    assert settings.inference_params.model_dump() == {
        "conf": 0.6,
        "iou": 0.3,
        "imgsz": 1280,
        "max_det": 50,
        "device": "cpu",
        "half": True,
    }


def test_invalid_confidence_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _build_settings(tmp_path, yolo_conf=1.5)


def test_result_records_model_params_and_timings(tmp_path: Path) -> None:
    client = _build_client(tmp_path, yolo_conf=0.4)

    data = _post_image(client).json()
    saved = json.loads(
        (tmp_path / "outputs" / f"{data['execution_id']}.json").read_text()
    )

    assert data["model"] == {"name": "missing_model.pt", "sha256": None}
    assert data["params"]["conf"] == 0.4
    assert set(data["timings"]) == {"decode_ms", "inference_ms", "total_ms"}
    assert data["timings"]["total_ms"] >= data["timings"]["inference_ms"]
    assert data["created_at"]
    assert saved == data


def test_save_uploads_false_skips_original_image(tmp_path: Path) -> None:
    client = _build_client(tmp_path, save_uploads=False)

    response = _post_image(client)

    assert response.status_code == 200
    assert list((tmp_path / "uploads").iterdir()) == []


def test_save_annotated_false_skips_annotated_image(tmp_path: Path) -> None:
    client = _build_client(tmp_path, save_annotated=False)

    data = _post_image(client).json()
    image_response = client.get(
        f"/api/v1/results/{data['execution_id']}/image",
        headers=AUTH_HEADERS,
    )
    listing = client.get("/api/v1/results", headers=AUTH_HEADERS).json()

    assert data["annotated_image_url"] is None
    assert image_response.status_code == 404
    assert listing["results"][0]["annotated_image_url"] is None


def test_annotated_image_can_be_saved_as_jpeg(tmp_path: Path) -> None:
    client = _build_client(tmp_path, annotated_image_format="jpg")

    execution_id = _post_image(client).json()["execution_id"]
    image_response = client.get(
        f"/api/v1/results/{execution_id}/image",
        headers=AUTH_HEADERS,
    )
    delete_response = client.delete(
        f"/api/v1/results/{execution_id}",
        headers=AUTH_HEADERS,
    )

    assert image_response.status_code == 200
    assert image_response.headers["content-type"] == "image/jpeg"
    assert delete_response.status_code == 200
    assert list((tmp_path / "outputs").iterdir()) == []
    assert list((tmp_path / "uploads").iterdir()) == []


def test_retention_deletes_only_expired_files(tmp_path: Path) -> None:
    directory = tmp_path / "outputs"
    directory.mkdir()
    old_file = directory / "old.json"
    new_file = directory / "new.json"
    keep_file = directory / ".gitkeep"

    for path in (old_file, new_file, keep_file):
        path.write_text("{}")

    two_days_ago = time.time() - 2 * 24 * 60 * 60
    os.utime(old_file, (two_days_ago, two_days_ago))
    os.utime(keep_file, (two_days_ago, two_days_ago))

    deleted = RetentionService([directory], retention_days=1).purge_expired_files()

    assert deleted == 1
    assert sorted(path.name for path in directory.iterdir()) == [
        ".gitkeep",
        "new.json",
    ]


def test_retention_runs_in_background_on_startup(tmp_path: Path) -> None:
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    old_file = uploads / "old.jpg"
    old_file.write_bytes(b"x")
    two_days_ago = time.time() - 2 * 24 * 60 * 60
    os.utime(old_file, (two_days_ago, two_days_ago))

    settings = _build_settings(tmp_path, retention_days=1)

    with TestClient(create_app(settings)):
        deadline = time.time() + 5
        while old_file.exists() and time.time() < deadline:
            time.sleep(0.05)

    assert not old_file.exists()


def test_response_includes_generated_request_id(tmp_path: Path) -> None:
    client = _build_client(tmp_path)

    response = client.get("/api/v1/health")

    assert len(response.headers["x-request-id"]) == 32


def test_valid_incoming_request_id_is_reused(tmp_path: Path) -> None:
    client = _build_client(tmp_path)

    response = client.get(
        "/api/v1/health",
        headers={"X-Request-ID": "line-3.cam-7_0001"},
    )

    assert response.headers["x-request-id"] == "line-3.cam-7_0001"


def test_invalid_incoming_request_id_is_replaced(tmp_path: Path) -> None:
    client = _build_client(tmp_path)

    response = client.get(
        "/api/v1/health",
        headers={"X-Request-ID": "bad id\nwith newline"},
    )

    assert response.headers["x-request-id"] != "bad id\nwith newline"
    assert len(response.headers["x-request-id"]) == 32


def test_json_logs_include_request_id_and_inference_fields(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = _build_client(tmp_path, log_format="json")

    response = _post_image(client, headers={"X-Request-ID": "req-42"})
    records = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("{")
    ]
    logging.getLogger().handlers.clear()

    inference_log = next(r for r in records if r["message"] == "Inference completed")
    access_log = next(r for r in records if r["logger"] == "app.access")

    assert inference_log["request_id"] == "req-42"
    assert inference_log["execution_id"] == response.json()["execution_id"]
    assert inference_log["detections_count"] == 1
    assert access_log["route"] == "/api/v1/inference/image"
    assert access_log["status"] == 200


def test_metrics_require_api_key(tmp_path: Path) -> None:
    client = _build_client(tmp_path)

    assert client.get("/metrics").status_code == 401


def test_metrics_report_inferences_and_route_templates(tmp_path: Path) -> None:
    client = _build_client(tmp_path)
    execution_id = _post_image(client).json()["execution_id"]
    client.get(f"/api/v1/results/{execution_id}", headers=AUTH_HEADERS)
    client.post(
        "/api/v1/inference/image",
        files={"file": ("a.txt", b"text", "text/plain")},
        headers=AUTH_HEADERS,
    )

    body = client.get("/metrics", headers=AUTH_HEADERS).text

    assert 'inference_total{outcome="success"} 1.0' in body
    assert 'inference_total{outcome="rejected"} 1.0' in body
    assert 'route="/api/v1/results/{execution_id}"' in body
    assert execution_id not in body
    assert "model_loaded 0.0" in body


def test_metrics_can_be_disabled(tmp_path: Path) -> None:
    client = _build_client(tmp_path, metrics_enabled=False)

    assert client.get("/metrics", headers=AUTH_HEADERS).status_code == 404


def _post_image(
    client: TestClient,
    headers: dict[str, str] | None = None,
) -> Any:
    _, encoded = cv2.imencode(".jpg", np.zeros((10, 10, 3), dtype=np.uint8))

    return client.post(
        "/api/v1/inference/image",
        files={"file": ("a.jpg", encoded.tobytes(), "image/jpeg")},
        headers={**AUTH_HEADERS, **(headers or {})},
    )


def _build_client(tmp_path: Path, **overrides: Any) -> TestClient:
    return TestClient(create_app(_build_settings(tmp_path, **overrides)))


def _build_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "environment": "test",
        "yolo_model_path": tmp_path / "missing_model.pt",
        "upload_dir": tmp_path / "uploads",
        "output_dir": tmp_path / "outputs",
        "api_key": "dev-secret-key",
        "load_model_on_startup": False,
        **overrides,
    }

    return Settings(**values)
