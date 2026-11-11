import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from app.core.config import Settings
from app.main import create_app
from app.schemas.inference_schema import BoundingBox, DetectionResponse
from app.services.demo_service import DemoService
from app.services.yolo_service import YoloService

AUTH = {"X-API-Key": "dev-secret-key"}
GOOD_BOX = {"class_id": 0, "x": 0.1, "y": 0.2, "w": 0.5, "h": 0.4}
BAD_BOX = {"class_id": 1, "x": 0.3, "y": 0.3, "w": 0.4, "h": 0.4}


def _client(tmp_path: Path, **overrides: Any) -> TestClient:
    settings = Settings(
        environment="test",
        yolo_model_path=tmp_path / "base.pt",
        upload_dir=tmp_path / "uploads",
        output_dir=tmp_path / "outputs",
        demo_dir=tmp_path / "demo",
        load_model_on_startup=False,
        api_key="dev-secret-key",
        demo_min_images=2,
        **overrides,
    )
    return TestClient(create_app(settings=settings))


def _jpeg(width: int = 200, height: int = 100) -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.full((height, width, 3), 120, np.uint8))
    assert ok
    return encoded.tobytes()


def _add(client: TestClient, boxes: list[dict[str, Any]]) -> Any:
    return client.post(
        "/api/v1/demo/dataset",
        headers=AUTH,
        files={"file": ("foto.jpg", _jpeg(), "image/jpeg")},
        data={"boxes": json.dumps(boxes)},
    )


def test_page_is_public_but_api_needs_key(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        page = client.get("/demo")
        assert page.status_code == 200
        assert "Inspeção de peças" in page.text

        assert client.get("/api/v1/demo/dataset").status_code == 401
        assert client.get("/api/v1/demo/train").status_code == 401


def test_demo_can_be_disabled(tmp_path: Path) -> None:
    with _client(tmp_path, demo_enabled=False) as client:
        assert client.get("/demo").status_code == 404
        assert client.get("/api/v1/demo/dataset", headers=AUTH).status_code == 404


def test_add_list_and_delete_image(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = _add(client, [GOOD_BOX])
        assert response.status_code == 200
        image_id = response.json()["image_id"]

        label = (tmp_path / "demo" / "labels" / f"{image_id}.txt").read_text()
        assert label == "0 0.350000 0.400000 0.500000 0.400000\n"

        dataset = client.get("/api/v1/demo/dataset", headers=AUTH).json()
        assert dataset["counts"] == {"peca": 1, "falta": 0, "imagens": 1}
        box = dataset["images"][0]["boxes"][0]
        assert box["x"] == pytest.approx(0.1) and box["w"] == pytest.approx(0.5)

        image = client.get(f"/api/v1/demo/dataset/{image_id}/image", headers=AUTH)
        assert image.status_code == 200
        assert image.content.startswith(b"\xff\xd8\xff")

        deleted = client.delete(f"/api/v1/demo/dataset/{image_id}", headers=AUTH)
        assert deleted.status_code == 204
        assert client.get("/api/v1/demo/dataset", headers=AUTH).json()["images"] == []


def test_rejects_bad_boxes_and_ids(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        assert _add(client, [{"class_id": 0, "x": 2, "y": 0, "w": 1, "h": 1}]).status_code == 422
        assert _add(client, [{**GOOD_BOX, "class_id": 7}]).status_code == 400
        response = client.post(
            "/api/v1/demo/dataset",
            headers=AUTH,
            files={"file": ("foto.jpg", _jpeg(), "image/jpeg")},
            data={"boxes": "not json"},
        )
        assert response.status_code == 422
        assert client.get("/api/v1/demo/dataset/..%2F..%2Fetc/image", headers=AUTH).status_code == 404
        assert client.delete("/api/v1/demo/dataset/abc", headers=AUTH).status_code == 404


def test_rejects_non_image_upload(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.post(
            "/api/v1/demo/dataset",
            headers=AUTH,
            files={"file": ("x.jpg", b"not an image", "image/jpeg")},
        )
        assert response.status_code == 400


def test_training_needs_examples_of_every_class(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        _add(client, [GOOD_BOX])
        response = client.post("/api/v1/demo/train", headers=AUTH)
        assert response.status_code == 400

        _add(client, [GOOD_BOX])
        response = client.post("/api/v1/demo/train", headers=AUTH)
        assert response.status_code == 400
        assert "falta" in response.json()["detail"]


def test_predict_before_training_is_refused(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.post(
            "/api/v1/demo/predict",
            headers=AUTH,
            files={"file": ("t.jpg", _jpeg(), "image/jpeg")},
        )
        assert response.status_code == 409
        assert client.get("/api/v1/demo/model", headers=AUTH).status_code == 404


def test_train_then_predict(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    def fake_load(self: YoloService) -> object:
        return object()

    def fake_train(self: YoloService, model: object, **kwargs: Any) -> Path:
        calls.update(kwargs)
        data = json.loads(Path(kwargs["data_yaml"]).read_text())
        assert data["names"] == {"0": "peca", "1": "falta"}
        for epoch in (1, 2):
            kwargs["on_epoch_end"](epoch, 2, {"metrics/mAP50(B)": 0.9})
        best = Path(kwargs["output_dir"]) / "best.pt"
        best.parent.mkdir(parents=True, exist_ok=True)
        best.write_bytes(b"weights")
        return best

    def fake_predict(self: YoloService, model: object, image: Any) -> list[DetectionResponse]:
        return [
            DetectionResponse(
                class_id=1,
                class_name="falta",
                confidence=0.8,
                box=BoundingBox(x1=20, y1=10, x2=120, y2=60),
            )
        ]

    monkeypatch.setattr(YoloService, "load_model", fake_load)
    monkeypatch.setattr(YoloService, "train", fake_train)
    monkeypatch.setattr(YoloService, "predict", fake_predict)

    with _client(tmp_path) as client:
        _add(client, [GOOD_BOX])
        _add(client, [BAD_BOX])

        started = client.post("/api/v1/demo/train", headers=AUTH, json={"epochs": 2})
        assert started.status_code == 200
        demo: DemoService = client.app.state.demo_service  # type: ignore[attr-defined]
        demo.wait_for_training(timeout=10)

        status = client.get("/api/v1/demo/train", headers=AUTH).json()
        assert status["state"] == "done"
        assert status["trained"] is True
        assert status["epoch"] == 2 and status["map50"] == 0.9
        assert status["trained_images"] == 2
        assert calls["epochs"] == 2 and calls["batch"] == 2

        result = client.post(
            "/api/v1/demo/predict",
            headers=AUTH,
            files={"file": ("t.jpg", _jpeg(200, 100), "image/jpeg")},
        ).json()
        assert result["verdict"] == "defeito"
        detection = result["detections"][0]
        assert detection["good"] is False
        assert detection["x"] == pytest.approx(0.1)
        assert detection["w"] == pytest.approx(0.5)

        model = client.get("/api/v1/demo/model", headers=AUTH)
        assert model.status_code == 200 and model.content == b"weights"


def test_training_failure_is_reported(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    def failing_load(self: YoloService) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr(YoloService, "load_model", failing_load)

    with _client(tmp_path) as client:
        _add(client, [GOOD_BOX])
        _add(client, [BAD_BOX])
        client.post("/api/v1/demo/train", headers=AUTH)
        demo: DemoService = client.app.state.demo_service  # type: ignore[attr-defined]
        demo.wait_for_training(timeout=10)

        status = client.get("/api/v1/demo/train", headers=AUTH).json()
        assert status["state"] == "error"
        assert status["trained"] is False
