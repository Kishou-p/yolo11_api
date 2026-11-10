import json
import threading
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
from app.exceptions.app_exceptions import ModelInferenceError, StorageWriteError
from app.main import create_app
from app.schemas.inference_schema import DetectionResponse
from app.services.inference_service import InferenceService
from app.services.model_registry_service import ModelRegistryService
from app.services.result_service import ResultService
from app.services.storage_service import StorageService
from app.services.yolo_service import YoloService
from app.utils.file_utils import save_json_to_file

AUTH_HEADERS = {"X-API-Key": "dev-secret-key"}


def test_model_is_loaded_and_warmed_up_on_startup(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    predict_calls = {"count": 0}

    def fake_predict(
        self: YoloService,
        model: Any,
        image: NDArray[np.uint8],
    ) -> list[DetectionResponse]:
        predict_calls["count"] += 1
        return []

    monkeypatch.setattr(YoloService, "load_model", lambda self: object())
    monkeypatch.setattr(YoloService, "predict", fake_predict)

    with TestClient(create_app(_build_settings(tmp_path))) as client:
        ready_response = client.get("/api/v1/health/ready")
        inference_response = client.post(
            "/api/v1/inference/image",
            files={"file": ("a.jpg", _build_jpeg(), "image/jpeg")},
            headers=AUTH_HEADERS,
        )

    assert ready_response.status_code == 200
    assert ready_response.json()["status"] == "ready"
    assert inference_response.status_code == 200
    assert predict_calls["count"] == 2  # warmup + request


def test_startup_without_model_keeps_api_up_but_not_ready(
    tmp_path: Path,
) -> None:
    with TestClient(create_app(_build_settings(tmp_path))) as client:
        live_response = client.get("/api/v1/health/live")
        ready_response = client.get("/api/v1/health/ready")

    assert live_response.status_code == 200
    assert ready_response.status_code == 503
    assert ready_response.json()["checks"] == {
        "model_loaded": False,
        "storage_writable": True,
        "disk_space_ok": True,
    }


def test_readiness_reports_low_disk_space(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(YoloService, "load_model", lambda self: object())
    monkeypatch.setattr(YoloService, "predict", lambda self, m, i: [])
    settings = _build_settings(tmp_path, min_free_disk_mb=10**12)

    with TestClient(create_app(settings)) as client:
        response = client.get("/api/v1/health/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["disk_space_ok"] is False


def test_concurrent_load_requests_load_model_once() -> None:
    load_calls = {"count": 0}

    class SlowYoloService:
        model_path = Path("model.pt")

        def load_model(self) -> Any:
            load_calls["count"] += 1
            time.sleep(0.05)
            return object()

    registry = ModelRegistryService(yolo_service=SlowYoloService())  # type: ignore[arg-type]
    threads = [threading.Thread(target=registry.load_model) for _ in range(8)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert load_calls["count"] == 1


def test_failed_inference_writes_no_files(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    def failing_inference(
        self: ModelRegistryService,
        image: NDArray[np.uint8],
    ) -> list[DetectionResponse]:
        raise ModelInferenceError()

    monkeypatch.setattr(ModelRegistryService, "run_inference", failing_inference)
    client = TestClient(create_app(_build_settings(tmp_path)))

    response = client.post(
        "/api/v1/inference/image",
        files={"file": ("a.jpg", _build_jpeg(), "image/jpeg")},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 500
    assert list((tmp_path / "uploads").iterdir()) == []
    assert list((tmp_path / "outputs").iterdir()) == []


def test_failed_result_write_removes_partial_files(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    storage_service = StorageService(
        upload_dir=tmp_path / "uploads",
        output_dir=tmp_path / "outputs",
    )

    def failing_save(self: StorageService, execution_id: str, result: Any) -> Path:
        raise StorageWriteError(tmp_path / "outputs" / f"{execution_id}.json")

    monkeypatch.setattr(StorageService, "save_inference_result", failing_save)

    class FakeRegistry:
        def run_inference(self, image: Any) -> list[DetectionResponse]:
            return []

    service = InferenceService(
        model_registry=FakeRegistry(),  # type: ignore[arg-type]
        storage_service=storage_service,
        max_image_size_mb=10,
        api_prefix="/api/v1",
    )

    with pytest.raises(StorageWriteError):
        service.run_image_inference(
            filename="a.jpg",
            content_type="image/jpeg",
            image_bytes=_build_jpeg(),
        )

    assert list((tmp_path / "uploads").iterdir()) == []
    assert list((tmp_path / "outputs").iterdir()) == []


def test_json_write_is_atomic_and_leaves_no_temp_files(tmp_path: Path) -> None:
    target = tmp_path / "result.json"

    save_json_to_file(target, {"a": 1})
    save_json_to_file(target, {"a": 2})

    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 2}
    assert [path.name for path in tmp_path.iterdir()] == ["result.json"]


def test_list_results_is_paginated(tmp_path: Path) -> None:
    storage_service = StorageService(
        upload_dir=tmp_path / "uploads",
        output_dir=tmp_path / "outputs",
    )

    for index in range(5):
        execution_id = f"{index:032x}"
        save_json_to_file(
            storage_service.build_result_path(execution_id),
            {"execution_id": execution_id},
        )

    page = ResultService(storage_service=storage_service).list_results(
        limit=2,
        offset=1,
    )

    assert page.count == 2
    assert page.total == 5
    assert page.limit == 2
    assert page.offset == 1


def test_list_results_route_validates_limit(tmp_path: Path) -> None:
    client = TestClient(create_app(_build_settings(tmp_path)))

    response = client.get(
        "/api/v1/results?limit=5000",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 422


def _build_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "environment": "test",
        "yolo_model_path": tmp_path / "missing_model.pt",
        "upload_dir": tmp_path / "uploads",
        "output_dir": tmp_path / "outputs",
        "api_key": "dev-secret-key",
        "min_free_disk_mb": 0,
        **overrides,
    }

    return Settings(**values)


def _build_jpeg() -> bytes:
    _, encoded = cv2.imencode(".jpg", np.zeros((10, 10, 3), dtype=np.uint8))
    return encoded.tobytes()
