import json
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient
from numpy.typing import NDArray
from pytest import MonkeyPatch

from app.core.config import Settings
from app.main import create_app
from app.schemas.inference_schema import BoundingBox, DetectionResponse
from app.services.model_registry_service import ModelRegistryService

AUTH_HEADERS = {"X-API-Key": "dev-secret-key"}


def test_image_inference_returns_detections(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    client = _build_test_client(tmp_path)
    image_bytes = _build_test_image_bytes()

    monkeypatch.setattr(
        ModelRegistryService,
        "run_inference",
        _fake_run_inference,
    )

    response = client.post(
        "/api/v1/inference/image",
        files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        headers=AUTH_HEADERS,
    )

    response_data = response.json()
    execution_id = response_data["execution_id"]

    saved_upload_path = tmp_path / "uploads" / f"{execution_id}.jpg"
    saved_result_path = tmp_path / "outputs" / f"{execution_id}.json"
    saved_annotated_path = (
        tmp_path / "outputs" / f"{execution_id}_annotated.png"
    )

    assert response.status_code == 200

    assert saved_upload_path.exists()
    assert saved_result_path.exists()
    assert saved_annotated_path.exists()

    assert response_data["filename"] == "test.jpg"
    assert response_data["result_url"] == f"/api/v1/results/{execution_id}"
    assert (
        response_data["annotated_image_url"]
        == f"/api/v1/results/{execution_id}/image"
    )

    assert response_data["detections_count"] == 1
    assert response_data["detections"] == [
        {
            "class_id": 0,
            "class_name": "person",
            "confidence": 0.95,
            "box": {
                "x1": 10.0,
                "y1": 20.0,
                "x2": 100.0,
                "y2": 200.0,
            },
        }
    ]

    saved_result = json.loads(saved_result_path.read_text(encoding="utf-8"))

    assert saved_result == response_data


def test_image_inference_rejects_unsupported_file_type(
    tmp_path: Path,
) -> None:
    client = _build_test_client(tmp_path)

    response = client.post(
        "/api/v1/inference/image",
        files={"file": ("test.txt", b"not an image", "text/plain")},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 415
    assert response.json() == {
        "detail": "Unsupported image type: text/plain",
    }


def test_image_inference_rejects_empty_image(
    tmp_path: Path,
) -> None:
    client = _build_test_client(tmp_path)

    response = client.post(
        "/api/v1/inference/image",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Uploaded image is empty.",
    }


def test_image_inference_requires_loaded_model(
    tmp_path: Path,
) -> None:
    client = _build_test_client(tmp_path)
    image_bytes = _build_test_image_bytes()
    model_path = tmp_path / "fake_model.pt"

    response = client.post(
        "/api/v1/inference/image",
        files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 409
    assert response.json() == {
        "detail": f"YOLO model is not loaded: {model_path}",
    }


def test_image_inference_rejects_mismatched_image_signature(
    tmp_path: Path,
) -> None:
    client = _build_test_client(tmp_path)

    response = client.post(
        "/api/v1/inference/image",
        files={"file": ("fake.png", b"not-a-real-png", "image/png")},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": (
            "Uploaded image content does not match content type: image/png"
        ),
    }


def _build_test_client(tmp_path: Path) -> TestClient:
    settings = Settings(
        app_name="Test YOLO11 API",
        app_version="0.0.1",
        environment="test",
        log_level="INFO",
        api_prefix="/api/v1",
        yolo_model_path=tmp_path / "fake_model.pt",
        max_image_size_mb=10,
        upload_dir=tmp_path / "uploads",
        output_dir=tmp_path / "outputs",
        api_key="dev-secret-key",
    )

    return TestClient(create_app(settings=settings))


def _build_test_image_bytes() -> bytes:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    success, encoded_image = cv2.imencode(".jpg", image)

    if not success:
        raise RuntimeError("Failed to build test image.")

    return encoded_image.tobytes()


def _fake_run_inference(
    self: ModelRegistryService,
    image: NDArray[np.uint8],
) -> list[DetectionResponse]:
    return [
        DetectionResponse(
            class_id=0,
            class_name="person",
            confidence=0.95,
            box=BoundingBox(
                x1=10.0,
                y1=20.0,
                x2=100.0,
                y2=200.0,
            ),
        )
    ]