from typing import Any, cast

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from app.exceptions.app_exceptions import (
    EmptyImageError,
    ImageSignatureMismatchError,
    UnsupportedImageTypeError,
)
from app.schemas.inference_schema import BoundingBox, DetectionResponse
from app.services.inference_service import InferenceService
from app.services.model_registry_service import ModelRegistryService
from app.services.storage_service import StorageService


def test_run_image_inference_returns_public_response() -> None:
    model_registry = FakeModelRegistryService()
    storage_service = FakeStorageService()
    service = _build_inference_service(model_registry, storage_service)
    image_bytes = _build_test_image_bytes()

    response = service.run_image_inference(
        filename="test.jpg",
        content_type="image/jpeg",
        image_bytes=image_bytes,
    )

    assert response.execution_id == "exec123"
    assert response.filename == "test.jpg"
    assert response.result_url == "/api/v1/results/exec123"
    assert response.annotated_image_url == "/api/v1/results/exec123/image"
    assert response.detections_count == 1
    assert response.detections == [_build_detection()]


def test_run_image_inference_saves_upload_annotated_image_and_result() -> None:
    model_registry = FakeModelRegistryService()
    storage_service = FakeStorageService()
    service = _build_inference_service(model_registry, storage_service)
    image_bytes = _build_test_image_bytes()

    response = service.run_image_inference(
        filename="test.jpg",
        content_type="image/jpeg",
        image_bytes=image_bytes,
    )

    assert storage_service.saved_uploads == [
        {
            "execution_id": "exec123",
            "original_filename": "test.jpg",
            "image_bytes": image_bytes,
        }
    ]
    assert storage_service.saved_annotated_images_count == 1
    assert storage_service.saved_results == [
        {
            "execution_id": "exec123",
            "result": response.model_dump(),
        }
    ]


def test_run_image_inference_calls_model_registry_with_decoded_image() -> None:
    model_registry = FakeModelRegistryService()
    storage_service = FakeStorageService()
    service = _build_inference_service(model_registry, storage_service)
    image_bytes = _build_test_image_bytes()

    service.run_image_inference(
        filename="test.jpg",
        content_type="image/jpeg",
        image_bytes=image_bytes,
    )

    assert model_registry.run_inference_calls == 1
    assert isinstance(model_registry.last_image, np.ndarray)
    assert model_registry.last_image.shape == (10, 10, 3)


def test_run_image_inference_rejects_invalid_content_type() -> None:
    model_registry = FakeModelRegistryService()
    storage_service = FakeStorageService()
    service = _build_inference_service(model_registry, storage_service)

    with pytest.raises(UnsupportedImageTypeError):
        service.run_image_inference(
            filename="test.txt",
            content_type="text/plain",
            image_bytes=b"not-image",
        )

    assert storage_service.saved_uploads == []
    assert storage_service.saved_results == []


def test_run_image_inference_rejects_empty_image() -> None:
    model_registry = FakeModelRegistryService()
    storage_service = FakeStorageService()
    service = _build_inference_service(model_registry, storage_service)

    with pytest.raises(EmptyImageError):
        service.run_image_inference(
            filename="empty.jpg",
            content_type="image/jpeg",
            image_bytes=b"",
        )

    assert storage_service.saved_uploads == []
    assert storage_service.saved_results == []


class FakeModelRegistryService:
    def __init__(self) -> None:
        self.run_inference_calls = 0
        self.last_image: NDArray[np.uint8] | None = None

    def run_inference(
        self,
        image: NDArray[np.uint8],
    ) -> list[DetectionResponse]:
        self.run_inference_calls += 1
        self.last_image = image

        return [_build_detection()]


class FakeStorageService:
    def __init__(self) -> None:
        self.saved_uploads: list[dict[str, Any]] = []
        self.saved_results: list[dict[str, Any]] = []
        self.saved_annotated_images_count = 0

    def create_execution_id(self) -> str:
        return "exec123"

    def save_upload(
        self,
        execution_id: str,
        original_filename: str,
        image_bytes: bytes,
    ) -> None:
        self.saved_uploads.append(
            {
                "execution_id": execution_id,
                "original_filename": original_filename,
                "image_bytes": image_bytes,
            }
        )

    def save_annotated_image(
        self,
        execution_id: str,
        image: NDArray[np.uint8],
    ) -> None:
        self.saved_annotated_images_count += 1

    def save_inference_result(
        self,
        execution_id: str,
        result: dict[str, Any],
    ) -> None:
        self.saved_results.append(
            {
                "execution_id": execution_id,
                "result": result,
            }
        )


def _build_inference_service(
    model_registry: FakeModelRegistryService,
    storage_service: FakeStorageService,
) -> InferenceService:
    return InferenceService(
        model_registry=cast(ModelRegistryService, model_registry),
        storage_service=cast(StorageService, storage_service),
        max_image_size_mb=10,
        api_prefix="/api/v1",
    )


def _build_test_image_bytes() -> bytes:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    success, encoded_image = cv2.imencode(".jpg", image)

    if not success:
        raise RuntimeError("Failed to build test image.")

    return encoded_image.tobytes()


def _build_detection() -> DetectionResponse:
    return DetectionResponse(
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

def test_run_image_inference_rejects_mismatched_image_signature() -> None:
    model_registry = FakeModelRegistryService()
    storage_service = FakeStorageService()
    service = _build_inference_service(model_registry, storage_service)

    with pytest.raises(ImageSignatureMismatchError) as exc_info:
        service.run_image_inference(
            filename="fake.png",
            content_type="image/png",
            image_bytes=b"not-a-real-png",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.message == (
        "Uploaded image content does not match content type: image/png"
    )

    assert storage_service.saved_uploads == []
    assert storage_service.saved_results == []
    assert storage_service.saved_annotated_images_count == 0