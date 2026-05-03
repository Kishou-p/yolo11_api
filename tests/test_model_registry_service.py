from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from app.exceptions.app_exceptions import ModelNotLoadedError
from app.schemas.inference_schema import BoundingBox, DetectionResponse
from app.services.model_registry_service import ModelRegistryService
from app.services.yolo_service import YoloService


def test_model_path_returns_yolo_service_model_path() -> None:
    yolo_service = _build_fake_yolo_service()
    registry = ModelRegistryService(
        yolo_service=cast(YoloService, yolo_service),
    )

    assert registry.model_path == Path("fake_model.pt")


def test_model_exists_delegates_to_yolo_service() -> None:
    yolo_service = _build_fake_yolo_service(model_exists=True)
    registry = ModelRegistryService(
        yolo_service=cast(YoloService, yolo_service),
    )

    assert registry.model_exists() is True


def test_registry_is_not_loaded_initially() -> None:
    yolo_service = _build_fake_yolo_service()
    registry = ModelRegistryService(
        yolo_service=cast(YoloService, yolo_service),
    )

    assert registry.is_loaded() is False


def test_load_model_loads_and_caches_model() -> None:
    yolo_service = _build_fake_yolo_service()
    registry = ModelRegistryService(
        yolo_service=cast(YoloService, yolo_service),
    )

    first_model = registry.load_model()
    second_model = registry.load_model()

    assert first_model is second_model
    assert registry.is_loaded() is True
    assert yolo_service.load_calls == 1


def test_get_loaded_model_raises_when_model_is_not_loaded() -> None:
    yolo_service = _build_fake_yolo_service()
    registry = ModelRegistryService(
        yolo_service=cast(YoloService, yolo_service),
    )

    with pytest.raises(ModelNotLoadedError) as exc_info:
        registry.get_loaded_model()

    assert exc_info.value.status_code == 409
    assert exc_info.value.message == "YOLO model is not loaded: fake_model.pt"


def test_run_inference_raises_when_model_is_not_loaded() -> None:
    yolo_service = _build_fake_yolo_service()
    registry = ModelRegistryService(
        yolo_service=cast(YoloService, yolo_service),
    )
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(ModelNotLoadedError):
        registry.run_inference(image)


def test_run_inference_uses_loaded_model() -> None:
    yolo_service = _build_fake_yolo_service()
    registry = ModelRegistryService(
        yolo_service=cast(YoloService, yolo_service),
    )
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    loaded_model = registry.load_model()
    detections = registry.run_inference(image)

    assert yolo_service.predict_calls == 1
    assert yolo_service.last_model is loaded_model
    assert yolo_service.last_image is image
    assert detections == [_build_detection()]


class FakeYoloService:
    def __init__(self, model_exists: bool = True) -> None:
        self._model_path = Path("fake_model.pt")
        self._model_exists = model_exists
        self._model = object()
        self.load_calls = 0
        self.predict_calls = 0
        self.last_model: Any | None = None
        self.last_image: NDArray[np.uint8] | None = None

    @property
    def model_path(self) -> Path:
        return self._model_path

    def model_exists(self) -> bool:
        return self._model_exists

    def load_model(self) -> Any:
        self.load_calls += 1
        return self._model

    def predict(
        self,
        model: Any,
        image: NDArray[np.uint8],
    ) -> list[DetectionResponse]:
        self.predict_calls += 1
        self.last_model = model
        self.last_image = image

        return [_build_detection()]


def _build_fake_yolo_service(model_exists: bool = True) -> FakeYoloService:
    return FakeYoloService(model_exists=model_exists)


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