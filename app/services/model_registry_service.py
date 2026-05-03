from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.exceptions.app_exceptions import ModelNotLoadedError
from app.schemas.inference_schema import DetectionResponse
from app.services.yolo_service import YoloService


class ModelRegistryService:
    def __init__(
        self,
        yolo_service: YoloService,
        loaded_model: Any | None = None,
    ) -> None:
        self._yolo_service = yolo_service
        self._loaded_model = loaded_model

    @property
    def model_path(self) -> Path:
        return self._yolo_service.model_path

    def model_exists(self) -> bool:
        return self._yolo_service.model_exists()

    def is_loaded(self) -> bool:
        return self._loaded_model is not None

    def load_model(self) -> Any:
        if self.is_loaded():
            return self._loaded_model

        self._loaded_model = self._yolo_service.load_model()
        return self._loaded_model

    def get_loaded_model(self) -> Any:
        if self._loaded_model is None:
            raise ModelNotLoadedError(self.model_path)

        return self._loaded_model

    def run_inference(
        self,
        image: NDArray[np.uint8],
    ) -> list[DetectionResponse]:
        loaded_model = self.get_loaded_model()
        return self._yolo_service.predict(loaded_model, image)