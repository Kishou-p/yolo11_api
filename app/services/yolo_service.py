import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO

from app.exceptions.app_exceptions import (
    ModelFileNotFoundError,
    ModelInferenceError,
    ModelLoadError,
)
from app.schemas.inference_schema import BoundingBox, DetectionResponse

logger = logging.getLogger(__name__)


class YoloService:
    def __init__(self, model_path: Path) -> None:
        self._model_path = model_path

    @property
    def model_path(self) -> Path:
        return self._model_path

    def model_exists(self) -> bool:
        return self._model_path.exists() and self._model_path.is_file()

    def load_model(self) -> Any:
        self._ensure_model_file_exists()
        return self._build_model()

    def predict(
        self,
        model: Any,
        image: NDArray[np.uint8],
    ) -> list[DetectionResponse]:
        try:
            results = model.predict(source=image, verbose=False)
        except (RuntimeError, OSError, ValueError, TypeError) as exc:
            logger.exception("Failed to run YOLO inference.")
            raise ModelInferenceError() from exc

        return self._parse_prediction_results(results)

    def _ensure_model_file_exists(self) -> None:
        if not self.model_exists():
            logger.warning("YOLO model file not found: %s", self._model_path)
            raise ModelFileNotFoundError(self._model_path)

    def _build_model(self) -> Any:
        try:
            logger.info("Loading YOLO model from: %s", self._model_path)
            return YOLO(str(self._model_path))
        except (RuntimeError, OSError, ValueError) as exc:
            logger.exception("Failed to load YOLO model: %s", self._model_path)
            raise ModelLoadError(self._model_path) from exc

    def _parse_prediction_results(
        self,
        results: Any,
    ) -> list[DetectionResponse]:
        if not results:
            return []

        first_result = results[0]
        return self._parse_result_boxes(first_result)

    def _parse_result_boxes(self, result: Any) -> list[DetectionResponse]:
        boxes = getattr(result, "boxes", None)

        if boxes is None:
            return []

        names = getattr(result, "names", {})
        return [self._build_detection(box, names) for box in boxes]

    def _build_detection(
        self,
        box: Any,
        names: dict[int, str],
    ) -> DetectionResponse:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = self._extract_xyxy(box)
        class_name = str(names.get(class_id, f"class_{class_id}"))

        return DetectionResponse(
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        )

    def _extract_xyxy(self, box: Any) -> tuple[float, float, float, float]:
        coordinates = box.xyxy[0].tolist()
        x1, y1, x2, y2 = coordinates

        return float(x1), float(y1), float(x2), float(y2)