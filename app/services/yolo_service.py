import hashlib
import hmac
import logging
from pathlib import Path
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO

from app.exceptions.app_exceptions import (
    ModelChecksumMismatchError,
    ModelFileNotFoundError,
    ModelInferenceError,
    ModelLoadError,
    ModelTrainingError,
)
from app.schemas.inference_schema import (
    BoundingBox,
    DetectionResponse,
    InferenceParams,
    ModelInfo,
)

logger = logging.getLogger(__name__)


class YoloService:
    def __init__(
        self,
        model_path: Path,
        expected_sha256: str | None = None,
        params: InferenceParams | None = None,
    ) -> None:
        self._model_path = model_path
        self._expected_sha256 = (
            expected_sha256.strip().lower() if expected_sha256 else None
        )
        self._params = params or InferenceParams()
        self._loaded_sha256: str | None = None

    @property
    def params(self) -> InferenceParams:
        return self._params

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(name=self._model_path.name, sha256=self._loaded_sha256)

    @property
    def model_path(self) -> Path:
        return self._model_path

    def model_exists(self) -> bool:
        return self._model_path.exists() and self._model_path.is_file()

    def load_model(self) -> Any:
        self._ensure_model_file_exists()
        self._verify_checksum()
        return self._build_model()

    def predict(
        self,
        model: Any,
        image: NDArray[np.uint8],
    ) -> list[DetectionResponse]:
        try:
            results = model.predict(
                source=image,
                verbose=False,
                **self._predict_kwargs(),
            )
        except (RuntimeError, OSError, ValueError, TypeError) as exc:
            logger.exception("Failed to run YOLO inference.")
            raise ModelInferenceError() from exc

        return self._parse_prediction_results(results)

    def train(
        self,
        model: Any,
        *,
        data_yaml: Path,
        output_dir: Path,
        epochs: int,
        imgsz: int,
        batch: int,
        on_epoch_end: Callable[[int, int, dict[str, float]], bool] | None = None,
    ) -> Path:
        """Fine-tune `model` on a YOLO dataset and return the best weights.

        `on_epoch_end(epoch, epochs, metrics)` is called after each epoch;
        returning True stops training early (the best weights so far are
        kept). Settings favour small datasets on CPU: no worker processes
        (Docker's /dev/shm is tiny), no plots and no AMP check download.
        """

        def _callback(trainer: Any) -> None:
            if on_epoch_end is None:
                return
            metrics = {
                str(key): float(value)
                for key, value in (getattr(trainer, "metrics", None) or {}).items()
                if isinstance(value, (int, float))
            }
            if on_epoch_end(int(trainer.epoch) + 1, int(trainer.epochs), metrics):
                trainer.stop = True

        model.add_callback("on_fit_epoch_end", _callback)

        try:
            model.train(
                data=str(data_yaml),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=self._params.device,
                workers=0,
                project=str(output_dir),
                name="train",
                exist_ok=True,
                plots=False,
                amp=False,
                cache=False,
                patience=max(epochs, 50),
                verbose=False,
            )
        except Exception as exc:
            logger.exception("YOLO training failed.")
            raise ModelTrainingError() from exc

        best = Path(str(getattr(model.trainer, "best", "")))
        if not best.is_file():
            raise ModelTrainingError()

        return best

    def _predict_kwargs(self) -> dict[str, Any]:
        kwargs = self._params.model_dump(exclude_none=True)

        # Recent Ultralytics versions warn that `half` is deprecated, so
        # only send it when half precision was actually requested.
        if not kwargs.get("half"):
            kwargs.pop("half", None)

        return kwargs

    def _ensure_model_file_exists(self) -> None:
        if not self.model_exists():
            logger.warning("YOLO model file not found: %s", self._model_path)
            raise ModelFileNotFoundError(self._model_path)

    def _verify_checksum(self) -> None:
        # .pt files are pickles: loading a tampered file runs arbitrary code,
        # so the hash is checked before the file reaches torch.
        actual_sha256 = sha256_file(self._model_path)
        self._loaded_sha256 = actual_sha256

        if self._expected_sha256 is None:
            logger.warning(
                "YOLO_MODEL_SHA256 is not set; model integrity is not checked "
                "(current sha256: %s).",
                actual_sha256,
            )
            return

        if not hmac.compare_digest(actual_sha256, self._expected_sha256):
            logger.error(
                "Model checksum mismatch for %s: expected %s, got %s",
                self._model_path,
                self._expected_sha256,
                actual_sha256,
            )
            self._loaded_sha256 = None
            raise ModelChecksumMismatchError(self._model_path)

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

def sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()

    with file_path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()
