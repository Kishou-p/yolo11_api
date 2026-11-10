import logging
import time
from datetime import UTC, datetime

from app.schemas.inference_schema import InferenceResponse, InferenceTimings
from app.services.model_registry_service import ModelRegistryService
from app.services.storage_service import StorageService
from app.utils.annotation_utils import draw_detections_on_image
from app.utils.image_utils import (
    decode_image_bytes,
    validate_image_content_type,
    validate_image_dimensions,
    validate_image_signature,
    validate_image_size,
)

logger = logging.getLogger(__name__)

MAX_FILENAME_LENGTH = 255


class InferenceService:
    def __init__(
        self,
        model_registry: ModelRegistryService,
        storage_service: StorageService,
        max_image_size_mb: int,
        api_prefix: str,
        max_image_pixels: int = 40_000_000,
        save_uploads: bool = True,
        save_annotated: bool = True,
    ) -> None:
        self._model_registry = model_registry
        self._storage_service = storage_service
        self._max_image_size_mb = max_image_size_mb
        self._max_image_pixels = max_image_pixels
        self._api_prefix = api_prefix
        self._save_uploads = save_uploads
        self._save_annotated = save_annotated

    def run_image_inference(
        self,
        filename: str,
        content_type: str | None,
        image_bytes: bytes,
    ) -> InferenceResponse:
        validate_image_content_type(content_type)
        validate_image_size(image_bytes, self._max_image_size_mb)
        validate_image_signature(content_type, image_bytes)
        validate_image_dimensions(image_bytes, self._max_image_pixels)

        started_at = time.perf_counter()
        filename = filename[:MAX_FILENAME_LENGTH]
        image = decode_image_bytes(image_bytes)
        decoded_at = time.perf_counter()
        detections = self._model_registry.run_inference(image)
        inferred_at = time.perf_counter()
        execution_id = self._storage_service.create_execution_id()

        response = InferenceResponse(
            execution_id=execution_id,
            filename=filename,
            result_url=self._build_result_url(execution_id),
            annotated_image_url=(
                self._build_annotated_image_url(execution_id)
                if self._save_annotated
                else None
            ),
            detections_count=len(detections),
            detections=detections,
            created_at=datetime.now(UTC),
            model=self._model_registry.model_info,
            params=self._model_registry.params,
        )
        timings = InferenceTimings(
            decode_ms=_elapsed_ms(started_at, decoded_at),
            inference_ms=_elapsed_ms(decoded_at, inferred_at),
            total_ms=0.0,
        )

        # Nothing is written until inference succeeds; if a write fails,
        # remove what was already saved so no orphan files are left behind.
        try:
            if self._save_uploads:
                self._storage_service.save_upload(
                    execution_id=execution_id,
                    content_type=content_type,
                    image_bytes=image_bytes,
                )
            if self._save_annotated:
                self._storage_service.save_annotated_image(
                    execution_id=execution_id,
                    image=draw_detections_on_image(image, detections),
                )

            timings.total_ms = _elapsed_ms(started_at, time.perf_counter())
            response.timings = timings
            self._storage_service.save_inference_result(
                execution_id=execution_id,
                result=response.model_dump(mode="json"),
            )
        except Exception:
            self._storage_service.delete_execution_files(execution_id)
            raise

        logger.info(
            "Inference completed",
            extra={
                "execution_id": execution_id,
                "detections_count": response.detections_count,
                "decode_ms": timings.decode_ms,
                "inference_ms": timings.inference_ms,
                "total_ms": timings.total_ms,
            },
        )

        return response

    def _build_result_url(self, execution_id: str) -> str:
        return f"{self._api_prefix}/results/{execution_id}"

    def _build_annotated_image_url(self, execution_id: str) -> str:
        return f"{self._api_prefix}/results/{execution_id}/image"

def _elapsed_ms(start: float, end: float) -> float:
    return round((end - start) * 1000, 2)
