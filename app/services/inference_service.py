from app.schemas.inference_schema import InferenceResponse
from app.services.model_registry_service import ModelRegistryService
from app.services.storage_service import StorageService
from app.utils.annotation_utils import draw_detections_on_image
from app.utils.image_utils import (
    decode_image_bytes,
    validate_image_content_type,
    validate_image_signature,
    validate_image_size,
)


class InferenceService:
    def __init__(
        self,
        model_registry: ModelRegistryService,
        storage_service: StorageService,
        max_image_size_mb: int,
        api_prefix: str,
    ) -> None:
        self._model_registry = model_registry
        self._storage_service = storage_service
        self._max_image_size_mb = max_image_size_mb
        self._api_prefix = api_prefix

    def run_image_inference(
        self,
        filename: str,
        content_type: str | None,
        image_bytes: bytes,
    ) -> InferenceResponse:
        validate_image_content_type(content_type)
        validate_image_size(image_bytes, self._max_image_size_mb)
        validate_image_signature(content_type, image_bytes)

        image = decode_image_bytes(image_bytes)
        execution_id = self._storage_service.create_execution_id()

        self._storage_service.save_upload(
            execution_id=execution_id,
            original_filename=filename,
            image_bytes=image_bytes,
        )

        detections = self._model_registry.run_inference(image)
        annotated_image = draw_detections_on_image(image, detections)

        self._storage_service.save_annotated_image(
            execution_id=execution_id,
            image=annotated_image,
        )

        response = InferenceResponse(
            execution_id=execution_id,
            filename=filename,
            result_url=self._build_result_url(execution_id),
            annotated_image_url=self._build_annotated_image_url(execution_id),
            detections_count=len(detections),
            detections=detections,
        )

        self._storage_service.save_inference_result(
            execution_id=execution_id,
            result=response.model_dump(),
        )

        return response

    def _build_result_url(self, execution_id: str) -> str:
        return f"{self._api_prefix}/results/{execution_id}"

    def _build_annotated_image_url(self, execution_id: str) -> str:
        return f"{self._api_prefix}/results/{execution_id}/image"