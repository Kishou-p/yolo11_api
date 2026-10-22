from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile

from app.core.config import Settings
from app.core.dependencies import (
    get_inference_service,
    get_settings,
    validate_api_key,
)
from app.exceptions.app_exceptions import ImageTooLargeError
from app.schemas.inference_schema import InferenceResponse
from app.services.inference_service import InferenceService

router = APIRouter()


@router.post(
    "/inference/image",
    response_model=InferenceResponse,
    dependencies=[Depends(validate_api_key)],
)
async def run_image_inference(
    inference_service: Annotated[
        InferenceService,
        Depends(get_inference_service),
    ],
    settings: Annotated[Settings, Depends(get_settings)],
    file: Annotated[UploadFile, File(...)],
) -> InferenceResponse:
    max_size_bytes = settings.max_image_size_bytes
    image_bytes = await file.read(max_size_bytes + 1)

    if len(image_bytes) > max_size_bytes:
        raise ImageTooLargeError(settings.max_image_size_mb)

    return inference_service.run_image_inference(
        filename=file.filename or "uploaded_image",
        content_type=file.content_type,
        image_bytes=image_bytes,
    )