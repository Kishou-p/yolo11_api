from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile

from app.core.dependencies import get_inference_service, validate_api_key
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
    file: Annotated[UploadFile, File(...)],
) -> InferenceResponse:
    image_bytes = await file.read()

    return inference_service.run_image_inference(
        filename=file.filename or "uploaded_image",
        content_type=file.content_type,
        image_bytes=image_bytes,
    )