from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.config import Settings
from app.core.dependencies import (
    get_inference_service,
    get_metrics,
    get_settings,
    validate_api_key,
)
from app.core.metrics import Metrics
from app.exceptions.app_exceptions import AppError, ImageTooLargeError
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
    metrics: Annotated[Metrics, Depends(get_metrics)],
    file: Annotated[UploadFile, File(...)],
) -> InferenceResponse:
    try:
        response = await _run_inference(inference_service, settings, file)
    except AppError as exc:
        outcome = "rejected" if exc.status_code < 500 else "error"
        metrics.inferences.labels(outcome=outcome).inc()
        raise
    except Exception:
        metrics.inferences.labels(outcome="error").inc()
        raise

    _record_success(metrics, response)
    return response


async def _run_inference(
    inference_service: InferenceService,
    settings: Settings,
    file: UploadFile,
) -> InferenceResponse:
    max_size_bytes = settings.max_image_size_bytes
    image_bytes = await file.read(max_size_bytes + 1)

    if len(image_bytes) > max_size_bytes:
        raise ImageTooLargeError(settings.max_image_size_mb)

    # Decoding, YOLO and disk writes are blocking: keep them off the event loop.
    return await run_in_threadpool(
        inference_service.run_image_inference,
        filename=file.filename or "uploaded_image",
        content_type=file.content_type,
        image_bytes=image_bytes,
    )


def _record_success(metrics: Metrics, response: InferenceResponse) -> None:
    metrics.inferences.labels(outcome="success").inc()
    metrics.detections.observe(response.detections_count)

    if response.timings is None:
        return

    for stage, value_ms in (
        ("decode", response.timings.decode_ms),
        ("inference", response.timings.inference_ms),
        ("total", response.timings.total_ms),
    ):
        metrics.inference_stage_duration.labels(stage=stage).observe(
            value_ms / 1000
        )