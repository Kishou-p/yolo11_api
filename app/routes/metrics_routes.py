from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.core.dependencies import get_model_registry_service, validate_api_key
from app.core.metrics import Metrics
from app.services.model_registry_service import ModelRegistryService

router = APIRouter()


@router.get(
    "/metrics",
    dependencies=[Depends(validate_api_key)],
    include_in_schema=False,
)
def get_metrics(
    request: Request,
    model_registry: Annotated[
        ModelRegistryService,
        Depends(get_model_registry_service),
    ],
) -> Response:
    metrics: Metrics = request.app.state.metrics
    metrics.model_loaded.set(1 if model_registry.is_loaded() else 0)

    return Response(
        content=generate_latest(metrics.registry),
        media_type=CONTENT_TYPE_LATEST,
    )
