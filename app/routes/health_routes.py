from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.core.dependencies import get_health_service
from app.schemas.health_schema import HealthResponse, ReadinessResponse
from app.services.health_service import HealthService

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
@router.get("/health/live", response_model=HealthResponse)
def health_check(
    health_service: Annotated[HealthService, Depends(get_health_service)],
) -> HealthResponse:
    return health_service.get_health()


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    responses={503: {"model": ReadinessResponse}},
)
def readiness_check(
    health_service: Annotated[HealthService, Depends(get_health_service)],
) -> JSONResponse:
    readiness = health_service.get_readiness()
    status_code = 200 if readiness.status == "ready" else 503

    return JSONResponse(
        status_code=status_code,
        content=readiness.model_dump(),
    )
