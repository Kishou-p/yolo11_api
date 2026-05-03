from typing import Annotated

from fastapi import APIRouter, Depends

from app.core.dependencies import get_health_service
from app.schemas.health_schema import HealthResponse
from app.services.health_service import HealthService

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check(
    health_service: Annotated[HealthService, Depends(get_health_service)],
) -> HealthResponse:
    return health_service.get_health()