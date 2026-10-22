from typing import Annotated

from fastapi import APIRouter, Depends

from app.core.dependencies import get_model_registry_service, validate_api_key
from app.schemas.model_schema import ModelLoadResponse, ModelStatusResponse
from app.services.model_registry_service import ModelRegistryService

router = APIRouter()


@router.get(
    "/models/status",
    response_model=ModelStatusResponse,
    dependencies=[Depends(validate_api_key)],
)
def get_model_status(
    model_registry: Annotated[
        ModelRegistryService,
        Depends(get_model_registry_service),
    ],
) -> ModelStatusResponse:
    return ModelStatusResponse(
        model_name=model_registry.model_path.name,
        exists=model_registry.model_exists(),
        loaded=model_registry.is_loaded(),
    )


@router.post(
    "/models/load",
    response_model=ModelLoadResponse,
    dependencies=[Depends(validate_api_key)],
)
def load_model(
    model_registry: Annotated[
        ModelRegistryService,
        Depends(get_model_registry_service),
    ],
) -> ModelLoadResponse:
    model_registry.load_model()

    return ModelLoadResponse(
        model_name=model_registry.model_path.name,
        loaded=model_registry.is_loaded(),
        message="YOLO model loaded successfully.",
    )