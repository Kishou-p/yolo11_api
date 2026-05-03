from app.services.result_service import ResultService

from typing import Annotated, cast
from fastapi import Depends, Header, Request
from app.core.config import Settings
from app.services.health_service import HealthService
from app.services.inference_service import InferenceService
from app.services.model_registry_service import ModelRegistryService
from app.services.storage_service import StorageService
from app.services.yolo_service import YoloService
from app.exceptions.app_exceptions import UnauthorizedApiKeyError


def get_settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)

def validate_api_key(
    settings: Annotated[Settings, Depends(get_settings)],
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    if x_api_key != settings.api_key:
        raise UnauthorizedApiKeyError()


def get_health_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> HealthService:
    return HealthService(settings=settings)


def get_yolo_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> YoloService:
    return YoloService(model_path=settings.yolo_model_path)


def get_model_registry_service(
    request: Request,
    yolo_service: Annotated[YoloService, Depends(get_yolo_service)],
) -> ModelRegistryService:
    current_registry = request.app.state.model_registry

    if isinstance(current_registry, ModelRegistryService):
        return current_registry

    new_registry = ModelRegistryService(yolo_service=yolo_service)
    request.app.state.model_registry = new_registry

    return new_registry


def get_storage_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> StorageService:
    return StorageService(
        upload_dir=settings.upload_dir,
        output_dir=settings.output_dir,
    )


def get_inference_service(
    settings: Annotated[Settings, Depends(get_settings)],
    model_registry: Annotated[
        ModelRegistryService,
        Depends(get_model_registry_service),
    ],
    storage_service: Annotated[
        StorageService,
        Depends(get_storage_service),
    ],
) -> InferenceService:
    return InferenceService(
        model_registry=model_registry,
        storage_service=storage_service,
        max_image_size_mb=settings.max_image_size_mb,
        api_prefix=settings.api_prefix,
    )


def get_result_service(
    storage_service: Annotated[
        StorageService,
        Depends(get_storage_service),
    ],
) -> ResultService:
    return ResultService(storage_service=storage_service)