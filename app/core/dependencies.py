import hmac
from typing import Annotated, cast

from fastapi import Depends, Header, Request
from app.core.config import Settings
from app.core.metrics import Metrics
from app.services.health_service import HealthService
from app.services.inference_service import InferenceService
from app.services.result_service import ResultService
from app.services.model_registry_service import ModelRegistryService
from app.services.storage_service import StorageService
from app.exceptions.app_exceptions import UnauthorizedApiKeyError


def get_settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)

def get_metrics(request: Request) -> Metrics:
    return cast(Metrics, request.app.state.metrics)


def validate_api_key(
    settings: Annotated[Settings, Depends(get_settings)],
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    if not x_api_key or not _matches_any_api_key(
        x_api_key,
        settings.valid_api_keys(),
    ):
        raise UnauthorizedApiKeyError()


def _matches_any_api_key(candidate: str, valid_keys: list[str]) -> bool:
    candidate_bytes = candidate.encode("utf-8")
    matched = False

    for valid_key in valid_keys:
        if hmac.compare_digest(candidate_bytes, valid_key.encode("utf-8")):
            matched = True

    return matched


def get_model_registry_service(request: Request) -> ModelRegistryService:
    return cast(ModelRegistryService, request.app.state.model_registry)


def get_health_service(
    settings: Annotated[Settings, Depends(get_settings)],
    model_registry: Annotated[
        ModelRegistryService,
        Depends(get_model_registry_service),
    ],
) -> HealthService:
    return HealthService(settings=settings, model_registry=model_registry)


def get_storage_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> StorageService:
    return StorageService(
        upload_dir=settings.upload_dir,
        output_dir=settings.output_dir,
        annotated_format=settings.annotated_image_format,
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
        max_image_pixels=settings.max_image_pixels,
        save_uploads=settings.save_uploads,
        save_annotated=settings.save_annotated,
        api_prefix=settings.api_prefix,
    )


def get_result_service(
    settings: Annotated[Settings, Depends(get_settings)],
    storage_service: Annotated[
        StorageService,
        Depends(get_storage_service),
    ],
) -> ResultService:
    return ResultService(
        storage_service=storage_service,
        api_prefix=settings.api_prefix,
    )
