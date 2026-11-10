import os
import shutil
from pathlib import Path

from app.core.config import Settings
from app.schemas.health_schema import (
    HealthResponse,
    ReadinessChecks,
    ReadinessResponse,
)
from app.services.model_registry_service import ModelRegistryService


class HealthService:
    def __init__(
        self,
        settings: Settings,
        model_registry: ModelRegistryService,
    ) -> None:
        self._settings = settings
        self._model_registry = model_registry

    def get_health(self) -> HealthResponse:
        return HealthResponse(status="ok")

    def get_readiness(self) -> ReadinessResponse:
        checks = ReadinessChecks(
            model_loaded=self._model_registry.is_loaded(),
            storage_writable=all(
                self._is_writable(directory)
                for directory in (
                    self._settings.upload_dir,
                    self._settings.output_dir,
                )
            ),
            disk_space_ok=self._has_free_space(self._settings.output_dir),
        )
        is_ready = all(checks.model_dump().values())

        return ReadinessResponse(
            status="ready" if is_ready else "not_ready",
            checks=checks,
        )

    def _is_writable(self, directory: Path) -> bool:
        return directory.is_dir() and os.access(directory, os.W_OK)

    def _has_free_space(self, directory: Path) -> bool:
        try:
            free_bytes = shutil.disk_usage(directory).free
        except OSError:
            return False

        return free_bytes >= self._settings.min_free_disk_mb * 1024 * 1024
