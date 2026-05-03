from app.core.config import Settings
from app.schemas.health_schema import HealthResponse


class HealthService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def get_health(self) -> HealthResponse:
        return HealthResponse(
            status="ok",
            app_name=self._settings.app_name,
            version=self._settings.app_version,
            environment=self._settings.environment,
        )