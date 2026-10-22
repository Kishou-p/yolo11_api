from pathlib import Path
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

WEAK_API_KEYS = frozenset({"", "change-me", "dev-secret-key"})
MIN_API_KEY_LENGTH = 32
INSECURE_ENVIRONMENTS = frozenset({"development", "test"})


class Settings(BaseSettings):
    app_name: str = "YOLO11 API"
    app_version: str = "0.1.0"
    environment: Literal["development", "test", "staging", "production"] = (
        "production"
    )
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"

    yolo_model_path: Path = Path("data/models/yolo11n.pt")
    max_image_size_mb: int = 10
    max_image_pixels: int = 40_000_000

    upload_dir: Path = Path("data/uploads")
    output_dir: Path = Path("data/outputs")

    api_key: SecretStr = SecretStr("dev-secret-key")
    extra_api_keys: list[SecretStr] = []

    allowed_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def is_production_like(self) -> bool:
        return self.environment not in INSECURE_ENVIRONMENTS

    @property
    def max_image_size_bytes(self) -> int:
        return self.max_image_size_mb * 1024 * 1024

    def valid_api_keys(self) -> list[str]:
        keys = [self.api_key, *self.extra_api_keys]
        return [key.get_secret_value() for key in keys if key.get_secret_value()]

    @model_validator(mode="after")
    def _validate_api_keys(self) -> "Settings":
        if not self.valid_api_keys():
            raise ValueError("API_KEY must be set.")

        if not self.is_production_like:
            return self

        for key in self.valid_api_keys():
            if key in WEAK_API_KEYS or len(key) < MIN_API_KEY_LENGTH:
                raise ValueError(
                    "API keys must have at least "
                    f"{MIN_API_KEY_LENGTH} characters and must not use "
                    "example values outside development/test."
                )

        return self
