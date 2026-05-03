from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "YOLO11 API"
    app_version: str = "0.1.0"
    environment: str = "development"
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"

    yolo_model_path: Path = Path("data/models/yolo11n.pt")
    max_image_size_mb: int = 10

    upload_dir: Path = Path("data/uploads")
    output_dir: Path = Path("data/outputs")

    api_key: str = "dev-secret-key"

    allowed_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )