from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.schemas.inference_schema import InferenceParams

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
    log_format: Literal["text", "json"] = "text"
    metrics_enabled: bool = True
    api_prefix: str = "/api/v1"

    yolo_model_path: Path = Path("data/models/yolo11n.pt")
    yolo_model_sha256: str | None = None
    load_model_on_startup: bool = True

    # Inference parameters (Ultralytics predict arguments). They are process
    # decisions: every result records the values that produced it.
    yolo_conf: float = Field(default=0.25, ge=0.0, le=1.0)
    yolo_iou: float = Field(default=0.7, ge=0.0, le=1.0)
    yolo_imgsz: int = Field(default=640, ge=32, le=4096)
    yolo_max_det: int = Field(default=300, ge=1, le=10000)
    yolo_device: str | None = None
    yolo_half: bool = False

    max_image_size_mb: int = 10
    max_image_pixels: int = 40_000_000

    upload_dir: Path = Path("data/uploads")
    output_dir: Path = Path("data/outputs")
    min_free_disk_mb: int = 500
    save_uploads: bool = True
    save_annotated: bool = True
    annotated_image_format: Literal["png", "jpg"] = "png"
    # 0 disables automatic cleanup of uploads and results.
    retention_days: int = Field(default=0, ge=0)
    retention_interval_minutes: int = Field(default=60, ge=1)

    # Demo web app (/demo): label photos, fine-tune and test in the browser.
    demo_enabled: bool = True
    demo_dir: Path = Path("data/demo")
    # The first class is the part itself; every other class marks a defect
    # (e.g. the spot where a component is missing).
    demo_classes: list[str] = ["peca", "falta"]
    demo_train_epochs: int = Field(default=30, ge=1, le=300)
    demo_train_imgsz: int = Field(default=512, ge=128, le=1280)
    demo_train_batch: int = Field(default=8, ge=1, le=64)
    demo_min_images: int = Field(default=4, ge=1)
    # Each photo shows one part: keep only the most confident part box and
    # the defects inside it.
    demo_single_part: bool = True

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

    @property
    def inference_params(self) -> InferenceParams:
        return InferenceParams(
            conf=self.yolo_conf,
            iou=self.yolo_iou,
            imgsz=self.yolo_imgsz,
            max_det=self.yolo_max_det,
            device=self.yolo_device,
            half=self.yolo_half,
        )

    def valid_api_keys(self) -> list[str]:
        keys = [self.api_key, *self.extra_api_keys]
        return [key.get_secret_value() for key in keys if key.get_secret_value()]

    @field_validator("yolo_device", "yolo_model_sha256", mode="before")
    @classmethod
    def _empty_string_as_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("demo_classes")
    @classmethod
    def _validate_demo_classes(cls, value: list[str]) -> list[str]:
        names = [name.strip() for name in value]
        if len(names) < 2 or any(not name for name in names):
            raise ValueError("DEMO_CLASSES needs at least two non-empty names.")
        if len(set(names)) != len(names):
            raise ValueError("DEMO_CLASSES must not repeat names.")
        return names

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
