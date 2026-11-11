from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class DemoBox(BaseModel):
    """A labelled box, normalised to the image size (0..1, YOLO style)."""

    class_id: int = Field(ge=0)
    x: float = Field(ge=0.0, le=1.0, description="Left edge")
    y: float = Field(ge=0.0, le=1.0, description="Top edge")
    w: float = Field(gt=0.0, le=1.0)
    h: float = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _inside_image(self) -> "DemoBox":
        # Small rounding from the browser is fine; clamp instead of failing.
        self.w = min(self.w, 1.0 - self.x)
        self.h = min(self.h, 1.0 - self.y)
        if self.w <= 0 or self.h <= 0:
            raise ValueError("Box must lie inside the image.")
        return self


class DemoImage(BaseModel):
    image_id: str
    boxes: list[DemoBox]
    created_at: datetime


class DemoDataset(BaseModel):
    classes: list[str]
    images: list[DemoImage]
    counts: dict[str, int] = Field(
        description="Boxes per class, plus 'imagens' (total images)"
    )


class DemoTrainRequest(BaseModel):
    epochs: int | None = Field(default=None, ge=1, le=300)


class DemoTrainStatus(BaseModel):
    state: Literal["idle", "running", "stopping", "done", "error"]
    epoch: int = 0
    epochs: int = 0
    message: str | None = None
    map50: float | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    trained: bool = False
    trained_at: datetime | None = None
    trained_images: int | None = None
    model_size_mb: float | None = None


class DemoDetection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    good: bool
    # Normalised to the image size, like DemoBox.
    x: float
    y: float
    w: float
    h: float


class DemoPrediction(BaseModel):
    verdict: Literal["ok", "defeito", "nada"]
    detections: list[DemoDetection]
    inference_ms: float
    image_width: int
    image_height: int
