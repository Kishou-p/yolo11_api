from datetime import datetime

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionResponse(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    box: BoundingBox


class InferenceParams(BaseModel):
    conf: float = 0.25
    iou: float = 0.7
    imgsz: int = 640
    max_det: int = 300
    device: str | None = None
    half: bool = False


class ModelInfo(BaseModel):
    name: str
    sha256: str | None = None


class InferenceTimings(BaseModel):
    decode_ms: float
    inference_ms: float
    total_ms: float


class InferenceResponse(BaseModel):
    execution_id: str
    filename: str
    result_url: str
    annotated_image_url: str | None
    detections_count: int
    detections: list[DetectionResponse]
    created_at: datetime | None = None
    model: ModelInfo | None = None
    params: InferenceParams | None = None
    timings: InferenceTimings | None = None