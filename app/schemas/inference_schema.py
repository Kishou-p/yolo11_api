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


class InferenceResponse(BaseModel):
    execution_id: str
    filename: str
    result_url: str
    annotated_image_url: str
    detections_count: int
    detections: list[DetectionResponse]