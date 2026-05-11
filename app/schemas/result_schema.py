from pydantic import BaseModel


class ResultSummaryResponse(BaseModel):
    execution_id: str
    filename: str | None = None
    result_url: str
    annotated_image_url: str
    detections_count: int | None = None


class ResultListResponse(BaseModel):
    count: int
    results: list[ResultSummaryResponse]


class ResultDeleteResponse(BaseModel):
    execution_id: str
    deleted: bool
    message: str
