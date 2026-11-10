from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class ReadinessChecks(BaseModel):
    model_loaded: bool
    storage_writable: bool
    disk_space_ok: bool


class ReadinessResponse(BaseModel):
    status: str
    checks: ReadinessChecks
