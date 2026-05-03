from pydantic import BaseModel


class ModelStatusResponse(BaseModel):
    model_path: str
    exists: bool
    loaded: bool


class ModelLoadResponse(BaseModel):
    model_path: str
    loaded: bool
    message: str