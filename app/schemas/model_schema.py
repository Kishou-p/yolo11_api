from pydantic import BaseModel


class ModelStatusResponse(BaseModel):
    model_name: str
    exists: bool
    loaded: bool


class ModelLoadResponse(BaseModel):
    model_name: str
    loaded: bool
    message: str