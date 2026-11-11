import json
from pathlib import Path
from typing import Annotated, cast

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import TypeAdapter, ValidationError
from starlette.concurrency import run_in_threadpool

from app.core.config import Settings
from app.core.dependencies import get_settings, validate_api_key
from app.exceptions.app_exceptions import DemoError, ImageTooLargeError
from app.schemas.demo_schema import (
    DemoBox,
    DemoDataset,
    DemoImage,
    DemoPrediction,
    DemoTrainRequest,
    DemoTrainStatus,
)
from app.services.demo_service import DemoService

DEMO_PAGE = Path(__file__).resolve().parent.parent / "static" / "demo.html"
_BOXES_ADAPTER = TypeAdapter(list[DemoBox])

# The page itself is public (it holds no data); every call it makes needs
# the API key, which the user types once on the phone.
page_router = APIRouter()
router = APIRouter(prefix="/demo", dependencies=[Depends(validate_api_key)])


def get_demo_service(request: Request) -> DemoService:
    return cast(DemoService, request.app.state.demo_service)


DemoServiceDep = Annotated[DemoService, Depends(get_demo_service)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


@page_router.get("/demo", include_in_schema=False)
async def demo_page() -> FileResponse:
    return FileResponse(
        DEMO_PAGE,
        media_type="text/html",
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/dataset", response_model=DemoDataset)
async def list_dataset(demo: DemoServiceDep) -> DemoDataset:
    return await run_in_threadpool(demo.list_dataset)


@router.post("/dataset", response_model=DemoImage)
async def add_image(
    demo: DemoServiceDep,
    settings: SettingsDep,
    file: Annotated[UploadFile, File(...)],
    boxes: Annotated[str, Form()] = "[]",
) -> DemoImage:
    try:
        parsed_boxes = _BOXES_ADAPTER.validate_python(json.loads(boxes))
    except (ValueError, ValidationError):
        raise DemoError("Caixas inválidas.", status_code=422) from None

    image_bytes = await _read_upload(file, settings)
    return await run_in_threadpool(
        demo.add_image,
        content_type=file.content_type,
        image_bytes=image_bytes,
        boxes=parsed_boxes,
    )


@router.delete("/dataset")
async def clear_dataset(demo: DemoServiceDep) -> dict[str, int]:
    return {"removed": await run_in_threadpool(demo.clear_dataset)}


@router.get("/dataset/{image_id}/image")
async def get_image(image_id: str, demo: DemoServiceDep) -> FileResponse:
    return FileResponse(demo.image_path(image_id), media_type="image/jpeg")


@router.delete("/dataset/{image_id}", status_code=204)
async def delete_image(image_id: str, demo: DemoServiceDep) -> None:
    await run_in_threadpool(demo.delete_image, image_id)


@router.post("/train", response_model=DemoTrainStatus)
async def start_training(
    demo: DemoServiceDep,
    body: DemoTrainRequest | None = None,
) -> DemoTrainStatus:
    epochs = body.epochs if body else None
    return await run_in_threadpool(demo.start_training, epochs)


@router.post("/train/stop", response_model=DemoTrainStatus)
async def stop_training(demo: DemoServiceDep) -> DemoTrainStatus:
    return demo.stop_training()


@router.get("/train", response_model=DemoTrainStatus)
async def training_status(demo: DemoServiceDep) -> DemoTrainStatus:
    return demo.status()


@router.post("/predict", response_model=DemoPrediction)
async def predict(
    demo: DemoServiceDep,
    settings: SettingsDep,
    file: Annotated[UploadFile, File(...)],
) -> DemoPrediction:
    image_bytes = await _read_upload(file, settings)
    return await run_in_threadpool(
        demo.predict,
        content_type=file.content_type,
        image_bytes=image_bytes,
    )


@router.get("/model")
async def download_model(demo: DemoServiceDep) -> FileResponse:
    if not demo.model_path.is_file():
        raise DemoError("Ainda não há modelo treinado.", status_code=404)
    return FileResponse(
        demo.model_path,
        media_type="application/octet-stream",
        filename="demo_best.pt",
    )


async def _read_upload(file: UploadFile, settings: Settings) -> bytes:
    max_size_bytes = settings.max_image_size_bytes
    image_bytes = await file.read(max_size_bytes + 1)
    if len(image_bytes) > max_size_bytes:
        raise ImageTooLargeError(settings.max_image_size_mb)
    return image_bytes
