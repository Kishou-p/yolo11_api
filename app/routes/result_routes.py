from typing import Annotated, Any

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from app.core.dependencies import get_result_service, validate_api_key
from app.services.result_service import ResultService

router = APIRouter()


@router.get(
    "/results/{execution_id}",
    dependencies=[Depends(validate_api_key)],
)
def get_inference_result(
    execution_id: str,
    result_service: Annotated[ResultService, Depends(get_result_service)],
) -> dict[str, Any]:
    return result_service.get_result_data(execution_id)


@router.get(
    "/results/{execution_id}/image",
    dependencies=[Depends(validate_api_key)],
)
def get_annotated_image(
    execution_id: str,
    result_service: Annotated[ResultService, Depends(get_result_service)],
) -> FileResponse:
    annotated_path = result_service.get_annotated_image_path(execution_id)

    return FileResponse(
        path=annotated_path,
        media_type="image/png",
        filename=annotated_path.name,
    )