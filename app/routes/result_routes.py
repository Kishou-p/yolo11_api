from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from app.core.dependencies import get_result_service, validate_api_key
from app.schemas.result_schema import ResultDeleteResponse, ResultListResponse
from app.services.result_service import ResultService

router = APIRouter(
    prefix="/results",
    tags=["Results"],
    dependencies=[Depends(validate_api_key)],
)


@router.get("", response_model=ResultListResponse)
def list_results(
    result_service: Annotated[ResultService, Depends(get_result_service)],
) -> ResultListResponse:
    return result_service.list_results()


@router.get("/{execution_id}")
def get_result(
    execution_id: str,
    result_service: Annotated[ResultService, Depends(get_result_service)],
) -> dict:
    return result_service.get_result_data(execution_id)


@router.get("/{execution_id}/image")
def get_annotated_image(
    execution_id: str,
    result_service: Annotated[ResultService, Depends(get_result_service)],
) -> FileResponse:
    annotated_image_path = result_service.get_annotated_image_path(
        execution_id,
    )

    return FileResponse(
        path=annotated_image_path,
        media_type="image/png",
        filename=annotated_image_path.name,
    )


@router.delete("/{execution_id}", response_model=ResultDeleteResponse)
def delete_result(
    execution_id: str,
    result_service: Annotated[ResultService, Depends(get_result_service)],
) -> ResultDeleteResponse:
    return result_service.delete_result(execution_id)
