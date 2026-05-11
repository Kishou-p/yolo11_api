import json
from pathlib import Path
from typing import Any

from app.exceptions.app_exceptions import ResultNotFoundError
from app.schemas.result_schema import (
    ResultDeleteResponse,
    ResultListResponse,
    ResultSummaryResponse,
)
from app.services.storage_service import StorageService
from app.utils.execution_id_utils import validate_execution_id


class ResultService:
    def __init__(
        self,
        storage_service: StorageService,
        api_prefix: str = "/api/v1",
    ) -> None:
        self._storage_service = storage_service
        self._api_prefix = api_prefix

    def list_results(self) -> ResultListResponse:
        results: list[ResultSummaryResponse] = []

        for result_path in self._storage_service.list_result_paths():
            try:
                result_data = self._read_json_file(result_path)
            except ResultNotFoundError:
                continue

            results.append(
                self._build_result_summary(
                    result_path=result_path,
                    result_data=result_data,
                )
            )

        return ResultListResponse(
            count=len(results),
            results=results,
        )

    def get_result_data(self, execution_id: str) -> dict[str, Any]:
        validate_execution_id(execution_id)

        result_path = self._storage_service.build_result_path(execution_id)

        if not result_path.exists():
            raise ResultNotFoundError(execution_id)

        return self._read_json_file(result_path)

    def get_annotated_image_path(self, execution_id: str) -> Path:
        validate_execution_id(execution_id)

        annotated_path = self._storage_service.build_annotated_path(
            execution_id,
        )

        if not annotated_path.exists():
            raise ResultNotFoundError(execution_id)

        return annotated_path

    def delete_result(self, execution_id: str) -> ResultDeleteResponse:
        validate_execution_id(execution_id)

        result_path = self._storage_service.build_result_path(execution_id)

        if not result_path.exists():
            raise ResultNotFoundError(execution_id)

        filename = self._get_filename_from_result(result_path)

        self._storage_service.delete_file_if_exists(result_path)
        self._storage_service.delete_file_if_exists(
            self._storage_service.build_annotated_path(execution_id)
        )

        if filename:
            upload_path = self._storage_service.build_upload_path(
                execution_id=execution_id,
                original_filename=filename,
            )
            self._storage_service.delete_file_if_exists(upload_path)

        return ResultDeleteResponse(
            execution_id=execution_id,
            deleted=True,
            message="Result deleted successfully.",
        )

    def _build_result_summary(
        self,
        result_path: Path,
        result_data: dict[str, Any],
    ) -> ResultSummaryResponse:
        execution_id = str(result_data.get("execution_id") or result_path.stem)

        return ResultSummaryResponse(
            execution_id=execution_id,
            filename=self._get_optional_str(result_data, "filename"),
            result_url=self._get_public_result_url(result_data, execution_id),
            annotated_image_url=self._get_public_annotated_image_url(
                result_data,
                execution_id,
            ),
            detections_count=self._get_optional_int(
                result_data,
                "detections_count",
            ),
        )

    def _get_filename_from_result(self, result_path: Path) -> str | None:
        try:
            result_data = self._read_json_file(result_path)
        except ResultNotFoundError:
            return None

        return self._get_optional_str(result_data, "filename")

    def _get_public_result_url(
        self,
        result_data: dict[str, Any],
        execution_id: str,
    ) -> str:
        result_url = result_data.get("result_url")

        if isinstance(result_url, str) and result_url:
            return result_url

        return f"{self._api_prefix}/results/{execution_id}"

    def _get_public_annotated_image_url(
        self,
        result_data: dict[str, Any],
        execution_id: str,
    ) -> str:
        annotated_image_url = result_data.get("annotated_image_url")

        if isinstance(annotated_image_url, str) and annotated_image_url:
            return annotated_image_url

        return f"{self._api_prefix}/results/{execution_id}/image"

    def _get_optional_str(
        self,
        data: dict[str, Any],
        key: str,
    ) -> str | None:
        value = data.get(key)

        if isinstance(value, str):
            return value

        return None

    def _get_optional_int(
        self,
        data: dict[str, Any],
        key: str,
    ) -> int | None:
        value = data.get(key)

        if isinstance(value, int):
            return value

        return None

    def _read_json_file(self, file_path: Path) -> dict[str, Any]:
        try:
            content = file_path.read_text(encoding="utf-8")
            data = json.loads(content)
        except (OSError, json.JSONDecodeError):
            raise ResultNotFoundError(file_path.stem) from None

        if isinstance(data, dict):
            return data

        raise ResultNotFoundError(file_path.stem)
