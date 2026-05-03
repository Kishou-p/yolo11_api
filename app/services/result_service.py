import json

from pathlib import Path
from typing import Any

from app.utils.execution_id_utils import validate_execution_id
from app.exceptions.app_exceptions import ResultNotFoundError
from app.services.storage_service import StorageService


class ResultService:
    def __init__(self, storage_service: StorageService) -> None:
        self._storage_service = storage_service

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

    def _read_json_file(self, file_path: Path) -> dict[str, Any]:
        content = file_path.read_text(encoding="utf-8")
        data = json.loads(content)

        if isinstance(data, dict):
            return data

        raise ResultNotFoundError(file_path.stem)