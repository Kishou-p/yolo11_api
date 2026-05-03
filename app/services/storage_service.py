from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.utils.file_utils import (
    build_execution_id,
    ensure_directory_exists,
    sanitize_filename,
    save_bytes_to_file,
    save_json_to_file,
)
from app.utils.image_utils import save_image_to_file


class StorageService:
    def __init__(
        self,
        upload_dir: Path,
        output_dir: Path,
    ) -> None:
        self._upload_dir = upload_dir
        self._output_dir = output_dir

        ensure_directory_exists(self._upload_dir)
        ensure_directory_exists(self._output_dir)

    def create_execution_id(self) -> str:
        return build_execution_id()

    def save_upload(
        self,
        execution_id: str,
        original_filename: str,
        image_bytes: bytes,
    ) -> Path:
        upload_path = self.build_upload_path(
            execution_id=execution_id,
            original_filename=original_filename,
        )

        return save_bytes_to_file(upload_path, image_bytes)

    def build_upload_path(
        self,
        execution_id: str,
        original_filename: str,
    ) -> Path:
        safe_filename = sanitize_filename(original_filename)
        extension = Path(safe_filename).suffix or ".bin"

        return self._upload_dir / f"{execution_id}{extension}"

    def build_result_path(self, execution_id: str) -> Path:
        return self._output_dir / f"{execution_id}.json"

    def build_annotated_path(self, execution_id: str) -> Path:
        return self._output_dir / f"{execution_id}_annotated.png"

    def save_inference_result(
        self,
        execution_id: str,
        result: dict[str, Any],
    ) -> Path:
        output_path = self.build_result_path(execution_id)

        return save_json_to_file(output_path, result)

    def save_annotated_image(
        self,
        execution_id: str,
        image: NDArray[np.uint8],
    ) -> Path:
        annotated_path = self.build_annotated_path(execution_id)

        return save_image_to_file(annotated_path, image)