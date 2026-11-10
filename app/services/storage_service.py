import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.exceptions.app_exceptions import StorageDeleteError
from app.utils.file_utils import (
    build_execution_id,
    ensure_directory_exists,
    save_bytes_to_file,
    save_json_to_file,
)
from app.utils.image_utils import get_image_extension, save_image_to_file

logger = logging.getLogger(__name__)

ANNOTATED_EXTENSIONS = (".png", ".jpg")


def _safe_mtime(path: Path) -> float:
    # A file may be deleted between glob() and stat(); sort it last.
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


class StorageService:
    def __init__(
        self,
        upload_dir: Path,
        output_dir: Path,
        annotated_format: str = "png",
    ) -> None:
        self._upload_dir = upload_dir
        self._output_dir = output_dir
        self._annotated_extension = f".{annotated_format}"

        ensure_directory_exists(self._upload_dir)
        ensure_directory_exists(self._output_dir)

    def create_execution_id(self) -> str:
        return build_execution_id()

    def save_upload(
        self,
        execution_id: str,
        content_type: str | None,
        image_bytes: bytes,
    ) -> Path:
        upload_path = self.build_upload_path(
            execution_id=execution_id,
            content_type=content_type,
        )

        return save_bytes_to_file(upload_path, image_bytes)

    def build_upload_path(
        self,
        execution_id: str,
        content_type: str | None,
    ) -> Path:
        extension = get_image_extension(content_type)

        return self._upload_dir / f"{execution_id}{extension}"

    def find_upload_paths(self, execution_id: str) -> list[Path]:
        return list(self._upload_dir.glob(f"{execution_id}.*"))

    def build_result_path(self, execution_id: str) -> Path:
        return self._output_dir / f"{execution_id}.json"

    def build_annotated_path(self, execution_id: str) -> Path:
        return (
            self._output_dir
            / f"{execution_id}_annotated{self._annotated_extension}"
        )

    def find_annotated_path(self, execution_id: str) -> Path | None:
        # Results saved before ANNOTATED_IMAGE_FORMAT changed keep their
        # original extension, so look for any of them.
        for extension in (self._annotated_extension, *ANNOTATED_EXTENSIONS):
            path = self._output_dir / f"{execution_id}_annotated{extension}"
            if path.exists():
                return path

        return None

    def list_result_paths(self) -> list[Path]:
        return sorted(
            self._output_dir.glob("*.json"),
            key=_safe_mtime,
            reverse=True,
        )

    def delete_file_if_exists(self, file_path: Path) -> bool:
        try:
            file_path.unlink()
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise StorageDeleteError(file_path) from exc

        return True

    def delete_execution_files(self, execution_id: str) -> None:
        paths = [
            self.build_result_path(execution_id),
            *self.find_annotated_paths(execution_id),
            *self.find_upload_paths(execution_id),
        ]

        for path in paths:
            try:
                self.delete_file_if_exists(path)
            except StorageDeleteError:
                logger.exception("Failed to delete %s", path)

    def find_annotated_paths(self, execution_id: str) -> list[Path]:
        return list(self._output_dir.glob(f"{execution_id}_annotated.*"))

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
