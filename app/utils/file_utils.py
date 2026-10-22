import json
import os
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.exceptions.app_exceptions import StorageWriteError


def ensure_directory_exists(directory_path: Path) -> None:
    try:
        directory_path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise StorageWriteError(directory_path) from exc


def build_unique_filename(original_filename: str) -> str:
    safe_filename = sanitize_filename(original_filename)
    file_extension = Path(safe_filename).suffix
    file_stem = Path(safe_filename).stem

    unique_id = uuid4().hex

    if file_extension:
        return f"{file_stem}_{unique_id}{file_extension}"

    return f"{file_stem}_{unique_id}"


def sanitize_filename(filename: str) -> str:
    clean_filename = Path(filename).name
    sanitized = re.sub(r"[^a-zA-Z0-9_.-]", "_", clean_filename)

    if sanitized:
        return sanitized

    return "uploaded_file"


def save_bytes_to_file(file_path: Path, content: bytes) -> Path:
    """Write to a temporary file and rename it, so a crash mid-write
    never leaves a truncated file at file_path."""
    temp_path = file_path.with_name(f".{file_path.name}.{uuid4().hex}.tmp")

    try:
        with temp_path.open("wb") as temp_file:
            temp_file.write(content)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, file_path)
    except OSError as exc:
        temp_path.unlink(missing_ok=True)
        raise StorageWriteError(file_path) from exc

    return file_path


def save_json_to_file(file_path: Path, content: dict[str, Any]) -> Path:
    serialized = json.dumps(content, ensure_ascii=False, indent=2)

    return save_bytes_to_file(file_path, serialized.encode("utf-8"))

def build_execution_id() -> str:
    return uuid4().hex