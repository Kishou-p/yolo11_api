import json
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
    try:
        file_path.write_bytes(content)
    except OSError as exc:
        raise StorageWriteError(file_path) from exc

    return file_path


def save_json_to_file(file_path: Path, content: dict[str, Any]) -> Path:
    try:
        file_path.write_text(
            json.dumps(content, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        raise StorageWriteError(file_path) from exc

    return file_path

def build_execution_id() -> str:
    return uuid4().hex