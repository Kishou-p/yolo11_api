import json
from pathlib import Path

import pytest

from app.exceptions.app_exceptions import StorageWriteError
from app.utils.file_utils import (
    build_execution_id,
    build_unique_filename,
    ensure_directory_exists,
    sanitize_filename,
    save_bytes_to_file,
    save_json_to_file,
)


def test_ensure_directory_exists_creates_directory(tmp_path: Path) -> None:
    directory_path = tmp_path / "new_directory"

    ensure_directory_exists(directory_path)

    assert directory_path.exists()
    assert directory_path.is_dir()


def test_sanitize_filename_keeps_safe_filename() -> None:
    filename = sanitize_filename("image_test-01.png")

    assert filename == "image_test-01.png"


def test_sanitize_filename_removes_path_parts() -> None:
    filename = sanitize_filename("../../dangerous.png")

    assert filename == "dangerous.png"


def test_sanitize_filename_replaces_invalid_characters() -> None:
    filename = sanitize_filename("minha imagem @ teste!.png")

    assert filename == "minha_imagem___teste_.png"


def test_sanitize_filename_returns_default_when_empty() -> None:
    filename = sanitize_filename("")

    assert filename == "uploaded_file"


def test_build_unique_filename_preserves_extension() -> None:
    filename = build_unique_filename("test.jpg")

    assert filename.startswith("test_")
    assert filename.endswith(".jpg")


def test_build_unique_filename_handles_filename_without_extension() -> None:
    filename = build_unique_filename("test")

    assert filename.startswith("test_")
    assert "." not in filename


def test_build_execution_id_returns_unique_hex_string() -> None:
    first_id = build_execution_id()
    second_id = build_execution_id()

    assert first_id != second_id
    assert len(first_id) == 32
    assert len(second_id) == 32
    assert first_id.isalnum()
    assert second_id.isalnum()


def test_save_bytes_to_file_writes_content(tmp_path: Path) -> None:
    file_path = tmp_path / "file.bin"
    content = b"hello"

    saved_path = save_bytes_to_file(file_path, content)

    assert saved_path == file_path
    assert file_path.read_bytes() == content


def test_save_bytes_to_file_raises_storage_error_for_invalid_path(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "missing" / "file.bin"

    with pytest.raises(StorageWriteError) as exc_info:
        save_bytes_to_file(file_path, b"hello")

    assert exc_info.value.status_code == 500
    assert exc_info.value.message == f"Failed to write file: {file_path}"


def test_save_json_to_file_writes_json_content(tmp_path: Path) -> None:
    file_path = tmp_path / "result.json"
    content = {
        "execution_id": "abc123",
        "detections_count": 0,
        "detections": [],
    }

    saved_path = save_json_to_file(file_path, content)

    assert saved_path == file_path
    assert json.loads(file_path.read_text(encoding="utf-8")) == content


def test_save_json_to_file_raises_storage_error_for_invalid_path(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "missing" / "result.json"

    with pytest.raises(StorageWriteError) as exc_info:
        save_json_to_file(file_path, {"ok": True})

    assert exc_info.value.status_code == 500
    assert exc_info.value.message == f"Failed to write file: {file_path}"