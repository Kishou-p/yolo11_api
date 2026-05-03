import json
from pathlib import Path

import numpy as np

from app.services.storage_service import StorageService


def test_storage_service_creates_directories(tmp_path: Path) -> None:
    upload_dir = tmp_path / "uploads"
    output_dir = tmp_path / "outputs"

    StorageService(upload_dir=upload_dir, output_dir=output_dir)

    assert upload_dir.exists()
    assert upload_dir.is_dir()
    assert output_dir.exists()
    assert output_dir.is_dir()


def test_create_execution_id_returns_hex_string(tmp_path: Path) -> None:
    storage_service = _build_storage_service(tmp_path)

    execution_id = storage_service.create_execution_id()

    assert len(execution_id) == 32
    assert execution_id.isalnum()


def test_build_upload_path_uses_execution_id_and_extension(
    tmp_path: Path,
) -> None:
    storage_service = _build_storage_service(tmp_path)

    upload_path = storage_service.build_upload_path(
        execution_id="abc123",
        original_filename="image.png",
    )

    assert upload_path == tmp_path / "uploads" / "abc123.png"


def test_build_upload_path_uses_bin_when_no_extension(
    tmp_path: Path,
) -> None:
    storage_service = _build_storage_service(tmp_path)

    upload_path = storage_service.build_upload_path(
        execution_id="abc123",
        original_filename="image",
    )

    assert upload_path == tmp_path / "uploads" / "abc123.bin"


def test_save_upload_writes_file(tmp_path: Path) -> None:
    storage_service = _build_storage_service(tmp_path)

    saved_path = storage_service.save_upload(
        execution_id="abc123",
        original_filename="image.jpg",
        image_bytes=b"fake-image",
    )

    assert saved_path == tmp_path / "uploads" / "abc123.jpg"
    assert saved_path.read_bytes() == b"fake-image"


def test_build_result_path_uses_execution_id(tmp_path: Path) -> None:
    storage_service = _build_storage_service(tmp_path)

    result_path = storage_service.build_result_path("abc123")

    assert result_path == tmp_path / "outputs" / "abc123.json"


def test_save_inference_result_writes_json(tmp_path: Path) -> None:
    storage_service = _build_storage_service(tmp_path)
    result = {
        "execution_id": "abc123",
        "detections_count": 0,
        "detections": [],
    }

    saved_path = storage_service.save_inference_result(
        execution_id="abc123",
        result=result,
    )

    assert saved_path == tmp_path / "outputs" / "abc123.json"
    assert json.loads(saved_path.read_text(encoding="utf-8")) == result


def test_build_annotated_path_uses_execution_id(tmp_path: Path) -> None:
    storage_service = _build_storage_service(tmp_path)

    annotated_path = storage_service.build_annotated_path("abc123")

    assert annotated_path == tmp_path / "outputs" / "abc123_annotated.png"


def test_save_annotated_image_writes_image(tmp_path: Path) -> None:
    storage_service = _build_storage_service(tmp_path)
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    saved_path = storage_service.save_annotated_image(
        execution_id="abc123",
        image=image,
    )

    assert saved_path == tmp_path / "outputs" / "abc123_annotated.png"
    assert saved_path.exists()
    assert saved_path.stat().st_size > 0


def _build_storage_service(tmp_path: Path) -> StorageService:
    return StorageService(
        upload_dir=tmp_path / "uploads",
        output_dir=tmp_path / "outputs",
    )