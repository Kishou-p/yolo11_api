import json
from pathlib import Path
from typing import Any

import pytest

from app.exceptions.app_exceptions import (
    InvalidExecutionIdError,
    ResultNotFoundError,
)
from app.services.result_service import ResultService
from app.services.storage_service import StorageService

VALID_EXECUTION_ID = "a" * 32
MISSING_EXECUTION_ID = "b" * 32


def test_get_result_data_returns_saved_json(tmp_path: Path) -> None:
    execution_id = VALID_EXECUTION_ID
    storage_service = _build_storage_service(tmp_path)
    result_path = storage_service.build_result_path(execution_id)
    expected_result = _build_result_payload(execution_id)

    result_path.write_text(
        json.dumps(expected_result),
        encoding="utf-8",
    )

    result_service = ResultService(storage_service=storage_service)

    result = result_service.get_result_data(execution_id)

    assert result == expected_result


def test_get_annotated_image_path_returns_existing_path(
    tmp_path: Path,
) -> None:
    execution_id = VALID_EXECUTION_ID
    storage_service = _build_storage_service(tmp_path)
    annotated_path = storage_service.build_annotated_path(execution_id)
    annotated_path.write_bytes(b"fake-image")

    result_service = ResultService(storage_service=storage_service)

    result = result_service.get_annotated_image_path(execution_id)

    assert result == annotated_path


def test_get_result_data_raises_when_result_does_not_exist(
    tmp_path: Path,
) -> None:
    execution_id = MISSING_EXECUTION_ID
    storage_service = _build_storage_service(tmp_path)
    result_service = ResultService(storage_service=storage_service)

    with pytest.raises(ResultNotFoundError) as exc_info:
        result_service.get_result_data(execution_id)

    assert exc_info.value.status_code == 404
    assert exc_info.value.message == (
        f"Inference result not found: {execution_id}"
    )


def test_get_annotated_image_path_raises_when_image_does_not_exist(
    tmp_path: Path,
) -> None:
    execution_id = MISSING_EXECUTION_ID
    storage_service = _build_storage_service(tmp_path)
    result_service = ResultService(storage_service=storage_service)

    with pytest.raises(ResultNotFoundError) as exc_info:
        result_service.get_annotated_image_path(execution_id)

    assert exc_info.value.status_code == 404
    assert exc_info.value.message == (
        f"Inference result not found: {execution_id}"
    )


def test_get_result_data_raises_when_json_is_not_object(
    tmp_path: Path,
) -> None:
    execution_id = VALID_EXECUTION_ID
    storage_service = _build_storage_service(tmp_path)
    result_path = storage_service.build_result_path(execution_id)

    result_path.write_text(
        json.dumps(["not", "a", "dict"]),
        encoding="utf-8",
    )

    result_service = ResultService(storage_service=storage_service)

    with pytest.raises(ResultNotFoundError) as exc_info:
        result_service.get_result_data(execution_id)

    assert exc_info.value.status_code == 404
    assert exc_info.value.message == (
        f"Inference result not found: {execution_id}"
    )


def test_get_result_data_rejects_invalid_execution_id(
    tmp_path: Path,
) -> None:
    storage_service = _build_storage_service(tmp_path)
    result_service = ResultService(storage_service=storage_service)

    with pytest.raises(InvalidExecutionIdError) as exc_info:
        result_service.get_result_data("../../danger")

    assert exc_info.value.status_code == 400
    assert exc_info.value.message == "Invalid execution_id: ../../danger"


def test_get_annotated_image_path_rejects_invalid_execution_id(
    tmp_path: Path,
) -> None:
    storage_service = _build_storage_service(tmp_path)
    result_service = ResultService(storage_service=storage_service)

    with pytest.raises(InvalidExecutionIdError) as exc_info:
        result_service.get_annotated_image_path("abc123")

    assert exc_info.value.status_code == 400
    assert exc_info.value.message == "Invalid execution_id: abc123"


def _build_storage_service(tmp_path: Path) -> StorageService:
    return StorageService(
        upload_dir=tmp_path / "uploads",
        output_dir=tmp_path / "outputs",
    )


def _build_result_payload(execution_id: str) -> dict[str, Any]:
    return {
        "execution_id": execution_id,
        "filename": "test.jpg",
        "result_url": f"/api/v1/results/{execution_id}",
        "annotated_image_url": f"/api/v1/results/{execution_id}/image",
        "detections_count": 0,
        "detections": [],
    }
def test_list_results_returns_saved_results(tmp_path: Path) -> None:
    first_execution_id = "a" * 32
    second_execution_id = "b" * 32

    storage_service = _build_storage_service(tmp_path)

    first_result_path = storage_service.build_result_path(first_execution_id)
    second_result_path = storage_service.build_result_path(second_execution_id)

    first_result_path.write_text(
        json.dumps(_build_result_payload(first_execution_id)),
        encoding="utf-8",
    )
    second_result_path.write_text(
        json.dumps(_build_result_payload(second_execution_id)),
        encoding="utf-8",
    )

    result_service = ResultService(storage_service=storage_service)

    result = result_service.list_results()

    assert result.count == 2

    results_by_id = {
        item.execution_id: item
        for item in result.results
    }

    assert first_execution_id in results_by_id
    assert second_execution_id in results_by_id
    assert results_by_id[first_execution_id].filename == "test.jpg"
    assert (
        results_by_id[first_execution_id].result_url
        == f"/api/v1/results/{first_execution_id}"
    )
    assert (
        results_by_id[first_execution_id].annotated_image_url
        == f"/api/v1/results/{first_execution_id}/image"
    )


def test_list_results_builds_public_urls_when_missing(
    tmp_path: Path,
) -> None:
    execution_id = VALID_EXECUTION_ID
    storage_service = _build_storage_service(tmp_path)

    result_path = storage_service.build_result_path(execution_id)
    result_path.write_text(
        json.dumps(
            {
                "execution_id": execution_id,
                "filename": "test.jpg",
                "detections_count": 3,
            }
        ),
        encoding="utf-8",
    )

    result_service = ResultService(
        storage_service=storage_service,
        api_prefix="/custom/api",
    )

    result = result_service.list_results()

    assert result.count == 1
    assert result.results[0].result_url == (
        f"/custom/api/results/{execution_id}"
    )
    assert result.results[0].annotated_image_url == (
        f"/custom/api/results/{execution_id}/image"
    )
    assert result.results[0].detections_count == 3


def test_delete_result_removes_result_image_and_upload(
    tmp_path: Path,
) -> None:
    execution_id = VALID_EXECUTION_ID
    storage_service = _build_storage_service(tmp_path)

    result_path = storage_service.build_result_path(execution_id)
    annotated_path = storage_service.build_annotated_path(execution_id)
    upload_path = storage_service.build_upload_path(
        execution_id=execution_id,
        original_filename="test.jpg",
    )

    result_path.write_text(
        json.dumps(_build_result_payload(execution_id)),
        encoding="utf-8",
    )
    annotated_path.write_bytes(b"fake-image")
    upload_path.write_bytes(b"fake-upload")

    result_service = ResultService(storage_service=storage_service)

    result = result_service.delete_result(execution_id)

    assert result.execution_id == execution_id
    assert result.deleted is True
    assert result.message == "Result deleted successfully."
    assert not result_path.exists()
    assert not annotated_path.exists()
    assert not upload_path.exists()


def test_delete_result_raises_when_result_does_not_exist(
    tmp_path: Path,
) -> None:
    execution_id = MISSING_EXECUTION_ID
    storage_service = _build_storage_service(tmp_path)
    result_service = ResultService(storage_service=storage_service)

    with pytest.raises(ResultNotFoundError) as exc_info:
        result_service.delete_result(execution_id)

    assert exc_info.value.status_code == 404
    assert exc_info.value.message == (
        f"Inference result not found: {execution_id}"
    )


def test_delete_result_rejects_invalid_execution_id(
    tmp_path: Path,
) -> None:
    storage_service = _build_storage_service(tmp_path)
    result_service = ResultService(storage_service=storage_service)

    with pytest.raises(InvalidExecutionIdError) as exc_info:
        result_service.delete_result("invalid-id")

    assert exc_info.value.status_code == 400
    assert exc_info.value.message == "Invalid execution_id: invalid-id"