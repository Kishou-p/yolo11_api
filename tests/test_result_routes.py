import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app

AUTH_HEADERS = {"X-API-Key": "dev-secret-key"}
VALID_EXECUTION_ID = "a" * 32
MISSING_EXECUTION_ID = "b" * 32


def test_get_inference_result_returns_saved_json(tmp_path: Path) -> None:
    execution_id = VALID_EXECUTION_ID
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True)

    expected_result = _build_result_payload(execution_id)
    result_path = output_dir / f"{execution_id}.json"
    result_path.write_text(
        json.dumps(expected_result),
        encoding="utf-8",
    )

    client = _build_test_client(tmp_path)
    response = client.get(
        f"/api/v1/results/{execution_id}",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    assert response.json() == expected_result


def test_get_annotated_image_returns_saved_image(tmp_path: Path) -> None:
    execution_id = VALID_EXECUTION_ID
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True)

    annotated_path = output_dir / f"{execution_id}_annotated.png"
    annotated_path.write_bytes(b"fake-png-content")

    client = _build_test_client(tmp_path)
    response = client.get(
        f"/api/v1/results/{execution_id}/image",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == b"fake-png-content"


def test_get_inference_result_returns_404_when_missing(
    tmp_path: Path,
) -> None:
    execution_id = MISSING_EXECUTION_ID
    client = _build_test_client(tmp_path)

    response = client.get(
        f"/api/v1/results/{execution_id}",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": f"Inference result not found: {execution_id}",
    }


def test_get_annotated_image_returns_404_when_missing(
    tmp_path: Path,
) -> None:
    execution_id = MISSING_EXECUTION_ID
    client = _build_test_client(tmp_path)

    response = client.get(
        f"/api/v1/results/{execution_id}/image",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": f"Inference result not found: {execution_id}",
    }


def test_get_inference_result_rejects_invalid_execution_id(
    tmp_path: Path,
) -> None:
    client = _build_test_client(tmp_path)

    response = client.get(
        "/api/v1/results/invalid-id",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Invalid execution_id: invalid-id",
    }


def test_get_annotated_image_rejects_invalid_execution_id(
    tmp_path: Path,
) -> None:
    client = _build_test_client(tmp_path)

    response = client.get(
        "/api/v1/results/abc123/image",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Invalid execution_id: abc123",
    }


def _build_test_client(tmp_path: Path) -> TestClient:
    settings = Settings(
        app_name="Test YOLO11 API",
        app_version="0.0.1",
        environment="test",
        log_level="INFO",
        api_prefix="/api/v1",
        yolo_model_path=tmp_path / "fake_model.pt",
        max_image_size_mb=10,
        upload_dir=tmp_path / "uploads",
        output_dir=tmp_path / "outputs",
        api_key="dev-secret-key",
    )

    return TestClient(create_app(settings=settings))


def _build_result_payload(execution_id: str) -> dict[str, Any]:
    return {
        "execution_id": execution_id,
        "filename": "test.jpg",
        "result_url": f"/api/v1/results/{execution_id}",
        "annotated_image_url": f"/api/v1/results/{execution_id}/image",
        "detections_count": 0,
        "detections": [],
    }