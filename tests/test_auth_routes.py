from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app

VALID_AUTH_HEADERS = {"X-API-Key": "dev-secret-key"}
INVALID_AUTH_HEADERS = {"X-API-Key": "wrong-key"}


def test_health_route_is_public_without_api_key(tmp_path: Path) -> None:
    client = _build_test_client(tmp_path)

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "app_name": "Test YOLO11 API",
        "version": "0.0.1",
        "environment": "test",
    }


def test_protected_route_rejects_missing_api_key(tmp_path: Path) -> None:
    client = _build_test_client(tmp_path)

    response = client.get("/api/v1/models/status")

    assert response.status_code == 401
    assert response.json() == {
        "detail": "Invalid or missing API key.",
    }


def test_protected_route_rejects_invalid_api_key(tmp_path: Path) -> None:
    client = _build_test_client(tmp_path)

    response = client.get(
        "/api/v1/models/status",
        headers=INVALID_AUTH_HEADERS,
    )

    assert response.status_code == 401
    assert response.json() == {
        "detail": "Invalid or missing API key.",
    }


def test_protected_route_accepts_valid_api_key(tmp_path: Path) -> None:
    model_path = tmp_path / "missing_model.pt"
    client = _build_test_client(tmp_path, model_path=model_path)

    response = client.get(
        "/api/v1/models/status",
        headers=VALID_AUTH_HEADERS,
    )

    assert response.status_code == 200
    assert response.json() == {
        "model_path": str(model_path),
        "exists": False,
        "loaded": False,
    }


def _build_test_client(
    tmp_path: Path,
    model_path: Path | None = None,
) -> TestClient:
    settings = Settings(
        app_name="Test YOLO11 API",
        app_version="0.0.1",
        environment="test",
        log_level="INFO",
        api_prefix="/api/v1",
        yolo_model_path=model_path or tmp_path / "fake_model.pt",
        max_image_size_mb=10,
        upload_dir=tmp_path / "uploads",
        output_dir=tmp_path / "outputs",
        api_key="dev-secret-key",
    )

    return TestClient(create_app(settings=settings))