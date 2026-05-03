from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app

ALLOWED_ORIGIN = "http://localhost:3000"
BLOCKED_ORIGIN = "http://malicious.local"


def test_allowed_origin_receives_cors_header(tmp_path: Path) -> None:
    client = _build_test_client(tmp_path)

    response = client.get(
        "/api/v1/health",
        headers={"Origin": ALLOWED_ORIGIN},
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ALLOWED_ORIGIN


def test_blocked_origin_does_not_receive_cors_header(
    tmp_path: Path,
) -> None:
    client = _build_test_client(tmp_path)

    response = client.get(
        "/api/v1/health",
        headers={"Origin": BLOCKED_ORIGIN},
    )

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_cors_preflight_allows_configured_origin(tmp_path: Path) -> None:
    client = _build_test_client(tmp_path)

    response = client.options(
        "/api/v1/inference/image",
        headers={
            "Origin": ALLOWED_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "X-API-Key, Content-Type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ALLOWED_ORIGIN
    assert "POST" in response.headers["access-control-allow-methods"]
    assert "X-API-Key" in response.headers["access-control-allow-headers"]
    assert "Content-Type" in response.headers["access-control-allow-headers"]


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
        allowed_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
    )

    return TestClient(create_app(settings=settings))