from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from app.core.config import Settings
from app.main import create_app
from app.services.yolo_service import YoloService

AUTH_HEADERS = {"X-API-Key": "dev-secret-key"}


def test_model_status_returns_false_when_model_file_does_not_exist(
    tmp_path: Path,
) -> None:
    missing_model_path = tmp_path / "missing_model.pt"
    client = _build_test_client(model_path=missing_model_path)

    response = client.get(
        "/api/v1/models/status",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    assert response.json() == {
        "model_path": str(missing_model_path),
        "exists": False,
        "loaded": False,
    }


def test_load_model_returns_404_when_model_file_does_not_exist(
    tmp_path: Path,
) -> None:
    missing_model_path = tmp_path / "missing_model.pt"
    client = _build_test_client(model_path=missing_model_path)

    response = client.post(
        "/api/v1/models/load",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": f"Model file not found: {missing_model_path}",
    }


def test_load_model_returns_success_when_service_loads_model(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    model_path = tmp_path / "fake_model.pt"
    client = _build_test_client(model_path=model_path)

    monkeypatch.setattr(YoloService, "load_model", _fake_load_model)

    response = client.post(
        "/api/v1/models/load",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    assert response.json() == {
        "model_path": str(model_path),
        "loaded": True,
        "message": "YOLO model loaded successfully.",
    }


def test_load_model_reuses_loaded_model(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    model_path = tmp_path / "fake_model.pt"
    client = _build_test_client(model_path=model_path)
    load_calls = {"count": 0}

    def fake_load_model(self: YoloService) -> Any:
        load_calls["count"] += 1
        return object()

    monkeypatch.setattr(YoloService, "load_model", fake_load_model)

    first_response = client.post(
        "/api/v1/models/load",
        headers=AUTH_HEADERS,
    )
    second_response = client.post(
        "/api/v1/models/load",
        headers=AUTH_HEADERS,
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert load_calls["count"] == 1


def _build_test_client(model_path: Path) -> TestClient:
    settings = Settings(
        app_name="Test YOLO11 API",
        app_version="0.0.1",
        environment="test",
        log_level="INFO",
        api_prefix="/api/v1",
        yolo_model_path=model_path,
        max_image_size_mb=10,
        upload_dir=model_path.parent / "uploads",
        output_dir=model_path.parent / "outputs",
        api_key="dev-secret-key",
    )

    return TestClient(create_app(settings=settings))


def _fake_load_model(self: YoloService) -> Any:
    return object()