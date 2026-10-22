from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from app.core.config import Settings
from app.main import create_app

STRONG_KEY = "k" * 40
ROTATED_KEY = "r" * 40


def test_production_rejects_default_api_key(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        _build_settings(tmp_path, environment="production", api_key=None)


def test_production_rejects_short_api_key(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        _build_settings(tmp_path, environment="production", api_key="short")


def test_production_rejects_weak_extra_api_key(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        _build_settings(
            tmp_path,
            environment="production",
            api_key=STRONG_KEY,
            extra_api_keys=[SecretStr("change-me")],
        )


def test_rejects_empty_api_key_in_any_environment(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        _build_settings(tmp_path, environment="development", api_key="")


def test_default_environment_is_production() -> None:
    assert Settings.model_fields["environment"].default == "production"


def test_production_accepts_strong_api_key(tmp_path: Path) -> None:
    settings = _build_settings(
        tmp_path,
        environment="production",
        api_key=STRONG_KEY,
    )

    assert settings.valid_api_keys() == [STRONG_KEY]


def test_extra_api_key_is_accepted_for_rotation(tmp_path: Path) -> None:
    client = TestClient(
        create_app(
            _build_settings(
                tmp_path,
                environment="production",
                api_key=STRONG_KEY,
                extra_api_keys=[SecretStr(ROTATED_KEY)],
            )
        )
    )

    for key in (STRONG_KEY, ROTATED_KEY):
        response = client.get(
            "/api/v1/models/status",
            headers={"X-API-Key": key},
        )
        assert response.status_code == 200


def test_api_key_comparison_rejects_prefix(tmp_path: Path) -> None:
    client = _build_client(tmp_path)

    response = client.get(
        "/api/v1/models/status",
        headers={"X-API-Key": STRONG_KEY[:-1]},
    )

    assert response.status_code == 401


def test_docs_are_disabled_in_production(tmp_path: Path) -> None:
    client = _build_client(tmp_path)

    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_docs_are_enabled_in_development(tmp_path: Path) -> None:
    client = TestClient(
        create_app(_build_settings(tmp_path, environment="development"))
    )

    assert client.get("/docs").status_code == 200


def test_model_routes_do_not_expose_internal_paths(tmp_path: Path) -> None:
    client = _build_client(tmp_path)

    status_response = client.get(
        "/api/v1/models/status",
        headers={"X-API-Key": STRONG_KEY},
    )
    load_response = client.post(
        "/api/v1/models/load",
        headers={"X-API-Key": STRONG_KEY},
    )

    assert str(tmp_path) not in status_response.text
    assert str(tmp_path) not in load_response.text
    assert load_response.status_code == 404


def test_upload_larger_than_limit_is_rejected(tmp_path: Path) -> None:
    client = _build_client(tmp_path, max_image_size_mb=1)
    oversized = b"\xff\xd8\xff" + b"0" * (1024 * 1024 + 10)

    response = client.post(
        "/api/v1/inference/image",
        files={"file": ("big.jpg", oversized, "image/jpeg")},
        headers={"X-API-Key": STRONG_KEY},
    )

    assert response.status_code == 413


def test_image_with_too_many_pixels_is_rejected(tmp_path: Path) -> None:
    client = _build_client(tmp_path, max_image_pixels=100 * 100)
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".png", image)

    response = client.post(
        "/api/v1/inference/image",
        files={"file": ("wide.png", encoded.tobytes(), "image/png")},
        headers={"X-API-Key": STRONG_KEY},
    )

    assert response.status_code == 413
    assert list((tmp_path / "uploads").iterdir()) == []


def _build_client(tmp_path: Path, **overrides: object) -> TestClient:
    return TestClient(create_app(_build_settings(tmp_path, **overrides)))


def _build_settings(
    tmp_path: Path,
    environment: str = "production",
    api_key: str | None = STRONG_KEY,
    **overrides: object,
) -> Settings:
    values: dict[str, object] = {
        "environment": environment,
        "yolo_model_path": tmp_path / "missing_model.pt",
        "upload_dir": tmp_path / "uploads",
        "output_dir": tmp_path / "outputs",
        **overrides,
    }

    if api_key is not None:
        values["api_key"] = api_key

    return Settings(**values)
