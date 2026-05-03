from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pytest import MonkeyPatch

from app.exceptions.app_exceptions import (
    ModelFileNotFoundError,
    ModelInferenceError,
)
from app.services import yolo_service
from app.services.yolo_service import YoloService


def test_model_exists_returns_true_when_file_exists(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"fake-model")

    service = YoloService(model_path=model_path)

    assert service.model_exists() is True


def test_model_exists_returns_false_when_file_does_not_exist(
    tmp_path: Path,
) -> None:
    service = YoloService(model_path=tmp_path / "missing.pt")

    assert service.model_exists() is False


def test_load_model_raises_when_model_file_does_not_exist(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "missing.pt"
    service = YoloService(model_path=model_path)

    with pytest.raises(ModelFileNotFoundError) as exc_info:
        service.load_model()

    assert exc_info.value.status_code == 404
    assert exc_info.value.message == f"Model file not found: {model_path}"


def test_load_model_calls_yolo_when_file_exists(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"fake-model")
    fake_yolo = FakeYoloFactory()

    monkeypatch.setattr(yolo_service, "YOLO", fake_yolo)

    service = YoloService(model_path=model_path)
    loaded_model = service.load_model()

    assert loaded_model == fake_yolo.created_model
    assert fake_yolo.received_path == str(model_path)


def test_predict_converts_yolo_boxes_to_detection_responses() -> None:
    service = YoloService(model_path=Path("fake_model.pt"))
    model = FakePredictModel(results=[_build_fake_result()])
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    detections = service.predict(model=model, image=image)

    assert len(detections) == 1
    assert detections[0].class_id == 0
    assert detections[0].class_name == "person"
    assert detections[0].confidence == 0.95
    assert detections[0].box.x1 == 10.0
    assert detections[0].box.y1 == 20.0
    assert detections[0].box.x2 == 100.0
    assert detections[0].box.y2 == 200.0
    assert model.predict_calls == 1
    assert model.last_source is image


def test_predict_returns_empty_list_when_results_are_empty() -> None:
    service = YoloService(model_path=Path("fake_model.pt"))
    model = FakePredictModel(results=[])
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    detections = service.predict(model=model, image=image)

    assert detections == []


def test_predict_raises_model_inference_error_when_model_fails() -> None:
    service = YoloService(model_path=Path("fake_model.pt"))
    model = FakeFailingPredictModel()
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(ModelInferenceError) as exc_info:
        service.predict(model=model, image=image)

    assert exc_info.value.status_code == 500
    assert exc_info.value.message == "Failed to run YOLO inference."


class FakeYoloFactory:
    def __init__(self) -> None:
        self.created_model = object()
        self.received_path: str | None = None

    def __call__(self, model_path: str) -> Any:
        self.received_path = model_path
        return self.created_model


class FakePredictModel:
    def __init__(self, results: list[Any]) -> None:
        self._results = results
        self.predict_calls = 0
        self.last_source: Any | None = None

    def predict(self, source: Any, verbose: bool) -> list[Any]:
        self.predict_calls += 1
        self.last_source = source
        return self._results


class FakeFailingPredictModel:
    def predict(self, source: Any, verbose: bool) -> list[Any]:
        raise RuntimeError("fake inference failure")


class FakeResult:
    def __init__(self) -> None:
        self.names = {0: "person"}
        self.boxes = [_build_fake_box()]


class FakeBox:
    def __init__(self) -> None:
        self.cls = FakeScalar(0)
        self.conf = FakeScalar(0.95)
        self.xyxy = [FakeTensor([10.0, 20.0, 100.0, 200.0])]


class FakeScalar:
    def __init__(self, value: int | float) -> None:
        self._value = value

    def item(self) -> int | float:
        return self._value


class FakeTensor:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


def _build_fake_result() -> FakeResult:
    return FakeResult()


def _build_fake_box() -> FakeBox:
    return FakeBox()