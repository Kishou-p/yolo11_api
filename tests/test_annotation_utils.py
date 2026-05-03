import numpy as np

from app.schemas.inference_schema import BoundingBox, DetectionResponse
from app.utils.annotation_utils import draw_detections_on_image


def test_draw_detections_returns_copy_when_no_detections() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    annotated_image = draw_detections_on_image(image, detections=[])

    assert annotated_image is not image
    assert annotated_image.shape == image.shape
    assert annotated_image.dtype == image.dtype
    assert np.array_equal(annotated_image, image)


def test_draw_detections_does_not_modify_original_image() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    original_image = image.copy()
    detections = [_build_detection()]

    draw_detections_on_image(image, detections)

    assert np.array_equal(image, original_image)


def test_draw_detections_changes_annotated_image() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [_build_detection()]

    annotated_image = draw_detections_on_image(image, detections)

    assert annotated_image.shape == image.shape
    assert annotated_image.dtype == image.dtype
    assert not np.array_equal(annotated_image, image)


def test_draw_detections_handles_multiple_detections() -> None:
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    detections = [
        _build_detection(
            class_id=0,
            class_name="person",
            confidence=0.95,
            x1=10.0,
            y1=20.0,
            x2=60.0,
            y2=90.0,
        ),
        _build_detection(
            class_id=1,
            class_name="car",
            confidence=0.87,
            x1=70.0,
            y1=30.0,
            x2=110.0,
            y2=100.0,
        ),
    ]

    annotated_image = draw_detections_on_image(image, detections)

    assert annotated_image.shape == image.shape
    assert not np.array_equal(annotated_image, image)


def _build_detection(
    class_id: int = 0,
    class_name: str = "person",
    confidence: float = 0.95,
    x1: float = 10.0,
    y1: float = 20.0,
    x2: float = 80.0,
    y2: float = 90.0,
) -> DetectionResponse:
    return DetectionResponse(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        box=BoundingBox(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        ),
    )