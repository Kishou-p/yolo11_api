import cv2
import numpy as np
from numpy.typing import NDArray

from app.schemas.inference_schema import DetectionResponse


def draw_detections_on_image(
    image: NDArray[np.uint8],
    detections: list[DetectionResponse],
) -> NDArray[np.uint8]:
    annotated_image = image.copy()

    for detection in detections:
        _draw_detection(annotated_image, detection)

    return annotated_image


def _draw_detection(
    image: NDArray[np.uint8],
    detection: DetectionResponse,
) -> None:
    start_point = (
        int(detection.box.x1),
        int(detection.box.y1),
    )
    end_point = (
        int(detection.box.x2),
        int(detection.box.y2),
    )

    _draw_box(image, start_point, end_point)
    _draw_label(image, detection, start_point)


def _draw_box(
    image: NDArray[np.uint8],
    start_point: tuple[int, int],
    end_point: tuple[int, int],
) -> None:
    cv2.rectangle(
        image,
        start_point,
        end_point,
        color=(0, 255, 0),
        thickness=2,
    )


def _draw_label(
    image: NDArray[np.uint8],
    detection: DetectionResponse,
    start_point: tuple[int, int],
) -> None:
    label = _build_label(detection)
    label_position = (start_point[0], max(start_point[1] - 10, 20))

    cv2.putText(
        image,
        label,
        label_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.6,
        color=(0, 255, 0),
        thickness=2,
    )


def _build_label(detection: DetectionResponse) -> str:
    confidence_percent = detection.confidence * 100
    return f"{detection.class_name} {confidence_percent:.1f}%"