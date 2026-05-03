from pathlib import Path
from typing import Callable, Final

import cv2
import numpy as np
from numpy.typing import NDArray

from app.exceptions.app_exceptions import (
    EmptyImageError,
    ImageDecodeError,
    ImageSignatureMismatchError,
    ImageTooLargeError,
    StorageWriteError,
    UnsupportedImageTypeError,
)

SUPPORTED_IMAGE_TYPES: Final[set[str]] = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

IMAGE_SIGNATURE_VALIDATORS: Final[dict[str, Callable[[bytes], bool]]] = {
    "image/jpeg": lambda image_bytes: image_bytes.startswith(b"\xff\xd8\xff"),
    "image/png": lambda image_bytes: image_bytes.startswith(
        b"\x89PNG\r\n\x1a\n"
    ),
    "image/webp": lambda image_bytes: (
        len(image_bytes) >= 12
        and image_bytes[:4] == b"RIFF"
        and image_bytes[8:12] == b"WEBP"
    ),
}


def validate_image_content_type(content_type: str | None) -> None:
    if content_type not in SUPPORTED_IMAGE_TYPES:
        raise UnsupportedImageTypeError(content_type)


def validate_image_size(image_bytes: bytes, max_size_mb: int) -> None:
    if not image_bytes:
        raise EmptyImageError()

    max_size_bytes = _megabytes_to_bytes(max_size_mb)

    if len(image_bytes) > max_size_bytes:
        raise ImageTooLargeError(max_size_mb)


def validate_image_signature(
    content_type: str | None,
    image_bytes: bytes,
) -> None:
    validate_image_content_type(content_type)

    validator = IMAGE_SIGNATURE_VALIDATORS[str(content_type)]

    if not validator(image_bytes):
        raise ImageSignatureMismatchError(content_type)


def decode_image_bytes(image_bytes: bytes) -> NDArray[np.uint8]:
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    decoded_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if decoded_image is None:
        raise ImageDecodeError()

    return decoded_image


def save_image_to_file(
    file_path: Path,
    image: NDArray[np.uint8],
) -> Path:
    success = cv2.imwrite(str(file_path), image)

    if not success:
        raise StorageWriteError(file_path)

    return file_path


def _megabytes_to_bytes(size_mb: int) -> int:
    return size_mb * 1024 * 1024