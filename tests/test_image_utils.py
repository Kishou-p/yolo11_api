from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from app.exceptions.app_exceptions import (
    EmptyImageError,
    ImageDecodeError,
    ImageSignatureMismatchError,
    ImageTooLargeError,
    StorageWriteError,
    UnsupportedImageTypeError,
)
from app.utils.image_utils import (
    decode_image_bytes,
    save_image_to_file,
    validate_image_content_type,
    validate_image_signature,
    validate_image_size,
)


def test_validate_image_content_type_accepts_supported_types() -> None:
    validate_image_content_type("image/jpeg")
    validate_image_content_type("image/png")
    validate_image_content_type("image/webp")


def test_validate_image_content_type_rejects_unsupported_type() -> None:
    with pytest.raises(UnsupportedImageTypeError) as exc_info:
        validate_image_content_type("text/plain")

    assert exc_info.value.status_code == 415
    assert exc_info.value.message == "Unsupported image type: text/plain"


def test_validate_image_size_rejects_empty_bytes() -> None:
    with pytest.raises(EmptyImageError) as exc_info:
        validate_image_size(b"", max_size_mb=10)

    assert exc_info.value.status_code == 400
    assert exc_info.value.message == "Uploaded image is empty."


def test_validate_image_size_rejects_large_image() -> None:
    image_bytes = b"0" * 11

    with pytest.raises(ImageTooLargeError) as exc_info:
        validate_image_size(image_bytes, max_size_mb=0)

    assert exc_info.value.status_code == 413
    assert exc_info.value.message == (
        "Uploaded image exceeds maximum size of 0 MB."
    )


def test_decode_image_bytes_returns_numpy_image() -> None:
    image_bytes = _build_test_image_bytes()

    decoded_image = decode_image_bytes(image_bytes)

    assert isinstance(decoded_image, np.ndarray)
    assert decoded_image.shape == (10, 10, 3)
    assert decoded_image.dtype == np.uint8


def test_decode_image_bytes_rejects_invalid_image_bytes() -> None:
    with pytest.raises(ImageDecodeError) as exc_info:
        decode_image_bytes(b"not-a-valid-image")

    assert exc_info.value.status_code == 400
    assert exc_info.value.message == "Could not decode uploaded image."


def test_save_image_to_file_writes_image(tmp_path: Path) -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    output_path = tmp_path / "output.png"

    saved_path = save_image_to_file(output_path, image)

    assert saved_path == output_path
    assert output_path.exists()


def test_save_image_to_file_raises_storage_error_for_invalid_path(
    tmp_path: Path,
) -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    invalid_path = tmp_path / "missing_dir" / "output.png"

    with pytest.raises(StorageWriteError) as exc_info:
        save_image_to_file(invalid_path, image)

    assert exc_info.value.status_code == 500
    assert exc_info.value.message == f"Failed to write file: {invalid_path}"


def _build_test_image_bytes() -> bytes:
    image: NDArray[np.uint8] = np.zeros((10, 10, 3), dtype=np.uint8)
    success, encoded_image = cv2.imencode(".jpg", image)

    if not success:
        raise RuntimeError("Failed to build test image.")

    return encoded_image.tobytes()

def test_validate_image_signature_accepts_jpeg_signature() -> None:
    image_bytes = b"\xff\xd8\xfffake-jpeg-content"

    validate_image_signature("image/jpeg", image_bytes)


def test_validate_image_signature_accepts_png_signature() -> None:
    image_bytes = b"\x89PNG\r\n\x1a\nfake-png-content"

    validate_image_signature("image/png", image_bytes)


def test_validate_image_signature_accepts_webp_signature() -> None:
    image_bytes = b"RIFF1234WEBPfake-webp-content"

    validate_image_signature("image/webp", image_bytes)


def test_validate_image_signature_rejects_mismatched_content() -> None:
    image_bytes = b"not-a-real-png"

    with pytest.raises(ImageSignatureMismatchError) as exc_info:
        validate_image_signature("image/png", image_bytes)

    assert exc_info.value.status_code == 400
    assert exc_info.value.message == (
        "Uploaded image content does not match content type: image/png"
    )