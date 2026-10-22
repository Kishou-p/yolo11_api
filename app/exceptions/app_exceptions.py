from pathlib import Path


class AppError(Exception):
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        log_detail: str | None = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.log_detail = log_detail
        super().__init__(message)


class ModelFileNotFoundError(AppError):
    def __init__(self, model_path: Path) -> None:
        super().__init__(
            message="Model file not found.",
            status_code=404,
            log_detail=f"Model file not found: {model_path}",
        )


class ModelLoadError(AppError):
    def __init__(self, model_path: Path) -> None:
        super().__init__(
            message="Failed to load YOLO model.",
            status_code=500,
            log_detail=f"Failed to load YOLO model: {model_path}",
        )


class ModelNotLoadedError(AppError):
    def __init__(self, model_path: Path) -> None:
        super().__init__(
            message="YOLO model is not loaded.",
            status_code=409,
            log_detail=f"YOLO model is not loaded: {model_path}",
        )


class UnsupportedImageTypeError(AppError):
    def __init__(self, content_type: str | None) -> None:
        message = f"Unsupported image type: {_truncate(content_type)}"
        super().__init__(message=message, status_code=415)


class EmptyImageError(AppError):
    def __init__(self) -> None:
        super().__init__(message="Uploaded image is empty.", status_code=400)


class ImageTooLargeError(AppError):
    def __init__(self, max_size_mb: int) -> None:
        message = f"Uploaded image exceeds maximum size of {max_size_mb} MB."
        super().__init__(message=message, status_code=413)


class ImageDimensionsTooLargeError(AppError):
    def __init__(self, max_pixels: int) -> None:
        message = (
            "Uploaded image exceeds maximum resolution of "
            f"{max_pixels} pixels."
        )
        super().__init__(message=message, status_code=413)


class ImageDecodeError(AppError):
    def __init__(self) -> None:
        super().__init__(
            message="Could not decode uploaded image.",
            status_code=400,
        )


class ModelInferenceError(AppError):
    def __init__(self) -> None:
        super().__init__(
            message="Failed to run YOLO inference.",
            status_code=500,
        )

class StorageWriteError(AppError):
    def __init__(self, target_path: Path) -> None:
        super().__init__(
            message="Failed to write file.",
            status_code=500,
            log_detail=f"Failed to write file: {target_path}",
        )

class ResultNotFoundError(AppError):
    def __init__(self, execution_id: str) -> None:
        message = f"Inference result not found: {execution_id}"
        super().__init__(message=message, status_code=404)

class InvalidExecutionIdError(AppError):
    def __init__(self, execution_id: str) -> None:
        super().__init__(
            message="Invalid execution_id.",
            status_code=400,
            log_detail=f"Invalid execution_id: {_truncate(execution_id)!r}",
        )

class ImageSignatureMismatchError(AppError):
    def __init__(self, content_type: str | None) -> None:
        message = (
            "Uploaded image content does not match "
            f"content type: {_truncate(content_type)}"
        )
        super().__init__(message=message, status_code=400)

class UnauthorizedApiKeyError(AppError):
    def __init__(self) -> None:
        super().__init__(message="Invalid or missing API key.", status_code=401)


def _truncate(value: str | None, max_length: int = 100) -> str | None:
    if value is None or len(value) <= max_length:
        return value

    return f"{value[:max_length]}..."
