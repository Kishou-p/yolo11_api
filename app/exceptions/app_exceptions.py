from pathlib import Path


class AppError(Exception):
    def __init__(self, message: str, status_code: int = 500) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class ModelFileNotFoundError(AppError):
    def __init__(self, model_path: Path) -> None:
        message = f"Model file not found: {model_path}"
        super().__init__(message=message, status_code=404)


class ModelLoadError(AppError):
    def __init__(self, model_path: Path) -> None:
        message = f"Failed to load YOLO model: {model_path}"
        super().__init__(message=message, status_code=500)


class ModelNotLoadedError(AppError):
    def __init__(self, model_path: Path) -> None:
        message = f"YOLO model is not loaded: {model_path}"
        super().__init__(message=message, status_code=409)


class UnsupportedImageTypeError(AppError):
    def __init__(self, content_type: str | None) -> None:
        message = f"Unsupported image type: {content_type}"
        super().__init__(message=message, status_code=415)


class EmptyImageError(AppError):
    def __init__(self) -> None:
        super().__init__(message="Uploaded image is empty.", status_code=400)


class ImageTooLargeError(AppError):
    def __init__(self, max_size_mb: int) -> None:
        message = f"Uploaded image exceeds maximum size of {max_size_mb} MB."
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
        message = f"Failed to write file: {target_path}"
        super().__init__(message=message, status_code=500)

class ResultNotFoundError(AppError):
    def __init__(self, execution_id: str) -> None:
        message = f"Inference result not found: {execution_id}"
        super().__init__(message=message, status_code=404)

class InvalidExecutionIdError(AppError):
    def __init__(self, execution_id: str) -> None:
        message = f"Invalid execution_id: {execution_id}"
        super().__init__(message=message, status_code=400)

class ImageSignatureMismatchError(AppError):
    def __init__(self, content_type: str | None) -> None:
        message = (
            "Uploaded image content does not match "
            f"content type: {content_type}"
        )
        super().__init__(message=message, status_code=400)

class UnauthorizedApiKeyError(AppError):
    def __init__(self) -> None:
        super().__init__(message="Invalid or missing API key.", status_code=401)