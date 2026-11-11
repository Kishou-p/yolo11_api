"""Demo flow: label photos in the browser, fine-tune YOLO11, test new photos.

Everything lives under DEMO_DIR:

    images/<id>.jpg       photos (re-encoded, so EXIF and metadata are dropped)
    labels/<id>.txt       YOLO labels: "class cx cy w h", normalised
    dataset.yaml          written before each training run
    runs/train/           Ultralytics output of the last run
    model/best.pt         the trained model used by /demo/predict
    model/meta.json       when it was trained, on how many images, mAP50

The dataset is tiny and is used for both training and validation: this is a
demo of the flow, not a way to measure real accuracy.
"""

import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2

from app.exceptions.app_exceptions import AppError, DemoError
from app.schemas.demo_schema import (
    DemoBox,
    DemoDataset,
    DemoDetection,
    DemoImage,
    DemoPrediction,
    DemoTrainStatus,
)
from app.schemas.inference_schema import InferenceParams
from app.services.yolo_service import YoloService, sha256_file
from app.utils.file_utils import save_bytes_to_file
from app.utils.image_utils import (
    decode_image_bytes,
    validate_image_dimensions,
    validate_image_signature,
    validate_image_size,
)

logger = logging.getLogger(__name__)

IMAGE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
MAX_BOXES_PER_IMAGE = 50
# Photos are stored at most this big on the longest side: plenty for
# training at 512-640 px, and it keeps the dataset small on a Raspberry Pi.
MAX_STORED_SIDE = 1280
MAP50_KEY = "metrics/mAP50(B)"


class DemoService:
    def __init__(
        self,
        *,
        demo_dir: Path,
        classes: list[str],
        base_model_path: Path,
        base_model_sha256: str | None,
        params: InferenceParams,
        train_epochs: int,
        train_imgsz: int,
        train_batch: int,
        min_images: int,
        max_image_size_mb: int,
        max_image_pixels: int,
    ) -> None:
        self._dir = demo_dir
        self._classes = classes
        self._base_model_path = base_model_path
        self._base_model_sha256 = base_model_sha256
        self._params = params
        self._predict_params = params.model_copy(update={"imgsz": train_imgsz})
        self._train_epochs = train_epochs
        self._train_imgsz = train_imgsz
        self._train_batch = train_batch
        self._min_images = min_images
        self._max_image_size_mb = max_image_size_mb
        self._max_image_pixels = max_image_pixels

        self._status_lock = threading.Lock()
        self._status = DemoTrainStatus(state="idle")
        self._stop_requested = False
        self._thread: threading.Thread | None = None

        # Ultralytics models are not safe for concurrent predict calls.
        self._predict_lock = threading.Lock()
        self._predict_model: Any | None = None
        self._predict_service: YoloService | None = None

    # ------------------------------------------------------------------ paths

    @property
    def classes(self) -> list[str]:
        return list(self._classes)

    @property
    def _images_dir(self) -> Path:
        return self._dir / "images"

    @property
    def _labels_dir(self) -> Path:
        return self._dir / "labels"

    @property
    def _model_dir(self) -> Path:
        return self._dir / "model"

    @property
    def model_path(self) -> Path:
        return self._model_dir / "best.pt"

    def ensure_directories(self) -> None:
        for directory in (self._images_dir, self._labels_dir, self._model_dir):
            directory.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- dataset

    def list_dataset(self) -> DemoDataset:
        images = []
        for image_path in sorted(
            self._images_dir.glob("*.jpg"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        ):
            image_id = image_path.stem
            if not IMAGE_ID_PATTERN.match(image_id):
                continue
            images.append(
                DemoImage(
                    image_id=image_id,
                    boxes=self._read_labels(image_id),
                    created_at=datetime.fromtimestamp(
                        image_path.stat().st_mtime, UTC
                    ),
                )
            )

        counts = {name: 0 for name in self._classes}
        for image in images:
            for box in image.boxes:
                if box.class_id < len(self._classes):
                    counts[self._classes[box.class_id]] += 1
        counts["imagens"] = len(images)

        return DemoDataset(classes=self.classes, images=images, counts=counts)

    def add_image(
        self,
        *,
        content_type: str | None,
        image_bytes: bytes,
        boxes: list[DemoBox],
    ) -> DemoImage:
        image = self._validate_and_decode(content_type, image_bytes)

        if len(boxes) > MAX_BOXES_PER_IMAGE:
            raise DemoError(f"No máximo {MAX_BOXES_PER_IMAGE} caixas por foto.")
        for box in boxes:
            if box.class_id >= len(self._classes):
                raise DemoError("Classe inválida.")

        image = _shrink(image, MAX_STORED_SIDE)
        success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not success:
            raise DemoError("Não foi possível salvar a foto.", status_code=500)

        self.ensure_directories()
        image_id = uuid.uuid4().hex
        # Label first: an image without its label file would be read as
        # "no objects" if the process died in between.
        save_bytes_to_file(
            self._labels_dir / f"{image_id}.txt",
            _labels_to_text(boxes).encode("utf-8"),
        )
        save_bytes_to_file(self._images_dir / f"{image_id}.jpg", encoded.tobytes())

        logger.info("Demo image %s saved with %d boxes.", image_id, len(boxes))
        return DemoImage(image_id=image_id, boxes=boxes, created_at=datetime.now(UTC))

    def image_path(self, image_id: str) -> Path:
        self._check_image_id(image_id)
        path = self._images_dir / f"{image_id}.jpg"
        if not path.is_file():
            raise DemoError("Foto não encontrada.", status_code=404)
        return path

    def delete_image(self, image_id: str) -> None:
        self._ensure_not_training()
        path = self.image_path(image_id)
        path.unlink(missing_ok=True)
        (self._labels_dir / f"{image_id}.txt").unlink(missing_ok=True)

    def clear_dataset(self) -> int:
        self._ensure_not_training()
        removed = 0
        for directory in (self._images_dir, self._labels_dir):
            for path in directory.glob("*"):
                if path.is_file():
                    path.unlink(missing_ok=True)
                    removed += directory == self._images_dir
        for cache in self._dir.glob("*.cache"):
            cache.unlink(missing_ok=True)
        return removed

    # --------------------------------------------------------------- training

    def status(self) -> DemoTrainStatus:
        with self._status_lock:
            status = self._status.model_copy()

        meta = self._read_meta()
        status.trained = self.model_path.is_file()
        if status.trained:
            trained_at = meta.get("trained_at")
            status.trained_at = (
                datetime.fromisoformat(trained_at) if trained_at else None
            )
            status.trained_images = meta.get("images")
            status.model_size_mb = round(self.model_path.stat().st_size / 1e6, 1)
            if status.map50 is None:
                status.map50 = meta.get("map50")
        return status

    def start_training(self, epochs: int | None = None) -> DemoTrainStatus:
        dataset = self.list_dataset()
        labelled = [image for image in dataset.images if image.boxes]
        if len(dataset.images) < self._min_images or not labelled:
            raise DemoError(
                f"Tire pelo menos {self._min_images} fotos (com caixas) "
                "antes de treinar."
            )
        missing = [
            name
            for class_id, name in enumerate(self._classes)
            if dataset.counts[name] == 0
        ]
        if missing:
            raise DemoError(
                "Faltam exemplos de: " + ", ".join(missing) + "."
            )

        with self._status_lock:
            if self._status.state in ("running", "stopping"):
                raise DemoError("Já existe um treino em andamento.", 409)
            total = epochs or self._train_epochs
            self._stop_requested = False
            self._status = DemoTrainStatus(
                state="running",
                epochs=total,
                message="Preparando o treino...",
                started_at=datetime.now(UTC),
            )
            self._thread = threading.Thread(
                target=self._run_training,
                args=(total, len(dataset.images)),
                name="demo-training",
                daemon=True,
            )
            self._thread.start()

        return self.status()

    def stop_training(self) -> DemoTrainStatus:
        with self._status_lock:
            if self._status.state == "running":
                self._stop_requested = True
                self._status.state = "stopping"
                self._status.message = (
                    "Parando ao fim desta época e guardando o melhor modelo..."
                )
        return self.status()

    def wait_for_training(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def _run_training(self, epochs: int, image_count: int) -> None:
        started = time.monotonic()
        try:
            data_yaml = self._write_dataset_yaml()
            trainer = YoloService(
                model_path=self._base_model_path,
                expected_sha256=self._base_model_sha256,
                params=self._params,
            )
            model = trainer.load_model()
            best = trainer.train(
                model,
                data_yaml=data_yaml,
                output_dir=(self._dir / "runs").resolve(),
                epochs=epochs,
                imgsz=self._train_imgsz,
                batch=max(1, min(self._train_batch, image_count)),
                on_epoch_end=self._on_epoch_end,
            )
            self._install_model(best, image_count)
        except AppError as exc:
            logger.error("Demo training failed: %s", exc.log_detail or exc.message)
            self._finish("error", "O treino falhou. Veja os logs da API.")
            return
        except Exception:
            logger.exception("Demo training failed.")
            self._finish("error", "O treino falhou. Veja os logs da API.")
            return

        minutes, seconds = divmod(int(time.monotonic() - started), 60)
        self._finish("done", f"Treino concluído em {minutes}min {seconds}s.")

    def _on_epoch_end(self, epoch: int, epochs: int, metrics: dict[str, float]) -> bool:
        with self._status_lock:
            # Ultralytics fires the callback once more for the final
            # validation of best.pt, so the count can pass `epochs`.
            self._status.epochs = epochs
            self._status.epoch = min(epoch, epochs)
            if MAP50_KEY in metrics:
                self._status.map50 = round(metrics[MAP50_KEY], 3)
            if self._status.state == "running":
                self._status.message = f"Época {epoch} de {epochs}"
            return self._stop_requested

    def _finish(self, state: str, message: str) -> None:
        with self._status_lock:
            self._status.state = state  # type: ignore[assignment]
            self._status.message = message
            self._status.finished_at = datetime.now(UTC)

    def _write_dataset_yaml(self) -> Path:
        # JSON is valid YAML, and avoids a YAML dependency and quoting issues.
        content = {
            "path": str(self._dir.resolve()),
            "train": "images",
            "val": "images",
            "names": dict(enumerate(self._classes)),
        }
        path = self._dir / "dataset.yaml"
        save_bytes_to_file(path, json.dumps(content, indent=2).encode("utf-8"))
        return path

    def _install_model(self, best: Path, image_count: int) -> None:
        self.ensure_directories()
        temporary = self.model_path.with_suffix(".pt.tmp")
        shutil.copyfile(best, temporary)
        os.replace(temporary, self.model_path)

        with self._status_lock:
            map50 = self._status.map50
        meta = {
            "trained_at": datetime.now(UTC).isoformat(),
            "images": image_count,
            "map50": map50,
            "sha256": sha256_file(self.model_path),
            "classes": self._classes,
        }
        save_bytes_to_file(
            self._model_dir / "meta.json",
            json.dumps(meta, indent=2).encode("utf-8"),
        )

        with self._predict_lock:
            self._predict_model = None
            self._predict_service = None
        logger.info("Demo model installed: %s", self.model_path)

    # ------------------------------------------------------------- prediction

    def predict(self, *, content_type: str | None, image_bytes: bytes) -> DemoPrediction:
        image = self._validate_and_decode(content_type, image_bytes)
        height, width = image.shape[:2]

        with self._predict_lock:
            service, model = self._get_predict_model()
            started = time.perf_counter()
            detections = service.predict(model, image)
            inference_ms = (time.perf_counter() - started) * 1000

        results = []
        for detection in detections:
            box = detection.box
            results.append(
                DemoDetection(
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                    confidence=round(detection.confidence, 3),
                    good=detection.class_id == 0,
                    x=max(0.0, box.x1 / width),
                    y=max(0.0, box.y1 / height),
                    w=min(1.0, (box.x2 - box.x1) / width),
                    h=min(1.0, (box.y2 - box.y1) / height),
                )
            )

        if any(not item.good for item in results):
            verdict = "defeito"
        elif results:
            verdict = "ok"
        else:
            verdict = "nada"

        return DemoPrediction(
            verdict=verdict,
            detections=results,
            inference_ms=round(inference_ms, 1),
            image_width=width,
            image_height=height,
        )

    def _get_predict_model(self) -> tuple[YoloService, Any]:
        if self._predict_model is None or self._predict_service is None:
            if not self.model_path.is_file():
                raise DemoError("Treine o modelo antes de testar.", status_code=409)
            service = YoloService(
                model_path=self.model_path,
                expected_sha256=self._read_meta().get("sha256"),
                params=self._predict_params,
            )
            self._predict_model = service.load_model()
            self._predict_service = service
        return self._predict_service, self._predict_model

    # ---------------------------------------------------------------- helpers

    def _validate_and_decode(self, content_type: str | None, image_bytes: bytes) -> Any:
        validate_image_size(image_bytes, self._max_image_size_mb)
        validate_image_signature(content_type, image_bytes)
        validate_image_dimensions(image_bytes, self._max_image_pixels)
        return decode_image_bytes(image_bytes)

    def _read_labels(self, image_id: str) -> list[DemoBox]:
        path = self._labels_dir / f"{image_id}.txt"
        if not path.is_file():
            return []
        boxes = []
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                class_id = int(parts[0])
                cx, cy, w, h = (float(value) for value in parts[1:])
                boxes.append(
                    DemoBox(
                        class_id=class_id,
                        x=max(0.0, cx - w / 2),
                        y=max(0.0, cy - h / 2),
                        w=w,
                        h=h,
                    )
                )
            except ValueError:
                continue
        return boxes

    def _read_meta(self) -> dict[str, Any]:
        path = self._model_dir / "meta.json"
        try:
            return dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return {}

    def _ensure_not_training(self) -> None:
        with self._status_lock:
            if self._status.state in ("running", "stopping"):
                raise DemoError("Espere o treino terminar.", status_code=409)

    @staticmethod
    def _check_image_id(image_id: str) -> None:
        if not IMAGE_ID_PATTERN.match(image_id):
            raise DemoError("Foto não encontrada.", status_code=404)


def _labels_to_text(boxes: list[DemoBox]) -> str:
    lines = [
        f"{box.class_id} {box.x + box.w / 2:.6f} {box.y + box.h / 2:.6f} "
        f"{box.w:.6f} {box.h:.6f}"
        for box in boxes
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def _shrink(image: Any, max_side: int) -> Any:
    height, width = image.shape[:2]
    scale = max_side / max(height, width)
    if scale >= 1:
        return image
    size = (round(width * scale), round(height * scale))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
