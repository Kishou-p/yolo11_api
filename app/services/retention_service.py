import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class RetentionService:
    """Deletes uploads, results and leftover temp files older than the
    configured retention, so the disk does not fill up on a busy line."""

    def __init__(
        self,
        directories: list[Path],
        retention_days: int,
    ) -> None:
        self._directories = directories
        self._retention_seconds = retention_days * 24 * 60 * 60

    def purge_expired_files(self, now: float | None = None) -> int:
        cutoff = (now if now is not None else time.time()) - self._retention_seconds
        deleted_count = 0

        for directory in self._directories:
            if not directory.is_dir():
                continue

            for path in directory.iterdir():
                if self._is_expired(path, cutoff) and self._delete(path):
                    deleted_count += 1

        if deleted_count:
            logger.info(
                "Retention cleanup removed files",
                extra={"deleted_count": deleted_count},
            )

        return deleted_count

    def _is_expired(self, path: Path, cutoff: float) -> bool:
        if path.name == ".gitkeep" or not path.is_file():
            return False

        try:
            return path.stat().st_mtime < cutoff
        except FileNotFoundError:
            return False

    def _delete(self, path: Path) -> bool:
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        except OSError:
            logger.exception("Retention cleanup failed to delete %s", path)
            return False

        return True
