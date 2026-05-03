import logging
import sys


def setup_logging(log_level: str) -> None:
    numeric_level = _get_numeric_level(log_level)

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )


def _get_numeric_level(log_level: str) -> int:
    normalized_level = log_level.upper()
    numeric_level = logging.getLevelName(normalized_level)

    if isinstance(numeric_level, int):
        return numeric_level

    raise ValueError(f"Invalid log level: {log_level}")