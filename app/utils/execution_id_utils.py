import re
from typing import Final

from app.exceptions.app_exceptions import InvalidExecutionIdError

EXECUTION_ID_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[a-f0-9]{32}$"
)


def validate_execution_id(execution_id: str) -> None:
    if not EXECUTION_ID_PATTERN.fullmatch(execution_id):
        raise InvalidExecutionIdError(execution_id)