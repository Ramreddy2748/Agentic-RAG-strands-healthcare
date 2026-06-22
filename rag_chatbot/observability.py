from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import json
import logging
import os
from time import perf_counter
from typing import Any
from uuid import uuid4


LOGGER_NAME = "rag_chatbot"


@dataclass(frozen=True)
class PipelineTimings:
    """Elapsed milliseconds for each query-time pipeline stage."""

    routing_ms: float
    retrieval_ms: float
    fusion_ms: float
    reranking_ms: float
    answer_generation_ms: float
    total_ms: float


class StageTimer:
    """Small monotonic timer used for one pipeline stage."""

    def __init__(self) -> None:
        self.started_at = perf_counter()

    def elapsed_ms(self) -> float:
        return round((perf_counter() - self.started_at) * 1000, 2)


def new_request_id() -> str:
    """Create a request identifier for logs and API responses."""
    return uuid4().hex


def get_logger() -> logging.Logger:
    """Configure and return the application logger."""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    logger.propagate = False
    return logger


def log_event(event: str, *, level: int = logging.INFO, **fields: Any) -> None:
    """Emit one machine-readable JSON log event."""
    payload = {"event": event, **fields}
    get_logger().log(level, json.dumps(payload, default=serialize_log_value))


def serialize_log_value(value: Any) -> Any:
    """Convert known values into JSON-compatible log fields."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return str(value)
