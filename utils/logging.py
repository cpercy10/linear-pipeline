"""Structured logging + per-stage timing.

- structlog with ConsoleRenderer (dev) or JSON (prod).
- `request_id` is bound to a contextvar at image entry so every downstream log
  line carries it automatically.
- `stage_timer` is the canonical way to time a stage; it logs `stage`,
  `elapsed_ms`, and `gpu_vram_used_mb` on exit (per the skill's logging rules).
"""

from __future__ import annotations

import contextlib
import contextvars
import logging as _stdlib_logging
import time
from typing import Iterator, Optional

import structlog

from utils.gpu_monitor import current_vram_used_mb, cuda_sync
from utils.metrics import get_metrics

# request_id propagates to every log line within an image's processing.
_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)


def _add_request_id(_logger, _name, event_dict):
    event_dict.setdefault("request_id", _request_id_var.get())
    return event_dict


def configure_logging(level: str = "INFO", environment: str = "prod") -> None:
    """Configure structlog once at startup."""
    renderer = (
        structlog.processors.JSONRenderer()
        if environment == "prod"
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _add_request_id,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(_stdlib_logging, level.upper(), _stdlib_logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


def bind_request_id(request_id: str) -> contextvars.Token:
    """Bind a request_id for the current context. Returns a token to reset with."""
    return _request_id_var.set(request_id)


def reset_request_id(token: contextvars.Token) -> None:
    _request_id_var.reset(token)


@contextlib.contextmanager
def stage_timer(
    stage: str,
    *,
    device: Optional[str] = None,
    gpu: bool = False,
    log: Optional[structlog.stdlib.BoundLogger] = None,
) -> Iterator[dict]:
    """Time a pipeline stage and log elapsed_ms + VRAM on exit.

    Set ``gpu=True`` for GPU stages so the clock is taken after
    ``torch.cuda.synchronize()`` (otherwise async kernels make timing meaningless).

    Yields a small dict you can stash extra fields into; they're logged too.
    """
    log = log or get_logger()
    extra: dict = {}
    if gpu:
        cuda_sync(device)
    t0 = time.perf_counter()
    try:
        yield extra
    finally:
        if gpu:
            cuda_sync(device)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        get_metrics().record_stage(stage, elapsed_ms)
        log.info(
            "stage.complete",
            stage=stage,
            elapsed_ms=elapsed_ms,
            gpu_vram_used_mb=current_vram_used_mb(device),
            **extra,
        )
