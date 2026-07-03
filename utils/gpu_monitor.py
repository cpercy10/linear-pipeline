"""GPU enumeration, VRAM polling, OOM guard, and instance-capacity math.

This module is the foundation of the "read what GPU we're on and load as many
model instances as fit" requirement. It deliberately imports only torch +
structlog (no project logging module) to avoid an import cycle.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Iterator, List, Optional

import structlog
import torch

from processing.exceptions import PipelineVRAMError

_log = structlog.get_logger("gpu_monitor")

_BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    total_mb: int
    free_mb: int

    @property
    def device_str(self) -> str:
        return f"cuda:{self.index}"


def cuda_available() -> bool:
    return torch.cuda.is_available()


def device_count() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def _normalize_index(device: Optional[str | int]) -> Optional[int]:
    if device is None:
        return None
    if isinstance(device, int):
        return device
    if device.startswith("cuda:"):
        return int(device.split(":")[-1])
    if device == "cuda":
        return torch.cuda.current_device()
    return None


def device_mem_mb(index: int) -> tuple[int, int]:
    """(free_mb, total_mb) for a device, querying the driver directly."""
    free_b, total_b = torch.cuda.mem_get_info(index)
    return free_b // _BYTES_PER_MB, total_b // _BYTES_PER_MB


def list_devices() -> List[DeviceInfo]:
    infos: List[DeviceInfo] = []
    for i in range(device_count()):
        props = torch.cuda.get_device_properties(i)
        free_mb, total_mb = device_mem_mb(i)
        infos.append(DeviceInfo(index=i, name=props.name, total_mb=total_mb, free_mb=free_mb))
    return infos


def current_vram_used_mb(device: Optional[str | int] = None) -> int:
    """Process-allocated VRAM in MB (for log lines). Best-effort; never raises."""
    if not torch.cuda.is_available():
        return 0
    try:
        idx = _normalize_index(device)
        return int(torch.cuda.memory_allocated(idx) / _BYTES_PER_MB)
    except Exception:
        return 0


def cuda_sync(device: Optional[str | int] = None) -> None:
    if not torch.cuda.is_available():
        return
    idx = _normalize_index(device)
    if idx is None:
        torch.cuda.synchronize()
    else:
        torch.cuda.synchronize(idx)


def empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def instances_that_fit(free_mb: int, footprint_mb: int, safety_margin_mb: int) -> int:
    """How many additional instances of `footprint_mb` fit in `free_mb`, keeping
    `safety_margin_mb` free at all times. Never negative."""
    usable = free_mb - safety_margin_mb
    if usable <= 0 or footprint_mb <= 0:
        return 0
    return max(0, usable // footprint_mb)


@contextlib.contextmanager
def oom_guard(stage: str, device: Optional[str | int] = None) -> Iterator[None]:
    """Catch CUDA OOM in a GPU op, flush cache + log VRAM stats, and re-raise as
    PipelineVRAMError so the caller can retry/reject a single image."""
    try:
        yield
    except torch.cuda.OutOfMemoryError as exc:
        idx = _normalize_index(device)
        free_mb, total_mb = device_mem_mb(idx if idx is not None else 0)
        empty_cache()
        _log.error(
            "gpu.oom",
            stage=stage,
            device=device,
            free_mb=free_mb,
            total_mb=total_mb,
            allocated_mb=current_vram_used_mb(device),
        )
        raise PipelineVRAMError(f"CUDA OOM in stage '{stage}' on {device}") from exc
