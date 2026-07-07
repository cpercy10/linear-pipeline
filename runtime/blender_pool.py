"""Single-slot async gate to the ONE warm Blender worker.

There is exactly one warm Blender process (``render/blender_worker.py``) — Blender is
single-threaded and the scene/materials are loaded once and kept resident, so renders
MUST serialize through a single slot. This pool is that slot.

The worker can optionally release Blender after each render so the later FLUX refine
stage can take the GPU without a resident Blender scene competing for VRAM.

    pool = BlenderPool(worker)
    await pool.start()
    meta = await pool.render(camera, disc_diam, out_jpg, out_json, photo_w=W, photo_h=H)
    await pool.shutdown()

Construct INSIDE a running event loop (the asyncio primitives bind to it).
"""

from __future__ import annotations

import asyncio
from typing import Dict, Optional

from render.blender_worker import BlenderWorker
from utils.logging import get_logger

_log = get_logger("blender_pool")


class BlenderPool:
    """Async single-slot gate to the warm Blender worker."""

    def __init__(self, worker: BlenderWorker) -> None:
        self._worker = worker
        # Single render slot — only one job dispatched to the warm worker at once.
        self._slot = asyncio.Semaphore(1)
        self._started = False

    @property
    def size(self) -> int:
        """One warm worker → one slot."""
        return 1

    async def start(self) -> None:
        """Spawn + warm the underlying worker (idempotent)."""
        if self._started:
            return
        await self._worker.start()
        self._started = True
        _log.info("blender_pool.ready")

    async def render(
        self,
        camera: Dict[str, float],
        disc_diam: float,
        out_jpg: str,
        out_json: str,
        *,
        photo_w: int,
        photo_h: int,
        studio: Optional[Dict[str, str]] = None,
        studio_frame: bool = False,
        no_threeq_zoom: bool = False,
        long_edge: Optional[int] = None,
        samples: Optional[int] = None,
    ) -> Dict:
        """Acquire the single render slot, then render one plate. Blocks (awaits) when
        another render holds the slot. Returns the worker's meta dict; propagates
        ``BlenderRenderError`` on failure.
        """
        if not self._started:
            await self.start()

        async with self._slot:
            try:
                return await self._worker.render(
                    camera,
                    disc_diam,
                    out_jpg,
                    out_json,
                    photo_w=photo_w,
                    photo_h=photo_h,
                    studio=studio,
                    studio_frame=studio_frame,
                    no_threeq_zoom=no_threeq_zoom,
                    long_edge=long_edge,
                    samples=samples,
                )
            finally:
                if self._worker.release_after_render:
                    _log.info("blender_pool.release_after_render")
                    await self.shutdown()

    async def ping(self) -> bool:
        return await self._worker.ping()

    async def shutdown(self) -> None:
        await self._worker.shutdown()
        self._started = False
