"""Async remove.bg client (interior + exterior_partial lanes).

Network-bound, so it's gated by a semaphore (not the GPU pool) and retries with
exponential backoff + jitter, honoring `Retry-After` on 429s. One shared
httpx.AsyncClient is reused across calls.
"""

from __future__ import annotations

import asyncio
import io
import random
from typing import Optional

import httpx
from PIL import Image

from config.pipeline_config import PipelineSettings
from processing.exceptions import BackgroundRemovalError, ConfigError
from utils.logging import get_logger

_log = get_logger("remove_bg")

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class BackgroundRemover:
    def __init__(self, settings: PipelineSettings) -> None:
        self._cfg = settings.removebg
        self._api_key = settings.remove_bg_api_key
        # Concurrency: the top-level override wins if set, else the nested default.
        concurrency = settings.removebg_concurrency or self._cfg.concurrency
        self._sem = asyncio.Semaphore(max(1, concurrency))
        self._client: Optional[httpx.AsyncClient] = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._cfg.timeout_s)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def remove(
        self,
        image_bytes: bytes,
        filename: str,
        *,
        add_shadow: Optional[bool] = None,
        size: Optional[str] = None,
    ) -> Image.Image:
        """Return an RGBA image with the background removed. Raises
        BackgroundRemovalError after exhausting retries.

        ``add_shadow`` / ``size`` override the config defaults per call, so the
        exterior-full manual-composite lane can pick its own shadow policy without
        affecting the interior/partial lanes."""
        if not self._api_key:
            raise ConfigError("REMOVE_BG_API_KEY is not set — required for the remove.bg lanes")

        client = self._ensure_client()
        use_shadow = self._cfg.add_shadow if add_shadow is None else add_shadow
        data = {
            "size": size or self._cfg.size,
            "add_shadow": "true" if use_shadow else "false",
            "shadow_opacity": str(self._cfg.shadow_opacity),
        }
        headers = {"X-Api-Key": self._api_key}

        async with self._sem:
            last_err: Optional[str] = None
            for attempt in range(self._cfg.max_retries + 1):
                try:
                    resp = await client.post(
                        self._cfg.endpoint,
                        files={"image_file": (filename, image_bytes)},
                        data=data,
                        headers=headers,
                    )
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    last_err = f"network error: {exc}"
                    await self._backoff(attempt)
                    continue

                if resp.status_code == 200:
                    return Image.open(io.BytesIO(resp.content)).convert("RGBA")

                if resp.status_code in _RETRYABLE_STATUS and attempt < self._cfg.max_retries:
                    retry_after = self._parse_retry_after(resp)
                    _log.warning("remove_bg.retry", status=resp.status_code,
                                 attempt=attempt + 1, retry_after=retry_after, filename=filename)
                    await self._backoff(attempt, retry_after)
                    continue

                # non-retryable or out of retries
                raise BackgroundRemovalError(
                    f"remove.bg {resp.status_code} for {filename}: {resp.text[:200]}"
                )

            raise BackgroundRemovalError(
                f"remove.bg exhausted retries for {filename} ({last_err})"
            )

    async def _backoff(self, attempt: int, retry_after: Optional[float] = None) -> None:
        if retry_after is not None:
            await asyncio.sleep(retry_after)
            return
        # exponential backoff with jitter: 0.5, 1, 2, ... + up to 0.5s jitter
        delay = min(0.5 * (2 ** attempt), 10.0) + random.random() * 0.5
        await asyncio.sleep(delay)

    @staticmethod
    def _parse_retry_after(resp: httpx.Response) -> Optional[float]:
        val = resp.headers.get("Retry-After")
        if not val:
            return None
        try:
            return float(val)
        except ValueError:
            return None
