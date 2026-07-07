"""Host-side async client for the WARM Blender worker (render/worker_entry.py).

Spawns ONE long-lived ``blender -b <master>.blend --python worker_entry.py``
process, talks to it over a localhost socket (newline-delimited JSON request /
response), and exposes::

    worker = BlenderWorker(settings)
    await worker.start()
    meta = await worker.render(camera, disc_diam, out_jpg, out_json,
                               photo_w, photo_h)
    await worker.shutdown()

The worker process loads the scene / materials / car-free setup ONCE and loops
over jobs, so per-image latency drops from a cold ``blender -b`` launch (seconds
of scene + material load each call) to just the render.

Robustness:
  * Per-job timeout (``BlenderConfig.job_timeout_s``) → ``BlenderRenderError``.
  * The Blender child's stderr/stdout is drained to the logger.
  * Nonzero exit / crash mid-job is detected; the in-flight render fails with
    ``BlenderRenderError`` and the process is auto-restarted before the next job.
  * One job in flight at a time (Blender is single-threaded); the public API is
    serialized by an asyncio.Lock so concurrent callers queue cleanly. The
    single-slot async gate (runtime/blender_pool.py) layers on top of this.

Why connect-back instead of the host connecting in: Blender's headless process
is slow and unpredictable to come up, so the HOST opens a listening socket on an
ephemeral port and passes ``--port`` to the child; the child dials back once it
has finished the (slow) scene/material load. The first line the child sends is
``{"ready": true}``, which is how ``start()`` knows warm-up finished.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Optional

from config.pipeline_config import PipelineSettings, get_settings
from processing.exceptions import BlenderRenderError, ConfigError
from utils.logging import get_logger, stage_timer

_log = get_logger("blender_worker")


class BlenderWorker:
    """Supervises one warm Blender process and renders plates over a socket."""

    def __init__(self, settings: Optional[PipelineSettings] = None) -> None:
        self._s = settings or get_settings()
        self._bcfg = self._s.blender

        self._proc: Optional[asyncio.subprocess.Process] = None
        self._server: Optional[asyncio.AbstractServer] = None
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._conn_ready: Optional[asyncio.Future] = None
        self._stderr_task: Optional[asyncio.Task] = None

        # One render in flight at a time (Blender is single-threaded).
        self._job_lock = asyncio.Lock()
        self._job_seq = 0
        self._started = False
        self._closing = False
        # Set once a restart exhausts its bounded retries; every subsequent render()
        # fails fast with a clear fatal instead of spinning on a broken respawn.
        self._dead = False

        self._validate_paths()

    # ── path validation ─────────────────────────────────────────────────────
    def _validate_paths(self) -> None:
        exe = Path(self._s.blender_exe)
        master = Path(self._s.master_blend)
        engine = Path(self._s.studio_engine_dir)
        worker_entry = Path(__file__).resolve().parent / "worker_entry.py"
        if not exe.exists():
            raise ConfigError(f"Blender executable not found: {exe}")
        if not master.exists():
            raise ConfigError(f"master .blend not found: {master}")
        if not engine.exists():
            raise ConfigError(f"studio engine dir not found: {engine}")
        if not worker_entry.exists():
            raise ConfigError(f"worker_entry.py not found: {worker_entry}")
        self._worker_entry = worker_entry

    # ── lifecycle ────────────────────────────────────────────────────────────
    async def start(self) -> None:
        """Spawn Blender and wait for the warm worker to finish loading."""
        if self._started:
            return
        self._closing = False
        await self._spawn()
        self._started = True

    @property
    def release_after_render(self) -> bool:
        return bool(getattr(self._bcfg, "release_after_render", False))

    async def _spawn(self) -> None:
        loop = asyncio.get_running_loop()
        self._conn_ready = loop.create_future()

        # Host listens on an ephemeral localhost port; the child dials back.
        self._server = await asyncio.start_server(
            self._on_child_connect, host="127.0.0.1", port=0
        )
        sockets = self._server.sockets or []
        if not sockets:
            raise BlenderRenderError("failed to open host listening socket")
        port = sockets[0].getsockname()[1]

        cmd = [
            str(self._s.blender_exe),
            "-b", str(self._s.master_blend),
            "--python", str(self._worker_entry),
            "--",
            "--port", str(port),
            "--host", "127.0.0.1",
            "--samples", str(self._bcfg.render_samples),
            "--long-edge", str(self._bcfg.render_long_edge),
            "--materials", str(self._s.materials_2k_dir),
        ]
        env = dict(os.environ)
        env.setdefault("MOTUVA_LIB2K", str(self._s.materials_2k_dir))

        _log.info("blender.spawn", port=port, exe=str(self._s.blender_exe),
                  master=str(self._s.master_blend))
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
        self._stderr_task = asyncio.create_task(self._drain_child_output())

        # Wait for the child to connect back AND announce readiness, bounded by
        # startup_timeout_s. A nonzero early exit surfaces as BlenderRenderError.
        #
        # The timeout MUST live here in _spawn() (not only in start()): _restart()
        # calls _spawn() directly while render() still holds _job_lock, so an unbounded
        # `await self._conn_ready` on a child that never dials back would block the job
        # lock FOREVER and, via the single-slot BlenderPool, wedge the whole batch.
        try:
            await asyncio.wait_for(
                self._conn_ready, timeout=self._bcfg.startup_timeout_s
            )
        except asyncio.TimeoutError as e:
            await self._kill()
            raise BlenderRenderError(
                f"warm Blender worker did not become ready within "
                f"{self._bcfg.startup_timeout_s}s"
            ) from e

    async def _on_child_connect(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        # Only the warm worker connects; keep the first connection.
        if self._reader is not None:
            writer.close()
            return
        self._reader = reader
        self._writer = writer
        try:
            line = await reader.readline()
            msg = json.loads(line.decode("utf-8")) if line else {}
        except Exception as e:  # noqa: BLE001
            if self._conn_ready and not self._conn_ready.done():
                self._conn_ready.set_exception(
                    BlenderRenderError(f"bad ready handshake: {e}")
                )
            return
        if msg.get("ready") and self._conn_ready and not self._conn_ready.done():
            _log.info("blender.ready")
            self._conn_ready.set_result(True)
        elif self._conn_ready and not self._conn_ready.done():
            self._conn_ready.set_exception(
                BlenderRenderError(f"unexpected handshake: {msg}")
            )

    async def _drain_child_output(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        try:
            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    _log.debug("blender.stdout", line=text)
        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001
            pass

    # ── rendering ────────────────────────────────────────────────────────────
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
        """Render ONE plate. Returns the worker's meta dict (with car_spot_px,
        pixels_per_metre, camera.resolution). Raises BlenderRenderError on any
        failure; the worker is auto-restarted before raising so the next call is
        clean.

        ``camera`` carries the continuous export values:
        ``azimuth_deg / distance_m / cam_height_m / elevation_deg / focal_mm /
        roll_deg`` (whatever the estimator produced; missing keys fall back to
        render_server defaults). ``disc_diam`` is sized to the RESIZED car.
        """
        if self._dead:
            raise BlenderRenderError(
                "warm Blender worker is permanently down (restart retries exhausted)"
            )
        if not self._started:
            await self.start()

        async with self._job_lock:
            job = {
                "camera": camera,
                "disc_diam": float(disc_diam),
                "photo_w": int(photo_w),
                "photo_h": int(photo_h),
                "out_jpg": str(out_jpg),
                "out_json": str(out_json),
                "studio_frame": bool(studio_frame),
                "no_threeq_zoom": bool(no_threeq_zoom),
            }
            if studio is not None:
                job["studio"] = studio
            if long_edge is not None:
                job["long_edge"] = int(long_edge)
            if samples is not None:
                job["samples"] = int(samples)

            # _exchange stamps a unique monotonic id and matches the reply strictly.
            with stage_timer("plate_render", log=_log) as extra:
                reply = await self._exchange(job, self._bcfg.job_timeout_s)
                extra["job_id"] = reply.get("id")
                extra["plate_ms"] = reply.get("elapsed_ms")

            if not reply.get("ok"):
                # Worker reported a render failure; it stays alive (the exception
                # was caught inside its loop) but we surface it to the caller.
                err = reply.get("error", "unknown render error")
                raise BlenderRenderError(f"plate render failed: {err}")

            meta = reply.get("meta") or {}
            # Defensive: the worker already validates these, but guard the contract.
            if "car_spot_px" not in meta or "pixels_per_metre" not in meta:
                raise BlenderRenderError(
                    "render meta missing car_spot_px / pixels_per_metre"
                )
            return meta

    async def _exchange(self, msg: dict, timeout: float) -> dict:
        """Send one request, await ITS reply, with timeout + crash detection.

        Every outgoing message (render job OR control message like ping/shutdown) is
        stamped with a unique monotonic id from the SAME sequence, and the reply is
        matched STRICTLY against it. A reply whose id does not match (a stale/buffered
        line left over from a previous slow exchange, or a duplicate) is DRAINED and
        skipped — we keep reading until the matching id arrives — rather than treating
        the benign desync as fatal and restarting the worker (which previously could
        kill a healthy worker on a ping that happened to read a stale render reply).
        On timeout or process death the worker is killed + restarted and a
        BlenderRenderError is raised.
        """
        if self._writer is None or self._reader is None:
            raise BlenderRenderError("worker connection not established")

        self._job_seq += 1
        req_id = self._job_seq
        out = dict(msg)
        out["id"] = req_id

        try:
            self._writer.write((json.dumps(out) + "\n").encode("utf-8"))
            await self._writer.drain()
        except Exception as e:  # noqa: BLE001
            await self._restart()
            raise BlenderRenderError(f"failed to send job to worker: {e}") from e

        while True:
            try:
                line = await asyncio.wait_for(self._reader.readline(), timeout=timeout)
            except asyncio.TimeoutError as e:
                await self._restart()
                raise BlenderRenderError(
                    f"plate render timed out after {timeout}s (job {req_id})"
                ) from e

            if not line:
                # EOF — the worker died mid-job.
                rc = await self._exit_code()
                await self._restart()
                raise BlenderRenderError(
                    f"warm Blender worker exited mid-render (rc={rc})"
                )

            try:
                reply = json.loads(line.decode("utf-8"))
            except Exception as e:  # noqa: BLE001
                await self._restart()
                raise BlenderRenderError(f"malformed worker reply: {e}") from e

            reply_id = reply.get("id")
            if reply_id == req_id:
                return reply
            # Stale/duplicate line from an earlier exchange — drain and keep reading
            # for OUR reply. Older ids are benign leftovers; a NEWER id should be
            # impossible (we hold _job_lock) and signals real corruption → restart.
            if isinstance(reply_id, int) and reply_id > req_id:
                await self._restart()
                raise BlenderRenderError(
                    f"worker reply id {reply_id} > request id {req_id} — stream corrupt"
                )
            _log.warning("blender.stale_reply", reply_id=reply_id, expected_id=req_id)

    async def _exit_code(self) -> Optional[int]:
        if self._proc is None:
            return None
        try:
            return await asyncio.wait_for(self._proc.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            return None

    # ── teardown / restart ────────────────────────────────────────────────────
    async def _close_conn(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:  # noqa: BLE001
                pass
        self._reader = None
        self._writer = None
        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:  # noqa: BLE001
                pass
            self._server = None

    async def _kill(self) -> None:
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            self._stderr_task = None
        await self._close_conn()
        if self._proc is not None and self._proc.returncode is None:
            try:
                self._proc.kill()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                pass
        self._proc = None

    async def _restart(self) -> None:
        """Kill + respawn the worker with bounded retry + linear backoff.

        Called from inside _exchange() while render() holds _job_lock, so each attempt's
        _spawn() is startup_timeout_s-bounded (it cannot block the lock forever). If
        every attempt fails the worker is marked permanently dead so subsequent
        render()s fail fast with a clear fatal rather than spin on a broken respawn.
        State (_started / _reader / _writer / _proc) is left consistent on failure.
        """
        if self._closing:
            return
        await self._kill()
        self._started = False

        attempts = max(1, int(self._bcfg.max_restart_attempts))
        last_exc: Optional[Exception] = None
        for i in range(1, attempts + 1):
            _log.warning("blender.restart", attempt=i, max_attempts=attempts)
            try:
                await self._spawn()
                self._started = True
                _log.info("blender.restart.ok", attempt=i)
                return
            except Exception as exc:  # noqa: BLE001 — bounded; we retry or give up below
                last_exc = exc
                self._started = False
                await self._kill()   # leave reader/writer/proc cleanly None for the next try
                if i < attempts:
                    await asyncio.sleep(i * max(0.0, float(self._bcfg.restart_backoff_s)))

        # All attempts exhausted — hard-down the worker.
        self._dead = True
        _log.error("blender.restart.exhausted", attempts=attempts,
                   error=str(last_exc) if last_exc else None)
        raise BlenderRenderError(
            f"warm Blender worker could not be restarted after {attempts} attempts"
            + (f": {last_exc}" if last_exc else "")
        )

    async def shutdown(self) -> None:
        """Politely ask the worker to exit, then tear down."""
        self._closing = True
        if self._writer is not None and self._reader is not None:
            try:
                self._writer.write(
                    (json.dumps({"cmd": "shutdown"}) + "\n").encode("utf-8")
                )
                await self._writer.drain()
                await asyncio.wait_for(self._reader.readline(), timeout=10.0)
            except Exception:  # noqa: BLE001
                pass
        await self._kill()
        self._started = False

    # ── health ─────────────────────────────────────────────────────────────────
    async def ping(self) -> bool:
        """Round-trip a ping through the worker; True if it answers."""
        if not self._started:
            return False
        async with self._job_lock:
            try:
                reply = await self._exchange({"cmd": "ping"}, timeout=15.0)
                return bool(reply.get("pong"))
            except BlenderRenderError:
                return False
