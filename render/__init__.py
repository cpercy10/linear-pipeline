"""Render layer — the warm Blender worker.

``BlenderWorker`` (host side) supervises a single long-lived Blender process
running ``worker_entry.py``, which reuses Module 2's render.py logic in a warm
loop (scene + materials loaded once, GPU Cycles). ``runtime/blender_pool.py``
layers a single-slot async gate on top of this for the VRAM controller.

``worker_entry`` is NOT imported here: it is meant to be run *inside* Blender's
bundled interpreter (it imports ``bpy``), never in the host process.
"""

from render.blender_worker import BlenderWorker

__all__ = ["BlenderWorker"]
