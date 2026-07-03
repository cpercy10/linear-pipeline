"""Runtime concurrency layer.

  * ``blender_pool.BlenderPool`` — single-slot async gate to the one warm Blender
    worker.

Submodules are imported directly (e.g. ``from runtime.blender_pool import
BlenderPool``) to avoid pulling heavy deps on a bare ``import runtime``.
"""
