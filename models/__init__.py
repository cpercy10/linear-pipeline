"""Model wrappers + device-aware ModelManager.

Submodules are imported directly (e.g. `from models.model_manager import ModelManager`)
to avoid pulling heavy deps (torch/timm/ultralytics/diffusers) on bare `import models`.
"""
