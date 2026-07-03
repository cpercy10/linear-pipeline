"""CLI entry point for the Motuva unified batch pipeline (ADAPTED from Module 3).

Run from inside the package directory so the top-level subpackages (config, models,
processing, perspective, stages, render, runtime, utils) are importable:

    cd motuva_pipeline
    python main.py                         # uses paths from .env
    python main.py --input-dir /data/in --output-dir /data/out
    python main.py --background /data/bg.jpg      # exterior-partial background

Startup order (BLUEPRINT §5/§6):
  1. Load the unified settings (.env + MOTOCUT_ env vars), apply CLI overrides.
  2. Configure structured logging.
  3. Load the visual-occupancy registry (fail fast on a bad orientation.yaml BEFORE
     touching any heavy models).
  4. Build the ModelManager — loads the small models + DINOv2 index + GeoCalib.
  5. Run the DirectoryRunner — starts the warm Blender worker, drains the batch across
     the tiers (preprocess → plate render → remove.bg cutout + manual composite), and
     shuts the worker down.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from config.pipeline_config import get_orientation_registry, get_settings
from models.model_manager import ModelManager
from runner import DirectoryRunner
from utils.logging import configure_logging, get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Motuva unified car-image batch pipeline")
    parser.add_argument(
        "--input-dir", "--input", dest="input_dir", help="override MOTOCUT_INPUT_DIR"
    )
    parser.add_argument(
        "--output-dir", "--output", dest="output_dir", help="override MOTOCUT_OUTPUT_DIR"
    )
    parser.add_argument(
        "--background",
        help="override MOTOCUT_BACKGROUND_IMAGE (exterior-partial lane background)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    if args.input_dir:
        settings.input_dir = Path(args.input_dir)
    if args.output_dir:
        settings.output_dir = Path(args.output_dir)
    if args.background:
        settings.background_image = Path(args.background)

    configure_logging(level=settings.log_level, environment=settings.environment)
    log = get_logger("main")

    # Fail fast on a bad orientation YAML before loading any heavy models.
    registry = get_orientation_registry()

    log.info(
        "startup",
        input_dir=str(settings.input_dir),
        output_dir=str(settings.output_dir),
        background_image=str(settings.background_image),
        blender_exe=str(settings.blender_exe),
    )

    manager = ModelManager.build(settings)
    runner = DirectoryRunner(settings, manager, registry)
    asyncio.run(runner.run())


if __name__ == "__main__":
    main()
