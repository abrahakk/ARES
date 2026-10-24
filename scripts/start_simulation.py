from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)

WORLD_CHOICES = {
    "simple": "simulation/webots/worlds/simple.wbt",
    "maze": "simulation/webots/worlds/maze.wbt",
    "dynamic": "simulation/webots/worlds/dynamic.wbt",
    "disaster_zone": "simulation/webots/worlds/disaster_zone.wbt",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--world", type=str, default="simulation/webots/worlds/maze.wbt",
        help=f"Path to a .wbt file, or one of the shorthand names: {list(WORLD_CHOICES)}",
    )
    parser.add_argument("--mode", type=str, default="frontier_baseline", choices=["policy", "frontier_baseline"],
                         help="Sets ARES_CONTROLLER_MODE - see simulation/webots/controllers/robot_controller/__init__.py")
    parser.add_argument("--model-path", type=str, default="models/onnx/ares_policy.onnx",
                         help="Sets ARES_MODEL_PATH, used only when --mode policy.")
    parser.add_argument("--scenario", type=str, default="exploration", choices=["exploration", "robot_failure"],
                         help="Sets ARES_SCENARIO, read by the supervisor controller.")
    parser.add_argument("--headless", action="store_true", help="Run without rendering (--batch --no-rendering).")
    parser.add_argument("--webots-bin", type=str, default=os.environ.get("WEBOTS_HOME", "") and
                         os.path.join(os.environ["WEBOTS_HOME"], "webots") or "webots",
                         help="Path to the webots executable; defaults to $WEBOTS_HOME/webots, or 'webots' on PATH.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    world_path = WORLD_CHOICES.get(args.world, args.world)
    if not os.path.exists(world_path):
        logger.error("world file not found: %s", world_path)
        return 1

    env = os.environ.copy()
    env["ARES_CONTROLLER_MODE"] = args.mode
    env["ARES_MODEL_PATH"] = args.model_path
    env["ARES_SCENARIO"] = args.scenario

    command = [args.webots_bin]
    if args.headless:
        command += ["--batch", "--no-rendering"]
    command += ["--stdout", "--stderr", world_path]

    logger.info("launching: %s", " ".join(command))
    logger.info("environment: ARES_CONTROLLER_MODE=%s ARES_SCENARIO=%s ARES_MODEL_PATH=%s",
                args.mode, args.scenario, args.model_path)

    try:
        result = subprocess.run(command, env=env)
    except FileNotFoundError:
        logger.error(
            "could not find the webots executable at '%s'. Set WEBOTS_HOME or pass --webots-bin explicitly.",
            args.webots_bin,
        )
        return 1

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
