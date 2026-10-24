from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

REQUIRED_PYTHON_MINOR = 11

REQUIRED_PACKAGES = [
    "numpy", "yaml", "torch", "onnx", "onnxruntime",
    "fastapi", "uvicorn", "pydantic", "redis", "prometheus_client",
]

REQUIRED_CONFIG_FILES = [
    "config/rewards.yaml", "config/robots.yaml", "config/simulation.yaml",
    "config/system.yaml", "config/training.yaml",
]

OPTIONAL_PACKAGES = ["scipy"]  


def check_python_version() -> bool:
    ok = sys.version_info.major == 3 and sys.version_info.minor >= REQUIRED_PYTHON_MINOR
    status = "OK" if ok else "FAIL"
    logger.info("[%s] Python version: %s (need >= 3.%d)", status, sys.version.split()[0], REQUIRED_PYTHON_MINOR)
    return ok


def check_packages() -> bool:
    all_ok = True
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
            logger.info("[OK] required package importable: %s", package)
        except ImportError:
            logger.error("[FAIL] required package NOT importable: %s (pip install -r requirements.txt)", package)
            all_ok = False

    for package in OPTIONAL_PACKAGES:
        try:
            importlib.import_module(package)
            logger.info("[OK] optional package importable: %s", package)
        except ImportError:
            logger.warning("[WARN] optional package not importable: %s (a fallback path will be used)", package)

    return all_ok


def check_config_files() -> bool:
    all_ok = True
    for path_str in REQUIRED_CONFIG_FILES:
        path = Path(path_str)
        if path.exists():
            logger.info("[OK] config file present: %s", path_str)
        else:
            logger.error("[FAIL] config file MISSING: %s", path_str)
            all_ok = False
    return all_ok


def check_webots_env() -> None:
    
    webots_home = os.environ.get("WEBOTS_HOME")
    if webots_home and Path(webots_home).exists():
        logger.info("[OK] WEBOTS_HOME set and exists: %s", webots_home)
    else:
        logger.warning(
            "[INFO] WEBOTS_HOME not set (or path does not exist) - fine for API-only or "
            "evaluation-only use; required for scripts/start_simulation.py and live training."
        )


def check_redis_reachable() -> None:
    
    try:
        import redis
        from coordination.redis_client import RedisSettings

        settings = RedisSettings.from_yaml()
        client = redis.Redis(host=settings.host, port=settings.port, socket_timeout=1.0)
        client.ping()
        logger.info("[OK] Redis reachable at %s:%d", settings.host, settings.port)
    except Exception as exc:  
        logger.warning(
            "[INFO] Redis not reachable right now (%s) - fine if you haven't started it yet "
            "(see `docker compose up` in the root README.md); required for coordination/ features.",
            exc,
        )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("=== ARES environment check ===")
    python_ok = check_python_version()
    packages_ok = check_packages()
    config_ok = check_config_files()
    check_webots_env()
    check_redis_reachable()

    if python_ok and packages_ok and config_ok:
        logger.info("=== All required checks passed. ===")
        return 0

    logger.error("=== One or more required checks FAILED - see above. ===")
    return 1


if __name__ == "__main__":
    sys.exit(main())
