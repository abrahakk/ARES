from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml


class CheckpointManager:
    def __init__(self, root_dir: str = "models/checkpoints") -> None:
        self.root_dir = Path(root_dir)
        self.latest_dir = self.root_dir / "latest"
        self.best_dir = self.root_dir / "best"
        self.registry_path = self.root_dir.parent / "metadata" / "model_registry.yaml"

        self.latest_dir.mkdir(parents=True, exist_ok=True)
        self.best_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        state: Dict[str, Any],
        run_name: str,
        step: int,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        config_path: Optional[str] = None,
    ) -> Path:
        filename = f"{run_name}_step{step}.pt"
        checkpoint_path = self.latest_dir / filename
        torch.save(state, checkpoint_path)

        if is_best:
            best_path = self.best_dir / f"{run_name}_best.pt"
            torch.save(state, best_path)

        self._update_registry(run_name, step, checkpoint_path, metrics or {}, is_best, config_path)
        return checkpoint_path

    def load(self, path: str, map_location: str = "cpu") -> Dict[str, Any]:
        return torch.load(path, map_location=map_location)

    def load_best(self, run_name: str, map_location: str = "cpu") -> Dict[str, Any]:
        path = self.best_dir / f"{run_name}_best.pt"
        if not path.exists():
            raise FileNotFoundError(f"no best checkpoint found for run '{run_name}' at {path}")
        return self.load(str(path), map_location=map_location)

    def _update_registry(
        self,
        run_name: str,
        step: int,
        checkpoint_path: Path,
        metrics: Dict[str, float],
        is_best: bool,
        config_path: Optional[str],
    ) -> None:
        registry = {}
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                registry = yaml.safe_load(f) or {}

        entries = registry.setdefault(run_name, [])
        entries.append(
            {
                "step": step,
                "path": str(checkpoint_path),
                "metrics": metrics,
                "is_best": is_best,
                "config": config_path,
                "timestamp": dt.datetime.utcnow().isoformat() + "Z",
            }
        )

        with open(self.registry_path, "w") as f:
            yaml.safe_dump(registry, f, sort_keys=False)
