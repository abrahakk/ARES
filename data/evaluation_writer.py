from __future__ import annotations

import statistics
from pathlib import Path
from typing import Dict, Iterator, List

from data.schemas import EvaluationRun

DEFAULT_ROOT = Path("data/evaluation")


class EvaluationWriter:
    def __init__(self, experiment_name: str, root: Path = DEFAULT_ROOT) -> None:
        self.experiment_name = experiment_name
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / f"{experiment_name}.jsonl"

    def write(self, run: EvaluationRun) -> None:
        with open(self.path, "a") as f:
            f.write(run.to_json_line() + "\n")

    def read_all(self) -> Iterator[EvaluationRun]:
        if not self.path.exists():
            return
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield EvaluationRun.from_json_line(line)

    def read_by_condition(self, condition: str) -> List[EvaluationRun]:
        return [r for r in self.read_all() if r.condition == condition]

    def summarize(self, condition: str, metric: str = "coverage") -> Dict[str, float]:
      
        values = [getattr(r, metric) for r in self.read_by_condition(condition)]
        if not values:
            return {"n": 0, "mean": float("nan"), "stdev": float("nan")}
        return {
            "n": len(values),
            "mean": statistics.mean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        }

    @staticmethod
    def find_all_experiments(root: Path = DEFAULT_ROOT) -> List[str]:
        if not root.exists():
            return []
        return sorted(p.stem for p in root.glob("*.jsonl"))
