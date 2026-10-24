from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from data.schemas import TrajectoryRecord

DEFAULT_ROOT = Path("data/trajectories")


class TrajectoryWriter:
    def __init__(self, experiment_name: str, condition: str, root: Path = DEFAULT_ROOT) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / f"{experiment_name}__{condition}.jsonl"

    def write(self, record: TrajectoryRecord) -> None:
        with open(self.path, "a") as f:
            f.write(record.to_json_line() + "\n")

    def read_all(self) -> Iterator[TrajectoryRecord]:
        if not self.path.exists():
            return
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield TrajectoryRecord.from_json_line(line)

    def read_by_outcome(self, outcome: str) -> Iterator[TrajectoryRecord]:
        
        for record in self.read_all():
            if record.outcome == outcome:
                yield record

    @staticmethod
    def find_all_files(root: Path = DEFAULT_ROOT) -> list:
        if not root.exists():
            return []
        return sorted(root.glob("*.jsonl"))

    @classmethod
    def from_file(cls, path: Path) -> "TrajectoryWriter":
        
        name = path.stem  
        experiment_name, _, condition = name.partition("__")
        writer = cls(experiment_name, condition, root=path.parent)
        return writer
