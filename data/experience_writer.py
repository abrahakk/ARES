from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

from data.schemas import ExperienceRecord

DEFAULT_ROOT = Path("data/experience")


class ExperienceWriter:
    def __init__(self, run_name: str, root: Path = DEFAULT_ROOT) -> None:
        self.run_name = run_name
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / f"{run_name}.jsonl"

    def write(self, record: ExperienceRecord) -> None:
        with open(self.path, "a") as f:
            f.write(record.to_json_line() + "\n")

    def write_batch(self, records: Iterable[ExperienceRecord]) -> None:
        with open(self.path, "a") as f:
            for record in records:
                f.write(record.to_json_line() + "\n")

    def read_all(self) -> Iterator[ExperienceRecord]:
        if not self.path.exists():
            return
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield ExperienceRecord.from_json_line(line)

    def count(self) -> int:
        if not self.path.exists():
            return 0
        with open(self.path) as f:
            return sum(1 for _ in f)
