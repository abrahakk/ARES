from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterator, Optional

DEFAULT_ROOT = Path("data/raw")
PROVENANCE_FILENAME = "_provenance.jsonl"


class RawDataManager:
    def __init__(self, root: Path = DEFAULT_ROOT) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.provenance_path = self.root / PROVENANCE_FILENAME

    def ingest(
        self,
        source_path: Path,
        source_description: str,
        license_or_terms: str,
        subdirectory: Optional[str] = None,
    ) -> Path:

        destination_dir = self.root / subdirectory if subdirectory else self.root
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / source_path.name

        shutil.copy2(source_path, destination_path)
        checksum = self._sha256(destination_path)

        entry = {
            "filename": str(destination_path.relative_to(self.root)),
            "source_description": source_description,
            "license_or_terms": license_or_terms,
            "ingested_at": dt.datetime.utcnow().isoformat() + "Z",
            "sha256": checksum,
        }
        with open(self.provenance_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        return destination_path

    def list_files(self, subdirectory: Optional[str] = None) -> Iterator[Path]:
        search_root = self.root / subdirectory if subdirectory else self.root
        if not search_root.exists():
            return
        for path in search_root.rglob("*"):
            if path.is_file() and path.name != PROVENANCE_FILENAME:
                yield path

    def read_provenance(self) -> Iterator[dict]:
        if not self.provenance_path.exists():
            return
        with open(self.provenance_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def verify_integrity(self) -> list:

        mismatches = []
        for entry in self.read_provenance():
            file_path = self.root / entry["filename"]
            if not file_path.exists():
                mismatches.append(entry["filename"])
                continue
            if self._sha256(file_path) != entry["sha256"]:
                mismatches.append(entry["filename"])
        return mismatches

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                digest.update(chunk)
        return digest.hexdigest()
