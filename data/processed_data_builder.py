from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

from data.raw_data_manager import RawDataManager
from data.trajectory_writer import TrajectoryWriter

DEFAULT_ROOT = Path("data/processed")


class ProcessedDataBuilder:
    def __init__(self, root: Path = DEFAULT_ROOT) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

   
    def build_episode_feature_table(
        self, trajectory_root: Path = Path("data/trajectories")
    ) -> Path:
   
        output_path = self.root / "episode_features.jsonl"
        rows: List[Dict] = []

        for file_path in TrajectoryWriter.find_all_files(root=trajectory_root):
            reader = TrajectoryWriter.from_file(file_path)
            for record in reader.read_all():
                rows.append(self._summarize_episode(record))

        with open(output_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        return output_path

    @staticmethod
    def _summarize_episode(record) -> Dict:
        steps = record.steps
        num_steps = len(steps)
        total_collisions = sum(1 for s in steps if s.collided)
        total_messages_sent = sum(s.messages_sent for s in steps)
        final_coverage = steps[-1].coverage_fraction if steps else 0.0
        total_reward = sum(s.reward for s in steps)

        path_length = 0.0
        for prev, curr in zip(steps, steps[1:]):
            dx = curr.pose[0] - prev.pose[0]
            dy = curr.pose[1] - prev.pose[1]
            path_length += (dx ** 2 + dy ** 2) ** 0.5

        return {
            "episode_id": record.episode_id,
            "experiment_name": record.experiment_name,
            "condition": record.condition,
            "seed": record.seed,
            "environment_name": record.environment_name,
            "outcome": record.outcome,
            "num_steps": num_steps,
            "final_coverage_fraction": final_coverage,
            "total_collisions": total_collisions,
            "total_messages_sent": total_messages_sent,
            "total_reward": total_reward,
            "path_length_m": path_length,
        }

     def build_curriculum_manifest(self, raw_manager: RawDataManager) -> Path:
        output_path = self.root / "curriculum_manifest.jsonl"
        manifest_rows: List[Dict] = []

        for world_path in raw_manager.list_files(subdirectory="webots_worlds"):
            if world_path.suffix != ".wbt":
                continue
            text = world_path.read_text(errors="ignore")
            obstacle_count = len(re.findall(r"\bSolid\s*{", text)) + len(re.findall(r"\bObstacle\s*{", text))
            manifest_rows.append(
                {
                    "environment_name": world_path.stem,
                    "file_size_bytes": world_path.stat().st_size,
                    "approx_obstacle_count": obstacle_count,
                }
            )

        with open(output_path, "w") as f:
            for row in manifest_rows:
                f.write(json.dumps(row) + "\n")

        return output_path
