# data/processed/

Deterministically produced data – all data in this folder can be recreated
using data from `data/raw/` and/or `data/trajectories/`
by running `ProcessedDataBuilder`. Nothing in this folder is an alternative
source of truth; if anything here contradicts what the builder produces,
the builder’s output takes precedence, and this folder needs to be recreated.

## Contents

- `episode_features.jsonl` - one summary row per recorded episode
  (coverage, collisions, path length, reward, outcome), built by
  `ProcessedDataBuilder.build_episode_feature_table()`. This is the
  table `evaluation/metrics.py` and any exploratory notebook should
  load rather than re-parsing raw trajectories.
- `curriculum_manifest.jsonl` - a lightweight per-environment summary
  (approximate obstacle count, file size) built by
  `ProcessedDataBuilder.build_curriculum_manifest()`, used as a sanity
  check before a training run starts (proposal Section 9).

## Regenerating

```python
from data.raw_data_manager import RawDataManager
from data.processed_data_builder import ProcessedDataBuilder

builder = ProcessedDataBuilder()
builder.build_episode_feature_table()
builder.build_curriculum_manifest(RawDataManager())
```
