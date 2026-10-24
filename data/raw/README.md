# data/raw/

Original capture data: Webots world exports, sensor outputs, and any
other input files exactly as they were generated or collected. None of
the contents of this directory are ever altered; if there is a need
to process any data from this directory, it will be done in
`data/processed/` using `ProcessedDataBuilder`.

## Provenance

Each and every file in this directory needs to be ingested using
`RawDataManager.ingest()`. It needs a source description and the license/
terms of use, which will then be logged along with the SHA-256 checksum
and the timestamp in `_provenance.jsonl`. This is what it means to do
the data-rights checkpoint of the capstone. One needs to make sure that one
has the right to use the file and that there are no restrictions on the
use of such data.

One needs to periodically run `RawDataManager.verify_integrity()` to
verify that no file has been modified from its checksum.

## What belongs here

- Webots `.wbt` world files under `webots_worlds/`
- Any exported sensor logs used as fixed test input (e.g. for a
  regression test fixture)

## What does not belong here

- Anything derived or computed (goes in `data/processed/`)
- Training rollouts or replay data (goes in `data/experience/`)
- Recorded episode trajectories (goes in `data/trajectories/`)
