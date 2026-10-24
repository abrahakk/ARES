# notebooks/

Exploratory analysis and one-off investigation - not a place for code
anything else in the repository depends on.

## What belongs here

- Exploratory code running on `data/processed/episode_features.jsonl` or 
  `data/evaluation/*.jsonl` while determining the structure for a
  new metric or visualization, prior to implementing it as a formal
  function in `evaluation/metrics.py`.
- Ad-hoc debugging: plotting a particular trajectory from 
  `data/trajectories/` to diagnose a failure mode before documenting
  it in `docs/experiments/`.
- Rough drafts of an idea which hasn't been developed enough to be
  put into `rl/`, `evaluation/`, or anywhere else.

## What does not belong here

- Anything that can be imported by any other component of the codebase.
  When a cell in the notebook ends up being useful several times,
  turn it into a function within the relevant package
  (`evaluation/`, `rl/`, etc.), test it and use in the notebook.
- Final figures/numbers for `docs/experiments/` write-ups – generate
  these using `evaluation/` functions executed in a script, which will
  make them reproducible through the command line as explained in
  the root `README.md`, not only in a notebook that needs to be
  opened and executed cell by cell.

## Housekeeping

Show clear outputs of cells before committing where feasible - a notebook
containing megabytes of plot images embedded inside is going to make git diffs a nightmare.
The directories `.ipynb_checkpoints/` are ignored via
`.gitignore` anyway.
