# Experiment Write-ups

This directory holds one write-up per experiment defined in the
capstone proposal (Section 11), summarizing what `data/evaluation/`
and `data/trajectories/` actually show once an experiment has been
run to completion.

## What belongs here

One markdown file per experiment, e.g. `experiment_1_single_robot_baseline.md`,
`experiment_3_cooperative_marl.md`, `experiment_5_communication_constraints.md`,
following the same naming pattern as `simulation/experiments/` (if
present) and `data/evaluation/<experiment_name>.jsonl`.

Each write-up should cover:

- **Setup** - which scenario, environment, and algorithm configuration
  was used, and a link to the exact `config/training.yaml` /
  `config/rewards.yaml` values in force for the run (or a copy, if
  the config has since changed).
- **Result** - the summary statistics from
  `EvaluationWriter.summarize()`, and the outcome of the relevant
  significance test from `evaluation/compare_baselines.py`, not just
  the mean.
- **Failure cases** - what `data/trajectories/` shows for the worst
  seeds, not only the best ones. This is the same honesty commitment
  described in `data/evaluation/README.md` - an experiment write-up
  that only shows favorable seeds is not a write-up a panel should
  trust.

## What does not belong here

Raw metrics or trajectory data - those live in `data/evaluation/` and
`data/trajectories/` respectively, and should be linked to, not
copied in. This directory is for the human-readable interpretation of
that data, not a second copy of it.
