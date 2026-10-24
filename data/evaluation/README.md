# data/evaluation/

Experimental data per-seed, written by `EvaluationWriter`, to be read
by `evaluation/compare_baselines.py`. This folder contains the data that
the statistical assertions made in the capstone proposal Section 12.5
are actually based on: all seed data is stored individually and not
averaged into one mean, allowing us to test, for instance, our
assertions of improved performance due to cooperation, e.g., in RQ1,
with a Mann-Whitney U test.

## Format

A single file per `experiment_name`, JSON Lines format, one `EvaluationRun`
per line: `condition`, `seed`, `environment_name`, and the entire set of
metrics from Section 12.1-12.4 of the proposal (coverage, mission
completion, collisions, path efficiency, energy consumption,
communications, mean episode reward, and reward component breakdown).

## Honesty requirement

All seeds for each condition should be written here,
including unsuccessful and poor seeds – don’t filter out the good
runs before writing them. `EvaluationWriter.summarize()`
will calculate the average and standard deviation based on what
is really there, so missing a bad seed means inflating the result.
