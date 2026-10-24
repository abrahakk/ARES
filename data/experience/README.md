# data/experience/

Persistent storage of RL transitions, authored by `ExperienceWriter`.
This is distinct from the in-memory `RolloutBuffer` / `ReplayBuffer`
objects in `rl/utils/replay_buffer.py`, which are process-local for
efficiency purposes during training; this step is chosen at the
discretion of a training script when the experience from the run is
worth preserving beyond the current process (e.g., in order to
retrain on just the alternate reward weighting in the ablation test
proposed in proposal Section 10.3 without re-running the simulation).

## Format

One file per `run_name`, JSON Lines format, each line contains one
`ExperienceRecord`: `observation`, `action`, `reward`,
`next_observation`, `done`, indexed by `robot_id` and `step`.

## Size caveat

The raw experience data will be voluminous. Only write to this path
when you want to preserve specific runs that the team wishes to use
later for replay.
