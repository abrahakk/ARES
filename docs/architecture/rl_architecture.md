# RL Architecture

This document covers the `rl/` package specifically: the formal
problem setup, why each algorithm was chosen, and how the pieces
connect. Corresponds to Sections 7 and 10 of the capstone proposal.

## Problem formulation

Exploration is modeled as a decentralized partially observable Markov
decision process (Dec-POMDP). Each robot receives only a local
observation - its own pose, a locally sensed occupancy grid, and any
messages received from teammates - never the full joint state. This
is implemented directly in `rl/environments/observation_space.py`:
`flatten_observation()` is the single function that turns a robot's
structured observation into the fixed-length vector every network in
`rl/networks/` consumes, and it is the same function used during
training and at serving time.

## Reward

`rl/environments/reward.py` implements the six-term reward from
proposal Section 7.3:

```
R_t = w1*R_coverage + w2*R_information + w3*R_cooperation
      - w4*R_collision - w5*R_energy - w6*R_communication
```

Two design choices matter here beyond the formula itself. First,
every component is returned individually, not just the weighted sum -
this is what makes the reward-ablation protocol
(`RewardConfig.ablated()`) and the component-level logging in
`data/schemas.py`'s `EpisodeStep.reward_breakdown` possible. Second,
`R_information` is computed from `rl.coordination.cooperative_exploration.CoverageMap`,
which tracks *team-wide* novelty separately from *per-robot* novelty -
this is the specific mechanism that rewards genuinely cooperative
behavior (reducing the team's collective uncertainty) differently from
a robot simply re-observing space it personally hasn't seen before.

## Networks

`rl/networks/actor.py`'s `ActorNetwork` is the only network ever
exported to ONNX. It has two forward paths: `act()` /
`evaluate_actions()` for training (which use `torch.distributions`
objects PPO's ratio computation needs), and `export_forward()`, a
separate trace-friendly deterministic path used only by
`scripts/export_onnx.py`, because ONNX export does not support
`torch.distributions` objects.

`rl/networks/centralized_critic.py` and `rl/networks/attention.py`
implement centralized-training/decentralized-execution (CTDE): the
critic sees every robot's joint observation and action during
training, aggregated through permutation-invariant attention pooling
so the same critic architecture works whether an experiment runs 2
robots or 12 - relevant to the scalability question in RQ6.

## Algorithms

Three tiers, matching the experiment progression in proposal Section
11:

- **Single-agent** (`rl/agents/single_agent/`): PPO is the primary
  Experiment 1 baseline, chosen for training stability under a short,
  fixed compute budget. DQN and SAC are included as secondary
  reference points, useful for sanity-checking the environment and
  reward pipeline independently of PPO's specific behavior.
- **Multi-agent** (`rl/agents/multi_agent/`): MAPPO is primary, for
  the same stability reasons as single-agent PPO. MADDPG and QMIX are
  included as cross-checks, so a reported cooperative-benefit result
  (RQ1) is not an artifact of one specific algorithm's quirks.
- **Hierarchical** (`rl/agents/hierarchical/`): a goal-conditioned
  high-level/low-level split following the HIRO pattern. The
  high-level policy selects from a small, fixed-size candidate set
  (known frontiers plus two special goals - regroup, hold) rather than
  a raw coordinate, which keeps its action space small and
  interpretable for the training-stability comparison RQ2 asks for.

## Communication

`rl/communication/` sits between the wire format
(`robots.communication.message.Message`) and the policy. A robot's
variable-length inbox is encoded (`message_encoder.py`) and pooled
via attention (`communication_policy.py`) into the fixed-size
`message_summary` slice of the observation vector - handling zero,
one, or many received messages uniformly, without the observation
vector's shape changing based on how many teammates happened to
transmit that step.

## What's deliberately not here

`rl/` contains no Webots imports and no Redis imports. If a future
change needs either inside this package, that is a sign the change
belongs in `simulation/` or `coordination/` instead - see
`system_architecture.md` for the dependency-direction rule this
package is built to respect.
