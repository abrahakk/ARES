# System Architecture

This document is the map of how every package in this repository fits
together. It corresponds to Section 6 of the capstone proposal, but
written against the actual code rather than the plan for it - if this
document and the proposal ever disagree, treat this one as current
and the proposal as the original intent.

## The six layers

ARES is organized into six layers, each with a narrow responsibility
and a dependency direction that only ever points one way: simulation
and coordination code depends on `robots/` and `rl/`; `robots/` and
`rl/` never import from `simulation/`, `coordination/`, or `serving/`.
Keeping that boundary one-directional is what lets the same
`robots.agents.robot.Robot` and `rl.networks.actor.ActorNetwork` code
run identically inside a training rollout and behind the deployed API.

```
Webots (simulation/webots/)
        |
robots/  <-- sensors, navigation, communication, per-robot agent
        |
rl/      <-- environments, networks, algorithms (consumes robots/ observations)
        |
coordination/  <-- Redis-backed shared state, message routing, task claims
        |
data/    <-- persistence for trajectories, experience, evaluation results
        |
serving/ (ONNX + FastAPI)  <-- deployment-time consumer of a trained rl/ policy
        |
deployment/, monitoring/   <-- containers, orchestration, observability
```

## Layer responsibilities

**`robots/`** is the only package that knows about Webots devices by
name (`"gps"`, `"lidar"`, `"left wheel motor"`, ...). It exposes
exactly two operations upward - `Robot.get_observation()` and
`Robot.apply_action()` - and a non-learned safety layer
(`ObstacleAvoidance`) that can override an unsafe policy command
regardless of what produced it. Everything else in the repository
reaches Webots only indirectly, through this package.

**`rl/`** never imports Webots. `rl.environments.ares_env.AresEnv`
wraps `robots.agents.robot_manager.RobotManager` to turn a sequence of
`Robot.step()` calls into a PettingZoo-style `reset()`/`step()` loop.
`rl.networks.actor.ActorNetwork` is the one network ever exported to
ONNX; the centralized critic, the high-level policy, and every
training-only network stay behind in this package.

**`simulation/`** holds the Webots-specific artifacts: world files,
robot PROTOs, and the two controller entry points Webots actually
launches (`robot_controller`, `supervisor_controller`). Scenario
configuration (`simulation/scenarios/`) and environment configuration
(`simulation/environments/`) are the glue between a `.wbt` file and
the kwargs `AresEnv` expects.

**`coordination/`** is the Redis-backed layer that exists specifically
because robots and services are separate processes that need a shared
view of the world: robot state (`state_manager.py`), robot-to-robot
messages (`event_bus.py`), and frontier/task claims
(`task_manager.py`), tied together by `mission_manager.py`. This is
distinct from `rl.coordination`, which holds in-process,
training-time-only logic (the joint coverage map, formation math) that
never touches a network.

**`data/`** is the persistence and provenance layer: raw asset
ingestion with checksum-verified provenance, deterministically
regenerable processed features, and append-only JSON-Lines logs for
trajectories, RL experience, and per-seed evaluation results. Every
number in an evaluation report should be traceable back through this
layer to the exact run that produced it.

**`serving/` + `deployment/` + `monitoring/`** are the production
path: an exported ONNX policy served through a schema-validated
FastAPI app, containerized per-service (`deployment/docker/`),
orchestrated via Compose for the demonstration or Kubernetes for a
future larger-scale deployment, and instrumented with Prometheus/
Grafana so an infrastructure failure and a policy failure never look
the same on a dashboard.

## Configuration

All five layers read from `config/*.yaml` rather than hard-coded
constants: `rewards.yaml`, `robots.yaml`, `simulation.yaml`,
`system.yaml`, `training.yaml`. A handful of physical constants
(wheel radius, axle length, sensor names) are necessarily duplicated
between `config/robots.yaml`, `robots/navigation/local_controller.py`,
and `simulation/webots/protos/ARESRobot.proto` - see the comment at
the top of `config/robots.yaml` for the exact list of what must stay
in sync if any one of them changes.

## Where to look next

- For the RL formulation and algorithm choices specifically, see
  `rl_architecture.md` in this directory.
- For the deployment/serving path specifically, see
  `deployment_architecture.md` in this directory.
- For the full research write-up (literature review, research
  questions, experimental design, evaluation methodology), see
  `docs/research/`.
