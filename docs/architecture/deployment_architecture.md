# Deployment Architecture

This document covers the path from a trained policy to a running,
monitored service: `serving/`, `deployment/`, and `monitoring/`.
Corresponds to Sections 13 and 14 of the capstone proposal.

## Export

`scripts/export_onnx.py` converts a trained `rl.networks.actor.ActorNetwork`
checkpoint into an ONNX file via its `export_forward()` path (see
`rl_architecture.md` for why a separate export path exists). Before
writing the final file, the script runs a fixed batch of test
observations through both the original PyTorch model and the
exported ONNX model and confirms the outputs match within a small
numerical tolerance - an export that silently changes model behavior
is a worse failure mode than one that crashes outright, so this check
is not optional.

## Serving

`serving/app/main.py` loads the ONNX model exactly once, at process
startup, via a FastAPI `lifespan` context manager - never inside a
request handler, which would reload the model from disk on every
call. Every incoming observation is validated against a Pydantic
schema (`serving/app/schemas.py`) before it reaches the model, so
malformed input fails with a clear `422` response rather than an
opaque model or ONNX Runtime error.

The service is stateless: nothing it needs to make a prediction lives
in the process itself. Robot state, task claims, and message routing
all live in Redis via `coordination/`. This is what makes horizontal
scaling in `deployment/kubernetes/api-deployment.yaml` safe without
any session-affinity configuration - any replica can answer any
request.

## Coordination

`coordination/redis_client.py` is the single connection point every
other coordination module depends on, constructed from
`config/system.yaml` rather than hard-coded host/port values, so the
same code runs against a local Redis instance and against the
`ares-redis` Docker Compose / Kubernetes service name without changes.
`state_manager.py`, `event_bus.py`, and `task_manager.py` each wrap
one Redis pattern (TTL'd key-value, pub/sub, atomic `SET NX`) rather
than one class trying to be a general Redis abstraction.

## Containerization

Three purpose-built images, each scoped to one service's actual
dependencies rather than one image trying to serve every purpose:

| Image | Contains | Does not contain |
|---|---|---|
| `deployment/docker/api.Dockerfile` | `serving/`, `rl/networks`, `rl/environments`, `robots/`, `coordination/`, `config/`, the ONNX model | Webots, full PyTorch training stack, `simulation/` |
| `deployment/docker/simulation.Dockerfile` | Webots (headless), `simulation/`, `robots/`, `rl/` | training optimizer loop |
| `deployment/docker/training.Dockerfile` | full PyTorch/CUDA stack, `rl/`, `training/`, `evaluation/` | Webots binary |

The root `Dockerfile` is a fourth, separate thing: a general
interactive dev container with the whole repository, used for local
iteration - not deployed anywhere. See the comment at the top of that
file for the full reasoning.

## Orchestration

`docker-compose.yml` (root) is the fast local-iteration stack - Redis
plus a bind-mounted dev container. `deployment/compose/docker-compose.yml`
is closer to the actual demonstration environment - Redis, the
scoped API image, Prometheus, and Grafana, all built from the
production Dockerfiles rather than the dev one.
`deployment/kubernetes/` is the documented future-scale path
(proposal Section 20): not required at this capstone's scale, but
present so a larger deployment has a starting point.

## Monitoring

The FastAPI service exposes a `/metrics` endpoint via
`prometheus-fastapi-instrumentator`, giving request latency,
throughput, and error rate for free. Reinforcement-learning and
robotics metrics (coverage, collision rate, mission completion) are a
separate stream, written through `data/evaluation_writer.py` and
read into Grafana dashboards alongside the infrastructure metrics.
The reason both streams matter side by side: a drop in mission
completion rate that coincides with a latency spike points at
infrastructure; the same drop with normal latency points at the
policy. Neither view alone can tell those two failure modes apart.

## Security notes specific to this layer

The inference API is the one component in this repository that
accepts external input by design, which makes it the primary attack
surface referenced in proposal Section 15.1. The concrete mitigations
implemented here: strict Pydantic schema validation on every request,
no public exposure for Redis (internal Docker/Kubernetes network
only), and no credentials committed anywhere in this repository -
`deployment/kubernetes/monitoring.yaml`'s Grafana admin password is a
placeholder (`changeme`) specifically so it is obvious it must be
replaced via a real secrets mechanism before any non-local use.
