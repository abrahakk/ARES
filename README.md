# ARES: Autonomous Reinforcement Learning Exploration System

Multi-agent cooperative exploration that was based on
multi-agent and hierarchical reinforcement learning simulations
performed in Webots and delivered through monitored containerized
inference stack. The current repository represents the actual code
implementation of the capstone research proposal (see
`docs/research/` for more information): Dec-POMDP formalization of
cooperative exploration, a communication-based reward function,
and an end-to-end pipeline from Webots simulation to a deployed
FastAPI policy.

In case you are not part of the original team members and looking at
this document to get the project up and running, then this file is
directly addressed to you - all commands below will work in a fresh
clone.
## What this is

A pair of more simulated robots are dropped into an unknown environment,
and they are required to explore the environment together, by agreeing
on which route to take, when to communicate with one another, and how
to continue operating in the event that a robot breaks down or if the
radio connection becomes problematic. A learned policy is exported to
ONNX and provided through a schema validated API, making it available
for queries in both training and deployment settings.

## Repository layout

```
robots/         per-robot agent, sensors, navigation, communication (Webots-facing)
rl/             environments, networks, and single/multi-agent/hierarchical algorithms
simulation/     Webots world files, robot PROTOs, controllers, scenario configs
coordination/   Redis-backed shared state, pub/sub messaging, task claiming
data/           I/O layer for raw/processed/experience/trajectory/evaluation data
config/         YAML configuration for rewards, robots, simulation, system, training
serving/        FastAPI inference service (ONNX Runtime)
deployment/     Dockerfiles, Docker Compose, Kubernetes manifests
monitoring/     Prometheus + Grafana configuration
evaluation/     metrics, baseline comparison, robustness/scalability analysis
training/       training-loop entry points and curriculum logic
scripts/        CLI entry points (train, evaluate, export_onnx, benchmark, ...)
tests/          unit, integration, and simulation tests
docs/           architecture notes and the capstone research write-up
```

## Prerequisites

- Python 3.11
- [Webots](https://cyberbotics.com/) R2023b (only required for running or
  training against the simulator - not required to run the API alone)
- Docker and Docker Compose (only required for the containerized stack)
- Redis (bundled via Docker Compose; install locally otherwise)

## Setup

```bash
git clone <this-repository>
cd ares

python3 -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows

pip install -r requirements.txt
```

## Running the simulation and training a policy

1. Confirm Webots is installed and `WEBOTS_HOME` is set.
2. Launch the single-robot baseline (Experiment 1 in the proposal):

   ```bash
   python3 scripts/train.py --config config/training.yaml --scenario exploration --algorithm single_agent_ppo
   ```

   Training loads hyperparameters from `config/training.yaml`, while the
   reward weights are loaded from `config/rewards.yaml`. Checkpoints are
   saved to `models/checkpoints/latest/`, while the currently best model
   is saved to `models/checkpoints/best/`, using metadata in
   `models/metadata/model_registry.yaml`.
3. For the cooperative multi-agent runs (Experiments 3-4):

   ```bash
   python3 scripts/train.py --config config/training.yaml --scenario exploration --algorithm mappo --num-robots 2
   ```

**Expected result:** episode logs include episode reward and coverage
raction for each episode; successful baseline run will have around
80-90% coverage of the `static` environment up to the step limit
defined in `config/training.yaml` parameter `single_agent_ppo.total_env_steps`.
If the coverage remains close to 0 after thousands of episodes, examine
the `logs/training/` folder first before concluding that there’s an issue
with the algorithm – a fixed reward indicates a problem with `world_bounds_m`
or reward weight configuration, not the algorithm.

## Exporting a trained policy

```bash
python3 scripts/export_onnx.py --checkpoint models/checkpoints/best/mappo_best.pt --output models/onnx/ares_policy.onnx
```

This validates the export by comparing PyTorch and ONNX Runtime
outputs on a fixed test batch before writing the final file - see
`serving/inference/onnx_engine.py` for the validation logic.

## Running the inference service and monitoring stack

```bash
docker compose -f deployment/compose/docker-compose.yml up --build
```

This starts Redis, the FastAPI inference API, Prometheus, and
Grafana. Once it's up:

- API: `http://localhost:8000/health` should return `{"status": "ok"}`
- API docs: `http://localhost:8000/docs`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (default credentials are set via
  `GF_SECURITY_ADMIN_PASSWORD` in `deployment/compose/docker-compose.yml` -
  change this before any non-local use)

## Running evaluation

```bash
python3 scripts/evaluate.py --experiment cooperative_marl --seeds 0 1 2 3 4
python3 -m evaluation.compare_baselines --experiment-a single_robot --experiment-b cooperative_marl --metric coverage
```


The comparison script applies the Mann-Whitney U-test detailed in the 
proposal's evaluation methodology (Section 12.5) to all recorded seeds 
in `data/evaluation/`, not only the mean; hence, any claimed improvement 
is based on a test result, not a visual one.
## Running tests

```bash
pytest tests/unit tests/integration
pytest tests/simulation   # requires Webots
```

## Data and model provenance

Raw simulation artifacts get ingested via `data.raw_data_manager.RawDataManager`,
  but require a description of the source and its license/terms upon ingestion -
  see `data/raw/README.md`.
- All seeds of all evaluation runs are retained in `data/evaluation/`,
  not just summary statistics - see `data/evaluation/README.md`.
- The pre-trained policy included in any release is accompanied by
  a model card describing what it was and wasn't evaluated on (proposal
  Section 15.4).
## Safety and scope

This simulation project only exists in simulation form. This
project is not connected to physical hardware in any way, and nothing from
this project repository should be considered indicative of readiness for
deployment in real-world scenarios or for use in a safety-critical
situation. `simulation/scenarios/surveillance.py`, in fact, comes with a
scoping disclaimer: it is a research tool used to study patrol
behavior, not a surveillance system being evaluated for real-world use.

## License

See `LICENSE`.
