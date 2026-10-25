# ARES: Autonomous Reinforcement Learning Exploration System

Multi-agent cooperative exploration based on multi-agent and hierarchical
reinforcement learning, simulated in Webots and delivered through a monitored,
containerized inference stack. This repository is the code implementation of
the capstone research proposal (see `docs/research/`): a Dec-POMDP formalization
of cooperative exploration, a communication-based reward function, and an
end-to-end pipeline from Webots simulation to a deployed FastAPI policy.

If you are not one of the original team members and want to get the project
running, this file is addressed to you. The fastest way to see the robots is the
Docker option in the next section.

## What this is

Several simulated robots are dropped into an unknown environment and must
explore it together: agreeing on which route to take, deciding when to
communicate, and continuing to operate when a robot fails or the radio link
degrades. A learned policy is exported to ONNX and served through a
schema-validated API, so it can be queried during both training and deployment.

## Quick start: see the simulation with Docker (no installation except Docker)

The image `abrahakk/ares-simulation:latest` on Docker Hub includes Webots R2025a,
the ARES code, and all Python dependencies (CPU-only PyTorch).

**Requirements:** Docker Desktop (Windows/Mac) or Docker Engine (Linux), about
5 GB of free disk space, and an internet connection for the first download.

**1. Start the simulation with browser streaming** (works on any OS):

```bash
docker run --rm -it -p 1234:1234 -e USER=root -e LIBGL_ALWAYS_SOFTWARE=1 --entrypoint xvfb-run abrahakk/ares-simulation:latest -a webots --batch --mode=realtime --stream --stdout --stderr /app/simulation/webots/worlds/simple.wbt
```

The first run takes a few minutes while the image downloads.

**2. View the robots.** When the `INFO: robot_controller: Starting controller`
lines appear, open https://cyberbotics.github.io/webots-streaming-viewer/ in a
browser and connect to `ws://localhost:1234`. Press Ctrl+C in the terminal to stop.

**Alternative (Windows 11 with WSL2 only): open the Webots window directly.**
Run this from the WSL terminal, then press Play in the window:

```bash
docker run --rm -it -e DISPLAY=$DISPLAY -e WAYLAND_DISPLAY=$WAYLAND_DISPLAY -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR -e LIBGL_ALWAYS_SOFTWARE=1 -e USER=root -v /tmp/.X11-unix:/tmp/.X11-unix -v /mnt/wslg:/mnt/wslg --entrypoint webots abrahakk/ares-simulation:latest /app/simulation/webots/worlds/simple.wbt
```

**Headless run (no display, logs only):**

```bash
docker run --rm -it -e USER=root --entrypoint xvfb-run abrahakk/ares-simulation:latest -a webots --batch --no-rendering --mode=fast --stdout --stderr /app/simulation/webots/worlds/simple.wbt
```

**Build the image yourself:**

```bash
docker build -f deployment/docker/simulation.Dockerfile --build-arg WEBOTS_VERSION=R2025a -t ares-sim .
```

Available worlds are in `simulation/webots/worlds/`: `simple.wbt`, `maze.wbt`,
`dynamic.wbt`, and `disaster_zone.wbt`. Software rendering is slow, so expect a
modest frame rate. Harmless warnings (locale, running as root, sound engine,
slow GPU) can be ignored.

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

## Prerequisites (running without Docker)

- Python 3.11
- [Webots](https://cyberbotics.com/) R2025a (only required for running or
  training against the simulator, not for running the API alone)
- Docker and Docker Compose (only required for the containerized stack)
- Redis (bundled via Docker Compose; install locally otherwise)

## Setup

```bash
git clone <this-repository>
cd ARES

python3 -m venv .venv
source .ARES_venv/bin/activate    

pip install -r requirements.txt
```

To open a world locally with Webots installed:

```bash
webots simulation/webots/worlds/simple.wbt
```

## Running the simulation and training a policy

1. Confirm Webots is installed and `WEBOTS_HOME` is set.
2. Launch the single-robot baseline (Experiment 1 in the proposal):

   ```bash
   python3 scripts/train.py --config config/training.yaml --scenario exploration --algorithm single_agent_ppo
   ```

   Training loads hyperparameters from `config/training.yaml` and reward
   weights from `config/rewards.yaml`. Checkpoints are saved to
   `models/checkpoints/latest/`, and the best model is saved to
   `models/checkpoints/best/`, with metadata in
   `models/metadata/model_registry.yaml`.
3. For the cooperative multi-agent runs (Experiments 3-4):

   ```bash
   python3 scripts/train.py --config config/training.yaml --scenario exploration --algorithm mappo --num-robots 2
   ```

**Expected result:** episode logs include episode reward and coverage fraction
for each episode. A successful baseline run reaches around 80-90% coverage of
the `static` environment by the step limit defined in `config/training.yaml`
(`single_agent_ppo.total_env_steps`). If coverage stays close to 0 after
thousands of episodes, check the `logs/training/` folder first before
concluding the algorithm is at fault: a fixed reward points to a problem with
`world_bounds_m` or the reward weight configuration.

## Exporting a trained policy

```bash
python3 scripts/export_onnx.py --checkpoint models/checkpoints/best/mappo_best.pt --output models/onnx/ares_policy.onnx
```

This validates the export by comparing PyTorch and ONNX Runtime outputs on a
fixed test batch before writing the final file. See
`serving/inference/onnx_engine.py` for the validation logic.

## Running the inference service and monitoring stack

```bash
docker compose -f deployment/compose/docker-compose.yml up --build
```

This starts Redis, the FastAPI inference API, Prometheus, and Grafana. Once it
is up:

- API: `http://localhost:8000/health` should return `{"status": "ok"}`
- API docs: `http://localhost:8000/docs`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (default credentials are set via
  `GF_SECURITY_ADMIN_PASSWORD` in `deployment/compose/docker-compose.yml`;
  change this before any non-local use)

## Running evaluation

```bash
python3 scripts/evaluate.py --experiment cooperative_marl --seeds 0 1 2 3 4
python3 -m evaluation.compare_baselines --experiment-a single_robot --experiment-b cooperative_marl --metric coverage
```

The comparison script applies the Mann-Whitney U-test described in the
proposal's evaluation methodology (Section 12.5) to all recorded seeds in
`data/evaluation/`, not only the mean, so any claimed improvement rests on a
test result rather than a visual impression.

## Running tests

```bash
pytest tests/unit tests/integration
pytest tests/simulation   # requires Webots
```

## Data and model provenance

- Raw simulation artifacts are ingested via
  `data.raw_data_manager.RawDataManager`, which requires a description of the
  source and its license/terms. See `data/raw/README.md`.
- All seeds of all evaluation runs are retained in `data/evaluation/`, not just
  summary statistics. See `data/evaluation/README.md`.
- The pre-trained policy included in any release is accompanied by a model card
  describing what it was and was not evaluated on (proposal Section 15.4).

## Safety and scope

This project exists only in simulation. It is not connected to physical
hardware, and nothing in this repository should be taken as evidence of
readiness for real-world or safety-critical deployment.
`simulation/scenarios/surveillance.py` carries a scoping disclaimer: it is a
research tool for studying patrol behavior, not a surveillance system being
evaluated for real-world use.

## License

See `LICENSE`.
