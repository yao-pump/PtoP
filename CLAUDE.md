# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PtoP** (Particles to Perils) is a research framework for testing autonomous driving systems (ADS) by generating hazardous traffic scenarios in the CARLA simulator. It combines adaptive random seed generation with Stein Variational Gradient Descent (SVGD) to discover diverse failure-inducing initial conditions. The paper targets FSE 2026.

## Setup and Running

Requires CARLA 0.9.13, Apollo 8.0, and the CARLA-Apollo Bridge. The `carla` package is not on PyPI — install from CARLA's provided `.egg`/`.whl`.

```bash
pip install -e .           # editable install (uses pyproject.toml)
```

Run the main framework:
```bash
python -m ptop             # or: ptop (console script)
```

Run baselines:
```bash
python -m ptop.baselines.baseline_garl
python -m ptop.baselines.baseline_mosat
python -m ptop.baselines.baseline_kings
```

Evaluate results:
```bash
python -m ptop.analysis.compute_diversity your_log.jsonl   # or: ptop-diversity your_log.jsonl
```

Full CARLA+Apollo startup sequence:
1. Start CARLA: `./CarlaUE4.sh -RenderOffScreen`
2. Configure map in bridge docker: `python carla-python-0.9.13/util/config.py -m Town04 --host 172.17.0.1`
3. Start manual control: `python examples/manual_control.py`
4. Launch Apollo via CARLA-Apollo Bridge
5. Run cyber bridge: `python carla_cyber_bridge/run_bridge.py`
6. Start listener in Apollo docker: `python listener.py` (kept at repo root — uses Apollo SDK)
7. Run main: `python -m ptop`

## Project Structure

```
src/ptop/
├── core/              # Simulation loop and CARLA integration
│   ├── ptop.py        # Main orchestrator (entry point)
│   ├── world.py       # MultiVehicleDemo: CARLA world, spawning, collision detection
│   └── carla_controller.py  # PID lane-keeping/change controller
├── optimization/      # Scenario generation and refinement
│   ├── seed_generator.py    # ART-based diverse seed selection
│   ├── svgd_runtime.py      # SVGD particle refinement of NPC positions
│   ├── surrogate_mlp.py     # MLP hazard surrogate (online training, EMA target)
│   └── offline_searcher.py  # GA-based population evolution
├── agents/            # NPC behavior policies
│   ├── dqn_agent.py         # DQN with replay buffer
│   ├── rl_selector.py       # ART online selector (risk scoring, KNN memory)
│   └── replay_buffer.py     # Near-miss sample storage
├── utils/             # Shared helpers
│   ├── utility.py           # NPC cleanup, Apollo comms, population distance
│   ├── feature.py           # Geometric transforms (SE2, lane offset, curvature)
│   └── math_tool.py         # Simulated lidar and geometry
├── analysis/          # Post-run evaluation
│   └── compute_diversity.py # Diversity, coverage, violation stats
└── baselines/         # Comparison implementations
    ├── baseline_garl.py     # GA + DQN
    ├── baseline_mosat.py    # GA + MOSAT motif control
    └── baseline_kings.py    # Kinematic bicycle MPC
```

`listener.py` stays at repo root — it runs inside the Apollo docker container with its own dependencies (`cyber.python.cyber_py3`).

## Architecture

### Core Pipeline (`src/ptop/core/ptop.py`)
Each episode:
1. **Seed generation** (`optimization/seed_generator.py`) — ART selects diverse initial positions using KNN distance maximization
2. **SVGD refinement** (`optimization/svgd_runtime.py`) — Refines NPC positions `(ds, dd, dyaw)` with gradient attraction + kernel repulsion
3. **Online NPC control** — Vehicle and walker planners run each tick with gradient-based trajectory optimization
4. **Surrogate model** (`optimization/surrogate_mlp.py`) — MLP trained online predicting hazard; provides gradients to SVGD
5. **GA evolution** (`optimization/offline_searcher.py`) — Population-level crossover/mutation on spawn positions

### Key Coordinate Convention
All NPC positioning uses ego-local coordinates `(s, d, dyaw)`:
- `s` = longitudinal (forward positive)
- `d` = lateral (right positive)
- `dyaw` = relative yaw angle

The `_ego_local_sd` / `_to_local` helper appears in multiple files with the same logic.

## Language

Code comments and docstrings are primarily in Chinese (Mandarin). Variable names and API interfaces use English.
