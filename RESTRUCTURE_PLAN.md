# Plan: Restructure PtoP into Industrial-Level Project Layout

## Context
The PtoP codebase was a flat directory of 17 Python files with no packaging, inconsistent naming (e.g., `DQN.py`, `ART_SEED_GENERATOR.py`), and a broken import (`from dqn_agent import DQNAgent` but the file was `DQN.py`). The goal was to reorganize into a proper `src/` layout with logical subdirectories, `__init__.py` files, and a `pyproject.toml`.

## Executed: Target Structure

```
PtoP/
├── pyproject.toml
├── README.md
├── CLAUDE.md
├── listener.py                          # stays at root (runs in Apollo docker)
├── FSE_2026_Supplementary.pdf
├── RAW Result.xlsx
├── demographic information.csv
└── src/
    └── ptop/
        ├── __init__.py
        ├── __main__.py                  # python -m ptop
        ├── core/
        │   ├── __init__.py
        │   ├── ptop.py                  # was PtoP.py
        │   ├── world.py
        │   └── carla_controller.py
        ├── optimization/
        │   ├── __init__.py
        │   ├── seed_generator.py        # was ART_SEED_GENERATOR.py
        │   ├── svgd_runtime.py          # was npc_svgd_runtime.py
        │   ├── surrogate_mlp.py         # was npc_surrogate_mlp.py
        │   └── offline_searcher.py
        ├── agents/
        │   ├── __init__.py
        │   ├── dqn_agent.py             # was DQN.py (fixes broken import)
        │   ├── rl_selector.py           # was RL.py
        │   └── replay_buffer.py
        ├── utils/
        │   ├── __init__.py
        │   ├── utility.py
        │   ├── feature.py
        │   └── math_tool.py
        ├── analysis/
        │   ├── __init__.py
        │   └── compute_diversity.py
        └── baselines/
            ├── __init__.py
            ├── baseline_garl.py         # was Baselines/Baseline_GARL.py
            ├── baseline_mosat.py        # was Baselines/Baseline_MOSAT.py
            └── baseline_kings.py        # was Baselines/Baseline_KINGS.py
```

## Executed: File Moves (17 files)

| Original | New Location | Renamed? |
|----------|-------------|----------|
| `PtoP.py` | `src/ptop/core/ptop.py` | yes |
| `world.py` | `src/ptop/core/world.py` | no |
| `carla_controller.py` | `src/ptop/core/carla_controller.py` | no |
| `ART_SEED_GENERATOR.py` | `src/ptop/optimization/seed_generator.py` | yes |
| `npc_svgd_runtime.py` | `src/ptop/optimization/svgd_runtime.py` | yes |
| `npc_surrogate_mlp.py` | `src/ptop/optimization/surrogate_mlp.py` | yes |
| `offline_searcher.py` | `src/ptop/optimization/offline_searcher.py` | no |
| `DQN.py` | `src/ptop/agents/dqn_agent.py` | yes |
| `RL.py` | `src/ptop/agents/rl_selector.py` | yes |
| `replay_buffer.py` | `src/ptop/agents/replay_buffer.py` | no |
| `utility.py` | `src/ptop/utils/utility.py` | no |
| `feature.py` | `src/ptop/utils/feature.py` | no |
| `math_tool.py` | `src/ptop/utils/math_tool.py` | no |
| `compute_diversity.py` | `src/ptop/analysis/compute_diversity.py` | no |
| `Baselines/Baseline_GARL.py` | `src/ptop/baselines/baseline_garl.py` | yes |
| `Baselines/Baseline_MOSAT.py` | `src/ptop/baselines/baseline_mosat.py` | yes |
| `Baselines/Baseline_KINGS.py` | `src/ptop/baselines/baseline_kings.py` | yes |
| `listener.py` | `listener.py` | no move |

## Executed: Import Changes (7 files modified)

### `src/ptop/core/ptop.py` (was PtoP.py)
```
from utility import ...          -> from ptop.utils.utility import ...
from npc_surrogate_mlp import ...-> from ptop.optimization.surrogate_mlp import ...
from world import ...            -> from ptop.core.world import ...
from ART_SEED_GENERATOR import...-> from ptop.optimization.seed_generator import ...
from npc_svgd_runtime import ... -> from ptop.optimization.svgd_runtime import ...
```

### `src/ptop/core/world.py`
```
from carla_controller import ... -> from ptop.core.carla_controller import ...
```

### `src/ptop/optimization/seed_generator.py` (was ART_SEED_GENERATOR.py)
```
from utility import ...          -> from ptop.utils.utility import ...
```

### `src/ptop/optimization/offline_searcher.py`
```
from utility import ...          -> from ptop.utils.utility import ...
```

### `src/ptop/baselines/baseline_garl.py`
```
from utility import ...          -> from ptop.utils.utility import ...
from world import ...            -> from ptop.core.world import ...
from offline_searcher import ... -> from ptop.optimization.offline_searcher import ...
from npc_surrogate_mlp import ...-> from ptop.optimization.surrogate_mlp import ...
from npc_svgd_runtime import ... -> from ptop.optimization.svgd_runtime import ...
from replay_buffer import ...    -> from ptop.agents.replay_buffer import ...
from dqn_agent import ...        -> from ptop.agents.dqn_agent import ...
```

### `src/ptop/baselines/baseline_mosat.py`
Same pattern as baseline_garl minus the dqn_agent import.

### `src/ptop/baselines/baseline_kings.py`
```
from utility import ... (inside try/except) -> from ptop.utils.utility import ...
from world import ...                       -> from ptop.core.world import ...
```

### 10 files needed NO import changes
carla_controller.py, surrogate_mlp.py, svgd_runtime.py, dqn_agent.py, rl_selector.py, replay_buffer.py, utility.py, feature.py, math_tool.py, compute_diversity.py

## Executed: New Files Created (10 files)

1. **`pyproject.toml`** — hatchling build system, dependencies (numpy, torch, scipy, opencv-python, requests, websocket-client), console scripts `ptop` and `ptop-diversity`
2. **`src/ptop/__init__.py`** — package root with version string
3. **`src/ptop/__main__.py`** — delegates to `ptop.core.ptop:main` for `python -m ptop`
4. **7 x `__init__.py`** — one per subpackage (core, optimization, agents, utils, analysis, baselines)

## Bug Fix Included
`DQN.py` was imported as `from dqn_agent import DQNAgent` in `Baseline_GARL.py` — this was broken since no `dqn_agent.py` existed. Renaming `DQN.py` to `dqn_agent.py` during the move resolved it.

## Implementation Order (as executed)

1. Created directory structure: `src/ptop/{core,optimization,agents,utils,analysis,baselines}/`
2. Moved all 17 files with `git mv` (preserves git history)
3. Created all `__init__.py` files
4. Updated imports in the 7 files listed above
5. Both `ptop.py` and `compute_diversity.py` already had `def main()` functions — no wrapping needed
6. Created `pyproject.toml` and `src/ptop/__main__.py`
7. Removed empty `Baselines/` directory
8. Updated `CLAUDE.md` with new paths and run commands

## Verification Steps
```bash
pip install -e .
python -c "from ptop.core.world import MultiVehicleDemo"
python -c "from ptop.optimization.surrogate_mlp import NPCHazardMLPSurrogate"
python -c "from ptop.agents.dqn_agent import DQNAgent"
python -m ptop.analysis.compute_diversity --help
git log --follow src/ptop/core/ptop.py
```
