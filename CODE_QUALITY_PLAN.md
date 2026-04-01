# Plan: Code Quality Improvements — Simplification & Professionalization

## Context
After restructuring PtoP into a proper `src/` layout and translating Chinese comments, the codebase still has significant code quality issues: duplicated helper functions across 7+ files, duplicate imports, a bug in `max_population_distance`, dead code, and print statements instead of logging. This plan addresses the most impactful issues.

## 1. Create Shared Geometry Module (`src/ptop/utils/geometry.py`)

**Problem:** `_yaw_to_unit`, `_ego_local_sd`, `ego_local_sd`, `_to_local`, `_vel_local`, `_wrap_yaw_deg`, `_apply_local_offset`, `_vec_norm`, `_unit_vec`, `_dot` are duplicated across 7+ files (some appear 3-5 times).

**Action:** Create `src/ptop/utils/geometry.py` with single canonical implementations, then replace all duplicates with imports.

Functions to centralize:
- `yaw_to_unit(yaw_deg) -> (cos, sin)` — from svgd_runtime.py, surrogate_mlp.py, utility.py, ptop.py, baselines
- `ego_local_sd(ego_tf, pt) -> (s, d)` — from ptop.py, utility.py, world.py, svgd_runtime.py, surrogate_mlp.py, baselines
- `to_local(ego_tf, loc) -> (lx, ly)` — from rl_selector.py (note: different rotation convention from ego_local_sd)
- `vel_local(ego_tf, vel) -> (vx, vy)` — from rl_selector.py
- `wrap_yaw_deg(a) -> float` — from svgd_runtime.py
- `apply_local_offset(ego_tf, ds, dd, dyaw_deg) -> Transform` — from svgd_runtime.py
- `decompose_to_local(ego_tf, npc_tf) -> (ds, dd, dyaw)` — from svgd_runtime.py
- `vec_norm(v) -> float` — from world.py
- `unit_vec(a, b) -> Vector3D` — from world.py
- `dot(a, b) -> float` — from world.py
- `spd_and_vec(actor) -> (speed, velocity)` — from world.py

**Files to update (remove local definitions, add imports):**
- `src/ptop/core/ptop.py` — remove `_yaw_to_unit`, `ego_local_sd`
- `src/ptop/core/world.py` — remove `_ego_local_sd`, `_vec_norm`, `_spd_and_vec`, `_unit_vec`, `_dot` (both module-level and static method duplicates), `_assign_blame_ego` module-level duplicate
- `src/ptop/optimization/svgd_runtime.py` — remove `_yaw_to_unit`, `_ego_local_sd`, `_wrap_yaw_deg`, `_apply_local_offset`, `_decompose_to_local`
- `src/ptop/optimization/surrogate_mlp.py` — remove `_yaw_to_unit`, `_ego_local_sd`, `_relative_yaw_deg`
- `src/ptop/agents/rl_selector.py` — remove `_to_local`, `_vel_local` (both module-level AND nested redefinitions inside `analyze_scene()`)
- `src/ptop/utils/utility.py` — remove `_yaw_to_unit`, `ego_local_sd`
- `src/ptop/baselines/baseline_garl.py` — remove `_yaw_to_unit`, `ego_local_sd`
- `src/ptop/baselines/baseline_mosat.py` — remove `_yaw_to_unit`, `ego_local_sd`
- `src/ptop/baselines/baseline_kings.py` — remove `_yaw_to_unit`, `ego_local_sd`

## 2. Fix Bug in `max_population_distance`

**File:** `src/ptop/utils/utility.py:456`
**Bug:** Returns `min(distances)` instead of `max(distances)`
**Fix:** Change `return min(distances)` to `return max(distances)`

## 3. Clean Up Duplicate Imports

**File:** `src/ptop/utils/utility.py` (lines 1-14)
Consolidate the messy import block:
```python
# Before (7 import lines with duplicates):
import numpy as np
import xml.etree.ElementTree as ET
import math
import carla
from typing import List, Set
from typing import List, Callable, Any, Sequence
import time
import requests, time, carla
import json, urllib.request
import time
import carla
from typing import Tuple, Optional, Iterable, Set

# After (clean, deduplicated):
import json
import math
import time
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Callable, Iterable, List, Optional, Sequence, Set, Tuple

import carla
import numpy as np
import requests
```

**File:** `src/ptop/core/ptop.py` — remove duplicate `import torch` (line 23)

## 4. Extract Shared `_sanitize_position_info`

**Problem:** Identical function in `optimization/seed_generator.py` and `optimization/offline_searcher.py`.
**Action:** Move to `src/ptop/utils/utility.py`, import from both files.

## 5. Remove Dead Code

- `src/ptop/core/world.py` lines 866-907 — large commented-out collision classification block
- `src/ptop/core/world.py` line 1130 — commented-out sensor callback override
- `src/ptop/core/ptop.py` — remove unused `from datetime import datetime`
- `src/ptop/agents/rl_selector.py` — remove nested redefinitions of `_to_local` and `_vel_local` inside `analyze_scene()` (lines 349-368), use module-level versions instead

## 6. Replace Print Statements with Logging

**Action:** Add `import logging` and `logger = logging.getLogger(__name__)` to each file, then:
- `print("[ERROR] ...")` → `logger.error(...)`
- `print("[WARN] ...")` → `logger.warning(...)`
- `print("[INFO] ...")` → `logger.info(...)`
- `print(f"[DEBUG] ...")` → `logger.debug(...)`

**Files (priority order):**
- `src/ptop/core/ptop.py` (~30 print statements)
- `src/ptop/core/world.py` (~40 print statements, already has `logging` imported but unused)
- `src/ptop/core/carla_controller.py` (2 print statements)
- `src/ptop/baselines/baseline_garl.py`
- `src/ptop/baselines/baseline_mosat.py`
- `src/ptop/baselines/baseline_kings.py`

## 7. Remove Duplicate Static Methods in `world.py`

**Problem:** `MultiVehicleDemo` class has static methods (`_vec_norm`, `_spd_and_vec`, `_unit_vec`, `_dot`, `_ego_local_sd`, `_assign_blame_ego`) that duplicate module-level functions.

**Action:** After step 1, remove the module-level duplicates. The class methods should import from `ptop.utils.geometry` instead. Or better: remove the static methods and have the class use the module-level imports directly.

## Implementation Order

1. Create `src/ptop/utils/geometry.py` with all canonical helpers
2. Fix `max_population_distance` bug
3. Clean up duplicate imports in utility.py and ptop.py  
4. Extract `_sanitize_position_info` to utility.py
5. Update all files to import from geometry.py (remove local duplicates)
6. Remove dead code blocks
7. Replace print → logging across all files
8. Update `src/ptop/utils/__init__.py` to re-export geometry

## Verification

- `grep -rn "def _yaw_to_unit\|def ego_local_sd\|def _ego_local_sd\|def _to_local\|def _vel_local\|def _vec_norm\|def _unit_vec\|def _dot\b" src/ptop/` — should only show definitions in `geometry.py`
- `grep -rn "def _sanitize_position_info" src/ptop/` — should only show one definition in `utility.py`
- `grep -rn "^import torch$" src/ptop/core/ptop.py` — should appear exactly once
- `grep -c "print(" src/ptop/core/ptop.py` — should be 0
- `python -c "from ptop.utils.geometry import ego_local_sd, yaw_to_unit"` — import smoke test
