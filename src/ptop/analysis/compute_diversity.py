#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute:
  1) Diversity ρ(x) over *collision* cases only
  2) Coverage of NPC spawn positions (within 50 m of ego) over CARLA map spawn points
  3) Violation stats across ALL scenarios

Distance (your spec):
  d(x_i, x_j) = (1/D) * sum_k |x_i[k] - x_j[k]|
  ρ(x)        = (1/n) * sum_i ( (1/n) * sum_j d(x_i,x_j) )  # double average
Preprocessing:
  - Per-column min–max to [0,1] so each component diff ≤ 1 → max d(x_i,x_j) = 1

Diversity subset:
  - ONLY cases with side_collision > 0 OR object_collision > 0
  - Vector = [ego(x,y,yaw), npc1(x,y,yaw), ..., npc20(x,y,yaw)] with D=63

Coverage:
  - For the same collision subset, take ONLY NPC positions within R meters of the ego (R=50 m)
  - Query CARLA map spawn points via carla_map.get_spawn_points()
  - Match used NPC positions to nearest spawn point with tolerance (default 0.05 m)
  - Coverage = 100 * (#unique spawn points used) / (#total spawn points on map)
  - If CARLA unreachable or no map spawn points, report N/A

Violation stats:
  - Across ALL scenarios (not just collision): for each type print:
    (#scenarios had it, total occurrences)

USAGE:
  python compute_diversity.py your_log.jsonl
"""

import argparse
import json
import os
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
from collections import Counter
import math

# ------------------- constants -------------------
NPC_EXPECTED = 20
COLLISION_KEYS = ["side_collision", "object_collision"]
ALL_VIOL_KEYS = ["side_collision", "object_collision", "timeout", "red_light", "cross_solid"]

# Coverage matching tolerance (meters) and ego radius filter
SPAWN_MATCH_TOL = 0.05   # meters; matching tolerance to a map spawn point
EGO_RADIUS_M    = 50   # meters; ONLY count NPC spawns within this radius from ego

# ------------------- helpers -------------------
def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if isinstance(x, (int, float)) and np.isfinite(x):
            return float(x)
        if isinstance(x, str):
            return float(x.strip())
    except Exception:
        pass
    return default

def is_collision_case(d: Dict[str, Any]) -> bool:
    """Used for diversity & coverage subsets: side_collision > 0 or object_collision > 0."""
    for k in COLLISION_KEYS:
        v = d.get(k, 0)
        try:
            if int(v) > 0:
                return True
        except Exception:
            if isinstance(v, bool) and v:
                return True
            if isinstance(v, str) and v.strip().lower() in {"true", "1", "yes", "y"}:
                return True
    return False

def build_vector_from_conf(conf: Dict[str, Any]) -> np.ndarray:
    """
    Vector per case:
      [ ego(x,y,yaw), npc1(x,y,yaw), ..., npc20(x,y,yaw) ]  → D = 63
    """
    # ego
    ego = conf.get("ego_transform", {}) or {}
    eloc = ego.get("location", {}) or {}
    erot = ego.get("rotation", {}) or {}
    ex = _to_float(eloc.get("x", 0.0))
    ey = _to_float(eloc.get("y", 0.0))
    eyaw = _to_float(erot.get("yaw", 0.0))
    vec = [ex, ey, eyaw]

    # NPCs (keep input order; pad/truncate to 20)
    arr = conf.get("surrounding_info", []) or []
    for i in range(NPC_EXPECTED):
        if i < len(arr) and isinstance(arr[i], dict):
            tf = arr[i].get("transform", {}) or {}
            loc = tf.get("location", {}) or {}
            rot = tf.get("rotation", {}) or {}
            nx = _to_float(loc.get("x", 0.0))
            ny = _to_float(loc.get("y", 0.0))
            nyaw = _to_float(rot.get("yaw", 0.0))
            vec.extend([nx, ny, nyaw])
        else:
            vec.extend([0.0, 0.0, 0.0])
    return np.asarray(vec, dtype=float)

def collect_npc_xy_within_radius_from_conf(conf: Dict[str, Any], radius_m: float) -> List[Tuple[float, float]]:
    """For coverage: collect (x,y) of NPCs that are within radius_m of the ego."""
    xy: List[Tuple[float, float]] = []
    # ego position
    ego = conf.get("ego_transform", {}) or {}
    eloc = ego.get("location", {}) or {}
    ex = _to_float(eloc.get("x", 0.0))
    ey = _to_float(eloc.get("y", 0.0))

    r2 = radius_m * radius_m

    arr = conf.get("surrounding_info", []) or []
    for e in arr:
        if not isinstance(e, dict):
            continue
        tf = e.get("transform", {}) or {}
        loc = tf.get("location", {}) or {}
        x = _to_float(loc.get("x", 0.0))
        y = _to_float(loc.get("y", 0.0))
        dx = x - ex
        dy = y - ey
        if (dx*dx + dy*dy) <= r2:
            xy.append((x, y))
    return xy

def minmax_to_01(X: np.ndarray) -> np.ndarray:
    """Column-wise min–max to [0,1]. Columns with zero range stay 0."""
    Xmin = X.min(axis=0)
    Xmax = X.max(axis=0)
    denom = Xmax - Xmin
    denom[denom == 0.0] = 1.0
    return (X - Xmin) / denom

def rho_componentwise_mean_abs(X01: np.ndarray) -> float:
    """
    With X in [0,1]^D:
      diff = |X_i - X_j| (n,n,D)
      d_ij = mean_k(diff_ijk)
      ρ = mean_i mean_j d_ij  == diff.mean() over (i,j,k).
    """
    if X01.ndim != 2 or X01.shape[0] == 0:
        raise ValueError("Empty vectors for diversity computation.")
    diff = np.abs(X01[:, None, :] - X01[None, :, :])  # (n,n,D)
    return float(diff.mean())

# ------------------- coverage via CARLA -------------------
def _get_carla_spawn_points() -> Optional[List[Tuple[float, float]]]:
    """
    Connect to CARLA and read map spawn points.
    Env: CARLA_HOST (default '127.0.0.1'), CARLA_PORT (default 2000).
    Returns list of (x,y) or None if unavailable.
    """
    host = os.environ.get("CARLA_HOST", "127.0.0.1")
    port_str = os.environ.get("CARLA_PORT", "2000")
    try:
        port = int(port_str)
    except Exception:
        port = 2000
    try:
        import carla  # requires CARLA egg in PYTHONPATH
        client = carla.Client(host, port)
        client.set_timeout(2.0)
        world = client.get_world()
        carla_map = world.get_map()
        sps = carla_map.get_spawn_points()  # List[carla.Transform]
        if not sps:
            return None
        pts = []
        for t in sps:
            loc = t.location
            pts.append((float(loc.x), float(loc.y)))
        return pts
    except Exception:
        return None

def _euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def compute_coverage(used_xy: List[Tuple[float, float]],
                     tol: float = SPAWN_MATCH_TOL) -> Tuple[Optional[int], Optional[int], Optional[float]]:
    """
    Compare used NPC (x,y) against CARLA map spawn points with a tolerance.
    Returns (total_spawn_points, used_spawn_points, coverage_percent) or (None,None,None) if unavailable.
    """
    sp_xy = _get_carla_spawn_points()
    if not sp_xy:
        return None, None, None

    matched_indices = set()
    m = len(sp_xy)
    for p in used_xy:
        # brute-force nearest
        best_idx = None
        best_d = float("inf")
        for idx in range(m):
            d = _euclidean(p, sp_xy[idx])
            if d < best_d:
                best_d = d
                best_idx = idx
            if best_d <= tol:
                break
        if best_d <= tol and best_idx is not None:
            matched_indices.add(best_idx)

    used_cnt = len(matched_indices)
    total_cnt = m
    cov = 100.0 * used_cnt / total_cnt if total_cnt > 0 else None
    return total_cnt, used_cnt, cov

# ------------------- main -------------------
def main():
    parser = argparse.ArgumentParser(description=(
        "Compute ρ(x) on collision cases; coverage vs CARLA spawn points using NPCs within 50 m of ego; "
        "print ALL-scenario violation stats."
    ))
    parser.add_argument("jsonl", type=str, help="Path to JSONL file")
    args = parser.parse_args()

    # Load JSONL
    objs: List[Dict[str, Any]] = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                objs.append(json.loads(s))
            except json.JSONDecodeError:
                continue

    if not objs:
        raise ValueError("No records loaded from JSONL.")

    # --- Diversity & Coverage subset: collision-only ---
    coll = [o for o in objs if isinstance(o, dict) and is_collision_case(o)]
    if not coll:
        raise ValueError("No collision cases found (side_collision/object_collision both zero).")

    # Build vectors and collect NPC positions WITHIN ego radius
    vecs: List[np.ndarray] = []
    used_xy_all: List[Tuple[float, float]] = []  # for coverage (within EGO_RADIUS_M)
    skipped_for_div = 0
    for o in coll:
        conf = o.get("scenario_conf", None)
        if not isinstance(conf, dict):
            skipped_for_div += 1
            continue
        vecs.append(build_vector_from_conf(conf))
        used_xy_all.extend(collect_npc_xy_within_radius_from_conf(conf, EGO_RADIUS_M))

    if not vecs:
        raise ValueError(
            "All collision cases are missing 'scenario_conf'. Cannot build vectors for diversity."
        )

    X = np.vstack(vecs)  # (n_used, 63)

    # Normalize per column to [0,1] and compute ρ(x)
    X01 = minmax_to_01(X)
    if np.all(np.std(X01, axis=0) == 0):
        raise ValueError(
            "All normalized features have zero variance. Likely failed to parse ego/NPC (x,y,yaw). "
            "Ensure 'scenario_conf.ego_transform' and 'scenario_conf.surrounding_info[*].transform' exist with valid numbers."
        )
    rho = rho_componentwise_mean_abs(X01)

    # --- Coverage vs CARLA spawn points (NPCs within 50 m of ego) ---
    total_sp, used_sp, cov = compute_coverage(used_xy_all, tol=SPAWN_MATCH_TOL)

    # --- Violation stats across ALL scenarios ---
    scen_flags_all = Counter({k: 0 for k in ALL_VIOL_KEYS})
    total_occ_all  = Counter({k: 0 for k in ALL_VIOL_KEYS})
    for o in objs:
        if not isinstance(o, dict):
            continue
        for k in ALL_VIOL_KEYS:
            v = o.get(k, 0)
            c = 0
            try:
                c = int(v)
            except Exception:
                if isinstance(v, bool) and v:
                    c = 1
                elif isinstance(v, str) and v.strip().lower() in {"true", "1", "yes", "y"}:
                    c = 1
            total_occ_all[k] += c
            if c > 0:
                scen_flags_all[k] += 1

    # --- Report ---
    print(f"Collision cases found: {len(coll)}  |  used with vectors: {len(vecs)}  |  skipped(no scenario_conf): {skipped_for_div}")
    print(f"Vector dimension (D): {X.shape[1]}  [ego(3) + {NPC_EXPECTED}*3]")
    print(f"Diversity ρ(x) = {rho:.6f}  (componentwise-mean |Δ|; per-feature min–max to [0,1])")

    if total_sp is None or used_sp is None or cov is None:
        print("\nCoverage (NPCs within {:.1f} m of ego): N/A (CARLA map not reachable or no spawn points).".format(EGO_RADIUS_M))
        print("  Hint: run a CARLA server and set CARLA_HOST/CARLA_PORT if needed;")
        print("        coverage compares NPC positions to map spawn points within {:.3f} m.".format(SPAWN_MATCH_TOL))
    else:
        print("\nCoverage over CARLA map spawn points (only NPCs within {:.1f} m of ego):".format(EGO_RADIUS_M))
        print(f"  total spawn points on map : {total_sp}")
        print(f"  used spawn points (unique): {used_sp}")
        print(f"  coverage percent          : {cov:.2f}%  (match tol = {SPAWN_MATCH_TOL:.3f} m)")

    print("\nViolation stats across ALL scenarios:")
    for k in ALL_VIOL_KEYS:
        print(f"  - {k}: {scen_flags_all[k]} scenarios had it; total occurrences = {total_occ_all[k]}")

if __name__ == "__main__":
    main()
