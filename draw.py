# -*- coding: utf-8 -*-
"""
Overlay 3 datasets (JSONL) in histograms with distinct colors,
SAVE SEPARATE RELATIVE DENSITY (hexbin) plots per dataset,
and ALSO SAVE SEPARATE ABSOLUTE NPC POSITION SCATTER plots per dataset.

Example:
    python overlay_relative_distributions_en_density_abs.py \
        --inputs fileA.jsonl fileB.jsonl fileC.jsonl \
        --outdir ./out \
        --labels A B C \
        --bins 60 --density \
        --gridsize 60 --mincnt 1 --log \
        --abs-scatter-alpha 0.5 --abs-scatter-max 0 --seed 0

Outputs:
- overlay_rel_x_hist.png, overlay_rel_y_hist.png, overlay_rel_heading_hist.png
- hexbin_<label>.png      (relative X-Y density per dataset)
- abs_scatter_<label>.png (absolute NPC X-Y scatter per dataset)
- combined_relative_points.csv (rel_x, rel_y, dataset)
- combined_relative_headings_deg.csv (rel_heading_deg, dataset)
- combined_abs_npc_points.csv (abs_x, abs_y, dataset)

Notes:
- Matplotlib only (no seaborn). Each chart is its own figure.
- Hist overlay uses explicit distinct colors because the user requested different colors for the 3 files.
- Hexbin uses default colormap (no explicit color setup).
- Absolute scatter uses no explicit color; one figure per dataset.
"""

import os
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ----------------------------- Parsing helpers -----------------------------

def _get_scalar(d: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except Exception:
                pass
    return None


def _extract_xy_from_any(obj: Any) -> Tuple[Optional[float], Optional[float]]:
    """Extract x, y from common structures (including 'transform' nesting)."""
    if isinstance(obj, dict):
        # Direct x/y
        x = _get_scalar(obj, ["x", "X"])
        y = _get_scalar(obj, ["y", "Y"])
        if x is not None and y is not None:
            return x, y

        # Nested common keys
        for nk in ["pos", "position", "location", "translation", "center", "coord", "coords"]:
            if nk in obj and obj[nk] is not None:
                v = obj[nk]
                if isinstance(v, dict):
                    x = _get_scalar(v, ["x", "X"])
                    y = _get_scalar(v, ["y", "Y"])
                    if x is not None and y is not None:
                        return x, y
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    try:
                        return float(v[0]), float(v[1])
                    except Exception:
                        pass

        # 'transform' structure
        if "transform" in obj and isinstance(obj["transform"], dict):
            return _extract_xy_from_any(obj["transform"])
    return None, None


def _extract_heading_from_any(obj: Any) -> Optional[float]:
    """Extract heading/yaw/theta (yaw preferred) from common structures."""
    if isinstance(obj, dict):
        direct = _get_scalar(obj, ["heading", "Heading", "yaw", "Yaw", "theta", "Theta"])
        if direct is not None:
            return direct
        for nk in ["rotation", "Rotation", "orientation", "Orientation", "rot", "Rot"]:
            if nk in obj and isinstance(obj[nk], dict):
                yaw = _get_scalar(obj[nk], ["yaw", "Yaw", "heading", "Heading", "theta", "Theta"])
                if yaw is not None:
                    return yaw
        if "transform" in obj and isinstance(obj["transform"], dict):
            return _extract_heading_from_any(obj["transform"])
    return None


def angle_diff_deg(a_deg: float, b_deg: float) -> float:
    """Return angular difference normalized to [-180, 180]."""
    d = a_deg - b_deg
    d = (d + 180.0) % 360.0 - 180.0
    return d


def rad_to_deg_if_needed(values: List[float]) -> List[float]:
    """Heuristically detect radians and convert to degrees; otherwise keep as degrees."""
    arr = np.array(values, dtype=float)
    if arr.size == 0 or not np.any(~np.isnan(arr)):
        return list(arr)
    q95 = np.quantile(np.abs(arr[~np.isnan(arr)]), 0.95)
    if q95 <= (2 * np.pi) * 1.2:
        return list(np.degrees(arr))
    else:
        return list(arr)


def parse_records(path: str):
    """
    Parse one JSONL file and return:
        rel_x, rel_y, rel_heading_deg, abs_x, abs_y, cases_used, npc_counted
    """
    rel_xs: List[float] = []
    rel_ys: List[float] = []
    rel_h_deg: List[float] = []
    abs_xs: List[float] = []
    abs_ys: List[float] = []
    testcase_count = 0
    npc_total = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            ego_obj = None
            npc_list = []

            sc = rec.get("scenario_conf")
            if isinstance(sc, dict):
                if "ego_transform" in sc and isinstance(sc["ego_transform"], dict):
                    ego_obj = sc["ego_transform"]
                if "surrounding_info" in sc and isinstance(sc["surrounding_info"], list):
                    npc_list = [it for it in sc["surrounding_info"] if isinstance(it, dict)]

            # Fallbacks
            if ego_obj is None:
                for k in ["ego", "ego_vehicle", "egoState", "ego_state"]:
                    if k in rec and isinstance(rec[k], dict):
                        ego_obj = rec[k]
                        break
            if not npc_list:
                for k in ["npcs", "npc_vehicles", "vehicles", "actors", "objects", "surrounding_info", "surroundings"]:
                    if k in rec and isinstance(rec.get(k), list):
                        npc_list = [it for it in rec[k] if isinstance(it, dict)]
                        break

            if ego_obj is None or not npc_list:
                continue

            ex, ey = _extract_xy_from_any(ego_obj)
            eh = _extract_heading_from_any(ego_obj)
            if ex is None or ey is None:
                continue

            local_used = 0
            h_pairs_ego: List[float] = []
            h_pairs_npc: List[float] = []

            for npc in npc_list:
                nx, ny = _extract_xy_from_any(npc)
                nh = _extract_heading_from_any(npc)
                if nx is None or ny is None:
                    continue

                # relative
                rel_xs.append(nx - ex)
                rel_ys.append(ny - ey)
                # absolute
                abs_xs.append(nx)
                abs_ys.append(ny)

                local_used += 1

                if (eh is not None) and (nh is not None):
                    h_pairs_ego.append(eh)
                    h_pairs_npc.append(nh)

            if local_used > 0:
                testcase_count += 1
                npc_total += local_used

            if h_pairs_ego and h_pairs_npc and len(h_pairs_ego) == len(h_pairs_npc):
                ego_deg = rad_to_deg_if_needed(h_pairs_ego)
                npc_deg = rad_to_deg_if_needed(h_pairs_npc)
                rel_h_deg.extend([angle_diff_deg(npc_deg[i], ego_deg[i]) for i in range(len(ego_deg))])

    return rel_xs, rel_ys, rel_h_deg, abs_xs, abs_ys, testcase_count, npc_total


# ----------------------------- Plotting helpers -----------------------------

def unified_edges(all_vals: List[float], bins: int):
    """Return common bin edges for comparable overlaid histograms."""
    arr = np.asarray(all_vals, dtype=float)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return np.linspace(0, 1, bins + 1)
    lo = np.nanmin(arr)
    hi = np.nanmax(arr)
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = 0.0, 1.0
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    return np.linspace(lo, hi, bins + 1)


def safe_label_to_filename(label: str) -> str:
    """Sanitize label text into a filename-friendly token."""
    txt = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label)
    return txt.strip("_") or "dataset"


def maybe_subsample(x: List[float], y: List[float], max_n: int, seed: int = 0):
    """Optionally subsample points for scatter plots to avoid overplotting heavy files."""
    if max_n is None or max_n <= 0 or len(x) <= max_n:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=max_n, replace=False)
    idx.sort()
    x_s = [x[i] for i in idx]
    y_s = [y[i] for i in idx]
    return x_s, y_s


# ----------------------------------- Main -----------------------------------

def main():
    parser = argparse.ArgumentParser(description="Overlay histograms for 3 datasets, hexbin relative density, and ABSOLUTE NPC scatter per dataset.")
    parser.add_argument("--inputs", nargs=3, required=True, help="Three JSONL input files (order = colors & legend).")
    parser.add_argument("--labels", nargs=3, help="Legend labels (optional; defaults to filenames).")
    parser.add_argument("--outdir", default=".", help="Output directory.")
    parser.add_argument("--bins", type=int, default=50, help="Number of bins for histograms.")
    parser.add_argument("--density", action="store_true", help="Use probability density instead of counts for histograms.")
    parser.add_argument("--gridsize", type=int, default=60, help="Hexbin grid size (relative plots).")
    parser.add_argument("--mincnt", type=int, default=1, help="Minimum count in a hexbin to be displayed (relative plots).")
    parser.add_argument("--log", action="store_true", help="Use logarithmic color scaling for hexbin density (relative plots).")
    parser.add_argument("--abs-scatter-alpha", type=float, default=0.6, help="Alpha for absolute NPC scatter points.")
    parser.add_argument("--abs-scatter-max", type=int, default=0, help="If >0, subsample to at most this many points per dataset for absolute scatter.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for subsampling absolute scatter.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    files = args.inputs
    labels = args.labels if args.labels else [os.path.basename(p) for p in files]

    # Distinct colors for overlaid histograms (explicit request)
    colors = ["tab:blue", "tab:orange", "tab:green"]

    # Parse three files
    datasets = []
    for p in files:
        rx, ry, rh, ax, ay, tc, npc = parse_records(p)
        print(f"[{os.path.basename(p)}] test cases used: {tc}, NPC counted: {npc}, rel points: {len(rx)}, rel heading: {len(rh)}, abs points: {len(ax)}")
        datasets.append((rx, ry, rh, ax, ay))

    # ---------------- Export combined CSVs with dataset column ----------------
    comb_points_path = os.path.join(args.outdir, "combined_relative_points.csv")
    comb_head_path = os.path.join(args.outdir, "combined_relative_headings_deg.csv")
    comb_abs_path = os.path.join(args.outdir, "combined_abs_npc_points.csv")
    rows_pts = []
    rows_h = []
    rows_abs = []
    for (rx, ry, rh, ax, ay), lab in zip(datasets, labels):
        rows_pts.extend([{"rel_x": x, "rel_y": y, "dataset": lab} for x, y in zip(rx, ry)])
        rows_h.extend([{"rel_heading_deg": h, "dataset": lab} for h in rh])
        rows_abs.extend([{"abs_x": x, "abs_y": y, "dataset": lab} for x, y in zip(ax, ay)])
    pd.DataFrame(rows_pts).to_csv(comb_points_path, index=False)
    pd.DataFrame(rows_h).to_csv(comb_head_path, index=False)
    pd.DataFrame(rows_abs).to_csv(comb_abs_path, index=False)
    print("Wrote:", comb_points_path)
    print("Wrote:", comb_head_path)
    print("Wrote:", comb_abs_path)

    # ---------------- Overlaid histograms (shared bin edges) ----------------
    # Relative X
    all_x = [v for (rx, _, _, _, _ ) in datasets for v in rx]
    x_edges = unified_edges(all_x, args.bins)
    plt.figure()
    for (rx, _, _, _, _), lab, c in zip(datasets, labels, colors):
        plt.hist(rx, bins=x_edges, density=args.density, histtype="step", linewidth=1.8, label=lab, color=c)
    plt.title("Relative X Distribution (Overlay)")
    plt.xlabel("Relative X (NPC - Ego)")
    plt.ylabel("Density" if args.density else "Count")
    plt.legend()
    plt.tight_layout()
    out_x = os.path.join(args.outdir, "overlay_rel_x_hist.png")
    plt.savefig(out_x, dpi=160)
    plt.close()
    print("Wrote:", out_x)

    # Relative Y
    all_y = [v for (_, ry, _, _, _ ) in datasets for v in ry]
    y_edges = unified_edges(all_y, args.bins)
    plt.figure()
    for (_, ry, _, _, _), lab, c in zip(datasets, labels, colors):
        plt.hist(ry, bins=y_edges, density=args.density, histtype="step", linewidth=1.8, label=lab, color=c)
    plt.title("Relative Y Distribution (Overlay)")
    plt.xlabel("Relative Y (NPC - Ego)")
    plt.ylabel("Density" if args.density else "Count")
    plt.legend()
    plt.tight_layout()
    out_y = os.path.join(args.outdir, "overlay_rel_y_hist.png")
    plt.savefig(out_y, dpi=160)
    plt.close()
    print("Wrote:", out_y)

    # Relative Heading (deg, normalized to [-180, 180])
    all_h = [v for (_, _, rh, _, _ ) in datasets for v in rh]
    h_edges = unified_edges(all_h, args.bins)
    plt.figure()
    for (_, _, rh, _, _), lab, c in zip(datasets, labels, colors):
        plt.hist(rh, bins=h_edges, density=args.density, histtype="step", linewidth=1.8, label=lab, color=c)
    plt.title("Relative Heading Distribution (deg, Overlay)")
    plt.xlabel("Relative Heading (deg, NPC - Ego, in [-180, 180])")
    plt.ylabel("Density" if args.density else "Count")
    plt.legend()
    plt.tight_layout()
    out_h = os.path.join(args.outdir, "overlay_rel_heading_hist.png")
    plt.savefig(out_h, dpi=160)
    plt.close()
    print("Wrote:", out_h)

    # --------------------- Hexbin density plots per dataset (relative) ---------------------
    for (rx, ry, _, _, _), lab in zip(datasets, labels):
        plt.figure()
        bins_arg = 'log' if args.log else None
        hb = plt.hexbin(rx, ry, gridsize=args.gridsize, mincnt=args.mincnt, bins=bins_arg)
        plt.colorbar(hb, label="Count" + (" (log)" if args.log else ""))
        # plt.title(f"Relative Position Density (Hexbin) - {lab}")
        plt.xlabel("Relative X")
        plt.ylabel("Relative Y")
        plt.tight_layout()
        fname = f"hexbin_{safe_label_to_filename(lab)}.png"
        out_hex = os.path.join(args.outdir, fname)
        plt.savefig(out_hex, dpi=160)
        plt.close()
        print("Wrote:", out_hex)

    # --------------------- Absolute NPC scatter plots per dataset ---------------------
    for (_, _, _, ax, ay), lab in zip(datasets, labels):
        xs, ys = maybe_subsample(ax, ay, args.abs_scatter_max, args.seed)
        plt.figure()
        plt.scatter(xs, ys, s=6, alpha=args.abs_scatter_alpha)  # no explicit color; separate figure
        # plt.title(f"Absolute NPC Position Scatter - {lab}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.tight_layout()
        fname = f"abs_scatter_{safe_label_to_filename(lab)}.png"
        out_scatter = os.path.join(args.outdir, fname)
        plt.savefig(out_scatter, dpi=160)
        plt.close()
        print("Wrote:", out_scatter)


if __name__ == "__main__":
    main()
