# sb_art.py
# -*- coding: utf-8 -*-
"""
Scenario-based ART with adaptive triggers (AGGRESSIVE baseline, fixed & improved).

Key changes:
- TTC is only computed when the target is “directly ahead with small lateral offset”; otherwise treated as infinity to avoid physical inconsistency.
- RiskScorer's lateral term changed to “centrality” scoring (ly near center is most dangerous), removed redundant rvx term.
- approaching_cnt counting corrected to: directly ahead and rvx<0.
- Trigger decision recalculates TTC using the policy's lat_range, decoupled from compute_phi, for consistency.
- KNN memory bank vector construction uniformly sorted by vehicle id (canonicalize) to avoid distance distortion caused by permutations.
- All other interfaces remain compatible: ARTSelectorScenario / compute_phi / update_adaptive_policy /
  filter_triggerables / build_actions_for.
"""

from typing import List, Tuple, Dict, Optional
import math
import random
import numpy as np
from itertools import product

from ptop.utils.geometry import to_local, vel_local

try:
    import carla
except Exception:
    carla = None


# ============ Local coordinates / features ============

def _ttc(lx: float, ly: float, rvx: float, lat_range: float = 6.0, eps: float = 1e-3) -> float:
    “””
    Compute longitudinal TTC only when “directly ahead with small lateral offset”; otherwise return +inf (approximated by a large number).
    rvx = ovx - evx (in local coordinate system); closing speed = -rvx.
    “””
    if not (lx > 0 and abs(ly) < lat_range):
        return 1e6
    rel_speed_long = -rvx
    if rel_speed_long <= eps:
        return 1e6
    return lx / rel_speed_long

def compute_phi(ego: "carla.Vehicle", other: "carla.Vehicle") -> np.ndarray:
    """
    φ = [lx_norm, ly_norm, rvx_norm, rvy_norm, inv_ttc] ∈ [0,1]^5
    Normalization ranges:
      lx ∈ [-40, 40]m, ly ∈ [-10, 10]m, rvx/rvy ∈ [-10, 10]m/s, ttc mapped to [0,1] (5s bandwidth)
    """
    etf = ego.get_transform()
    lx, ly = to_local(etf, other.get_location())
    evx, evy = vel_local(etf, ego.get_velocity())
    ovx, ovy = vel_local(etf, other.get_velocity())
    rvx, rvy = ovx - evx, ovy - evy
    ttc = _ttc(lx, ly, rvx, lat_range=6.0)

    phi = np.array([
        np.clip((lx + 40.0) / 80.0, 0.0, 1.0),
        np.clip((ly + 10.0) / 20.0, 0.0, 1.0),
        np.clip((rvx + 10.0) / 20.0, 0.0, 1.0),
        np.clip((rvy + 10.0) / 20.0, 0.0, 1.0),
        np.clip(1.0 - min(ttc / 5.0, 1.0), 0.0, 1.0),
    ], dtype=np.float32)
    return phi


# ============ Hazard scoring ============

class RiskScorer:
    def __init__(self, model: Optional[object] = None):
        self.model = model

    def __call__(self, phi: np.ndarray) -> float:
        “””
        If a model is provided (outputs logit), use the model; otherwise use heuristic:
        primarily based on inv_ttc, supplemented by a “laterally centered is more dangerous” preference.
        “””
        if self.model is not None:
            import torch
            with torch.no_grad():
                x = torch.from_numpy(phi).float().unsqueeze(0)
                y = self.model(x)            # assumed to output logit
                y = torch.sigmoid(y)
                return float(y.item())

        lx_norm, ly_norm, rvx_norm, rvy_norm, inv_ttc = phi
        # ly_norm ∈ [0,1] -> centrality (1 means ly≈0)
        c_ly = 2.0 * float(ly_norm) - 1.0
        lat_center = 1.0 - abs(c_ly)
        score = 0.75 * float(inv_ttc) + 0.25 * lat_center
        return float(np.clip(score, 0.0, 1.0))


# ============ ART main class ============

def _canonicalize(vehicles: List["carla.Vehicle"]) -> List["carla.Vehicle"]:
    """Sort by vehicle id to ensure consistency in vector construction."""
    return sorted(vehicles, key=lambda vv: vv.id)

class ARTSelectorScenario:
    def __init__(
        self,
        scale_action: float = 1.0,
        phi_weights: Tuple[float, ...] = (1, 1, 1, 1, 1),
        hazard_tau: float = 0.42,
        max_considered: int = 3,
        risk_scorer: Optional[RiskScorer] = None,
    ):
        self.scale_action = float(scale_action)
        self.phi_weights = np.asarray(phi_weights, dtype=np.float32)
        self.phi_dim = len(self.phi_weights)
        self.hazard_tau = float(hazard_tau)
        self.max_considered = int(max_considered)
        self.risk = risk_scorer or RiskScorer(model=None)

        # Note: keep naming consistent with external action_trans (use 'break' not 'brake')
        self.actions: Dict[str, Tuple[int, int]] = {
            'break': (0, -1),
            'accelerate': (0, 1),
            'right_change_acc': (1, 1),
            'right_change_dec': (1, -1),
            'left_change_acc': (-1, 1),
            'left_change_dec': (-1, -1)
        }
        self.code_to_action = {v: k for k, v in self.actions.items()}

        # Memory bank: bucketed by number of triggered vehicles (up to 4)
        self.risk_set: Dict[int, List[Tuple[float, ...]]] = {1: [], 2: [], 3: [], 4: []}
        self.safe_set: Dict[int, List[Tuple[float, ...]]] = {1: [], 2: [], 3: [], 4: []}

    # ---- KNN distance (with feature weights + action scale) ----
    def distance_KN(self, vec1: Tuple[float, ...], vec2: Tuple[float, ...], num_vehicle: int) -> float:
        total_sq = 0.0
        stride = self.phi_dim + 2
        W = self.phi_weights
        for i in range(num_vehicle):
            b = stride * i
            dphi = np.asarray(vec1[b:b + self.phi_dim]) - np.asarray(vec2[b:b + self.phi_dim])
            total_sq += float(np.sum((W * dphi) ** 2))
            a1x, a1y = vec1[b + self.phi_dim:b + self.phi_dim + 2]
            a2x, a2y = vec2[b + self.phi_dim:b + self.phi_dim + 2]
            total_sq += ((a1x - a2x) / self.scale_action) ** 2 + ((a1y - a2y) / self.scale_action) ** 2
        return math.sqrt(total_sq)

    # ---- Construct vector in unified order (sorted by id), actions_list corresponds one-to-one with vehicle_list ----
    def build_global_vector(self, ego_vehicle: "carla.Vehicle",
                            vehicle_list: List["carla.Vehicle"],
                            actions_list: List[Tuple[int, int]]) -> Tuple[float, ...]:
        vec: List[float] = []
        # Build id->action mapping, then concatenate sorted by id
        act_map: Dict[int, Tuple[int, int]] = {v.id: a for v, a in zip(vehicle_list, actions_list)}
        ordered = _canonicalize(vehicle_list)
        for v in ordered:
            phi = compute_phi(ego_vehicle, v)
            dx, dy = act_map[v.id]
            vec.extend(phi.tolist() + [float(dx), float(dy)])
        return tuple(vec)

    def parse_global_vector_to_actions(self, global_vec: Tuple[float, ...]) -> List[str]:
        """If you need to recover actions from a vector (in internal canonical order), use this function."""
        actions_list: List[str] = []
        stride = self.phi_dim + 2
        num_vehicle = len(global_vec) // stride
        for i in range(num_vehicle):
            b = stride * i
            ax = int(global_vec[b + self.phi_dim + 0])
            ay = int(global_vec[b + self.phi_dim + 1])
            actions_list.append(self.code_to_action.get((ax, ay), "Unknown"))
        return actions_list

    # ---- Memory bank recording: ensure vehicle_list is consistent with the “trigger set semantics” at selection time ----
    def record_outcome(self, ego_vehicle: "carla.Vehicle",
                       vehicle_list: List["carla.Vehicle"],
                       actions_name_list: List[str],
                       violated: bool = False) -> None:
        combos = [self.actions[a] for a in actions_name_list]
        gv = self.build_global_vector(ego_vehicle, vehicle_list, combos)
        key = min(len(vehicle_list), 4)
        (self.risk_set if violated else self.safe_set)[key].append(gv)

    # ---- Core: select actions for all vehicles in a frame (internally selects the trigger subset) ----
    def choose_actions_for_all_vehicles(self, ego_vehicle: "carla.Vehicle",
                                        vehicle_list: List["carla.Vehicle"],
                                        hazard_tau: Optional[float] = None,
                                        max_considered: Optional[int] = None
                                        ) -> Tuple[List[str], Tuple[float, ...]]:
        hz_tau = self.hazard_tau if hazard_tau is None else float(hazard_tau)
        max_cons = self.max_considered if max_considered is None else int(max_considered)

        # 1) First filter trigger targets based on heuristic/model (sorted by score descending)
        veh_scored: List[Tuple[float, "carla.Vehicle"]] = []
        for v in vehicle_list:
            phi = compute_phi(ego_vehicle, v)
            h = self.risk(phi)
            if h >= hz_tau:
                veh_scored.append((h, v))
        veh_scored.sort(key=lambda x: x[0], reverse=True)
        trigger_list = [v for _, v in veh_scored[:max_cons]]

        # If no trigger targets: assign random actions to all vehicles as fallback
        if not trigger_list:
            rnd_names = np.random.choice(list(self.actions.keys()), size=len(vehicle_list)).tolist()
            rnd_vec = self.build_global_vector(ego_vehicle, vehicle_list,
                                               [self.actions[a] for a in rnd_names])
            return rnd_names, rnd_vec

        # 2) KNN memory bank scoring (combinatorial search over trigger subset only)
        n = len(trigger_list)
        key = n if n < 4 else 4
        risk_set = self.risk_set[key]
        safe_set = self.safe_set[key]

        # Candidate action ordering: try “lateral deceleration/evasion” first, then acceleration/deceleration, etc.
        possible_actions = [
            self.actions['right_change_dec'],
            self.actions['left_change_dec'],
            self.actions['accelerate'],
            self.actions['break'],
            self.actions['right_change_acc'],
            self.actions['left_change_acc'],
        ]

        ordered_triggers = _canonicalize(trigger_list)
        all_combos = product(possible_actions, repeat=n)

        best_combo_codes: Optional[Tuple[Tuple[int,int], ...]] = None
        best_tuple = None  # sort key cache

        def sort_key(ds: Optional[float], dr: Optional[float]) -> Tuple[float, float, float]:
            # Want ds large (farther from safe cases), dr small (closer to risky cases)
            eps = random.random() * 1e-6
            if ds is None and dr is None:
                return (1e9, 1e9, eps)
            elif ds is None:
                return (0.0, float(dr), eps)
            elif dr is None:
                return (-float(ds), 0.0, eps)
            else:
                return (-float(ds), float(dr), eps)

        for combo in all_combos:
            gv_trigger = self.build_global_vector(ego_vehicle, ordered_triggers, list(combo))
            ds = min(self.distance_KN(gv_trigger, s, n) for s in safe_set) if safe_set else None
            dr = min(self.distance_KN(gv_trigger, r, n) for r in risk_set) if risk_set else None
            sk = sort_key(ds, dr)
            if (best_tuple is None) or (sk < best_tuple):
                best_tuple = sk
                best_combo_codes = combo

        # 3) Build action list for “all vehicles” (default for non-triggered vehicles is accelerate)
        id2act_name: Dict[int, str] = {}
        if best_combo_codes is not None:
            for v, code in zip(ordered_triggers, best_combo_codes):
                id2act_name[v.id] = self.code_to_action[code]

        actions_full_names: List[str] = []
        actions_full_codes: List[Tuple[int,int]] = []
        for v in vehicle_list:
            a_name = id2act_name.get(v.id, 'accelerate')
            actions_full_names.append(a_name)
            actions_full_codes.append(self.actions[a_name])

        gv_full = self.build_global_vector(ego_vehicle, vehicle_list, actions_full_codes)
        return actions_full_names, gv_full


# ============ Adaptive policy (more aggressive baseline) ============
def _safe_count_driving_lanes(wp, max_hops=8, stop_on_road_change=True):
    """Count drivable lanes to the left and right from the current waypoint, with max hops and visited set to prevent cycles."""
    if wp is None:
        return 1
    base_key = (wp.road_id, getattr(wp, "section_id", 0))
    visited = set([(wp.road_id, getattr(wp, "section_id", 0), wp.lane_id)])
    cnt = 1

    # To the left
    cur = wp
    for _ in range(max_hops):
        nxt = cur.get_left_lane()
        if not nxt or nxt.lane_type != carla.LaneType.Driving:
            break
        if stop_on_road_change and nxt.road_id != base_key[0]:
            break
        key = (nxt.road_id, getattr(nxt, "section_id", 0), nxt.lane_id)
        if key in visited:
            break
        visited.add(key)
        cnt += 1
        cur = nxt

    # To the right
    cur = wp
    for _ in range(max_hops):
        nxt = cur.get_right_lane()
        if not nxt or nxt.lane_type != carla.LaneType.Driving:
            break
        if stop_on_road_change and nxt.road_id != base_key[0]:
            break
        key = (nxt.road_id, getattr(nxt, "section_id", 0), nxt.lane_id)
        if key in visited:
            break
        visited.add(key)
        cnt += 1
        cur = nxt

    return max(1, cnt)

def analyze_scene(world_map: "carla.Map", ego: "carla.Vehicle",
                  vehicles: List["carla.Vehicle"],
                  long_probe: float = 70.0,
                  lat_probe: float = 9.0) -> Dict[str, float]:
    ego_tf = ego.get_transform()
    w = world_map.get_waypoint(ego_tf.location, project_to_road=True,
                               lane_type=carla.LaneType.Driving)

    # At junctions, degrade to 1 (or reduce max_hops): avoid looping in junction topology
    at_junction = bool(w and w.is_junction)
    if at_junction or not w:
        lane_cnt = 1
    else:
        # Safe counting with max hops and cycle prevention
        lane_cnt = _safe_count_driving_lanes(w, max_hops=8, stop_on_road_change=True)

    # Speed limit -> coarse road type classification
    try:
        speed_limit = float(ego.get_speed_limit())
    except Exception:
        speed_limit = 50.0
    if speed_limit >= 70: road_type = "highway"
    elif speed_limit >= 50: road_type = "urban"
    else: road_type = "low_speed"

    # Compute forward density / approaching statistics
    total_cnt = 0
    forward_cnt = 0
    approaching_cnt = 0
    min_ttc = 1e6

    def _ttc(lx, ly, rvx, lat_range=6.0, eps=1e-3):
        # Only use longitudinal TTC when directly ahead with small lateral offset
        if not (lx > 0 and abs(ly) < lat_range):
            return 1e6
        rel_speed_long = -rvx
        if rel_speed_long <= eps:
            return 1e6
        return lx / rel_speed_long

    evx, evy = vel_local(ego_tf, ego.get_velocity())
    for v in vehicles:
        if v.id == ego.id:
            continue
        total_cnt += 1
        lx, ly = to_local(ego_tf, v.get_location())
        if lx < -5 or lx > long_probe or abs(ly) > lat_probe:
            continue
        forward_cnt += 1
        vvx, vvy = vel_local(ego_tf, v.get_velocity())
        rvx = vvx - evx
        ttc = _ttc(lx, ly, rvx, lat_range=lat_probe)
        min_ttc = min(min_ttc, ttc)
        if lx > 0 and rvx < 0:  # directly ahead and closing
            approaching_cnt += 1

    forward_density = forward_cnt / max(1.0, lane_cnt * (long_probe / 30.0))

    return dict(
        npc_total=float(total_cnt),
        forward_cnt=float(forward_cnt),
        approaching_cnt=float(approaching_cnt),
        forward_density=float(forward_density),
        min_ttc=float(min_ttc),
        at_junction=float(1.0 if at_junction else 0.0),
        road_type=road_type,
        lane_cnt=float(lane_cnt),
    )

def adapt_triggers(stats: Dict[str, float]) -> Dict[str, float]:
    npc = stats["npc_total"]
    dens = stats["forward_density"]
    min_ttc = stats["min_ttc"]
    at_junc = bool(stats["at_junction"] > 0.5)
    road = stats["road_type"]

    # —— More aggressive baseline ——
    hazard_tau = 0.42
    trigger_ttc = 2.6
    lat_range = 7.5
    long_range = 42.0
    sbart_stride = 4
    max_considered = min(4, max(1, math.ceil((npc - 5.0) / 12.0) + 1))

    if road == "highway":
        long_range = 60.0
        lat_range = 6.0
        if dens >= 0.12:
            hazard_tau = 0.40
            trigger_ttc = 2.9
            max_considered = min(4, max_considered + 1)
            sbart_stride = 3
        else:
            hazard_tau = 0.48
            trigger_ttc = 2.5
            sbart_stride = 4
    elif at_junc:
        long_range = 45.0
        lat_range = 9.0
        hazard_tau = 0.38
        trigger_ttc = 3.2
        max_considered = min(4, max_considered + 1)
        sbart_stride = 2
    else:  # urban
        long_range = 45.0
        lat_range = 7.5
        if dens >= 0.16:
            hazard_tau = 0.40
            trigger_ttc = 2.9
            sbart_stride = 3
        else:
            hazard_tau = 0.45
            trigger_ttc = 2.6
            sbart_stride = 4

    # Emergency mode triggers earlier
    if min_ttc < 2.0:
        hazard_tau = min(hazard_tau, 0.36)
        trigger_ttc = max(trigger_ttc, 3.2)
        sbart_stride = 2
        max_considered = min(4, max_considered + 1)

    return dict(
        hazard_tau=float(hazard_tau),
        trigger_ttc=float(trigger_ttc),
        lat_range=float(lat_range),
        long_range=float(long_range),
        max_considered=int(max_considered),
        sbart_stride=int(sbart_stride),
    )

def update_adaptive_policy(world_map: "carla.Map", ego: "carla.Vehicle",
                           vehicles: List["carla.Vehicle"],
                           prev_policy: Optional[Dict[str, float]] = None
                           ) -> Tuple[Dict[str, float], Dict[str, float]]:
    stats = analyze_scene(world_map, ego, vehicles)
    policy = adapt_triggers(stats)
    return policy, stats


# ============ Trigger filtering & entry point ============

def filter_triggerables(ego: "carla.Vehicle", vehicles: List["carla.Vehicle"],
                        art: ARTSelectorScenario, policy: Dict[str, float]) -> List["carla.Vehicle"]:
    """
    Filter trigger targets by policy's ROI and thresholds; return sorted by risk score descending.
    (Note: choose_actions_for_all_vehicles also internally filters trigger targets;
     if you only want to control the trigger subset, pass it as vehicle_list to choose.)
    """
    ego_tf = ego.get_transform()
    L, W = policy["long_range"], policy["lat_range"]
    hz_tau = policy["hazard_tau"]
    ttc_th = policy["trigger_ttc"]

    evx, evy = vel_local(ego_tf, ego.get_velocity())

    scored: List[Tuple[float, "carla.Vehicle"]] = []
    for v in vehicles:
        if v.id == ego.id:
            continue
        lx, ly = to_local(ego_tf, v.get_location())
        if lx < -5.0 or lx > L or abs(ly) > W:
            continue

        phi = compute_phi(ego, v)
        hz = art.risk(phi)

        vvx, vvy = vel_local(ego_tf, v.get_velocity())
        rvx = vvx - evx
        ttc = _ttc(lx, ly, rvx, lat_range=W)

        if hz >= hz_tau or ttc < ttc_th:
            scored.append((hz, v))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [v for _, v in scored]

def build_actions_for(art: ARTSelectorScenario, ego: "carla.Vehicle",
                      trigger_vehicles: List["carla.Vehicle"],
                      full_vehicle_list: Optional[List["carla.Vehicle"]] = None
                      ) -> Tuple[List[str], Tuple[float, ...]]:
    """
    If full_vehicle_list is provided, generate actions for all vehicles (internally filters trigger subset);
    otherwise generate actions only for trigger_vehicles.
    """
    vehicle_list = full_vehicle_list if full_vehicle_list is not None else trigger_vehicles
    actions, gv_full = art.choose_actions_for_all_vehicles(ego, vehicle_list)
    return actions, gv_full


__all__ = [
    "RiskScorer",
    "ARTSelectorScenario",
    "compute_phi",
    "update_adaptive_policy",
    "filter_triggerables",
    "build_actions_for",
]
