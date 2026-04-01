import json
import logging
import math
import time
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Callable, Iterable, List, Optional, Sequence, Set, Tuple

import carla
import numpy as np
import requests

logger = logging.getLogger(__name__)


def sanitize_position_info(pi: dict) -> dict:
    """Backward compatibility: convert surrounding_transforms to surrounding_info."""
    if 'surrounding_transforms' in pi and 'surrounding_info' not in pi:
        pi['surrounding_info'] = [{'transform': t, 'type': 'car'} for t in pi['surrounding_transforms']]
        pi.pop('surrounding_transforms', None)
    if 'surrounding_info' in pi:
        pi['vehicle_num'] = len(pi['surrounding_info'])
    return pi


def purge_npcs(world: carla.World,
               client: carla.Client,
               tm: Optional["carla.TrafficManager"] = None,
               keep_actor_ids: Optional[Iterable[int]] = None,
               include_walkers: bool = True,
               hard_teleport: bool = True) -> Tuple[int, int]:
    """
    Remove all NPCs (vehicles/pedestrians) from the map, keeping Actors specified by keep_actor_ids (e.g., EGO).
    Returns: (remaining vehicle count, remaining pedestrian count) -- excluding Actors in keep_actor_ids.
    """
    keep: Set[int] = set(keep_actor_ids or [])

    def _tick(n: int = 1) -> None:
        for _ in range(n):
            try:
                world.tick()
            except Exception:
                world.wait_for_tick()

    def _flush(s: float = 0.03, n: int = 2) -> None:
        _tick(n)
        if s > 0:
            time.sleep(s)

    actors = world.get_actors()
    vehs = [a for a in actors.filter('vehicle.*') if a.id not in keep]
    if include_walkers:
        walkers = [a for a in actors.filter('walker.pedestrian.*') if a.id not in keep]
        controllers = list(actors.filter('controller.ai.walker'))
        valid_walker_ids = set(w.id for w in walkers)

        def _ctrl_parent_id(ctrl):
            try:
                p = ctrl.parent
                return p.id if p else None
            except Exception:
                return None

        controllers = [c for c in controllers if _ctrl_parent_id(c) in valid_walker_ids]
    else:
        walkers = []
        controllers = []

    # Disable autopilot & brake vehicles
    for v in vehs:
        try:
            if hasattr(v, "set_autopilot"):
                v.set_autopilot(False)
            v.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
        except Exception:
            pass
    _flush(0.03, 2)

    # Stop pedestrian controllers first
    for c in controllers:
        try:
            c.stop()
        except Exception:
            pass
    _flush(0.02, 2)

    # Teleport downward if needed to avoid nested rigid body destruction failure
    if hard_teleport:
        for a in vehs:
            try:
                tf = a.get_transform()
                tf.location.z -= 30.0
                a.set_transform(tf)
            except Exception:
                pass
        for a in walkers:
            try:
                tf = a.get_transform()
                tf.location.z -= 30.0
                a.set_transform(tf)
            except Exception:
                pass
        _flush(0.03, 2)

    # Batch destroy: controllers -> pedestrians -> vehicles
    def _batch_destroy(ids):
        if not ids:
            return []
        res = client.apply_batch_sync([carla.command.DestroyActor(i) for i in ids], True)
        return [ids[i] for i, r in enumerate(res) if r.error]

    leftovers = []
    leftovers += _batch_destroy([c.id for c in controllers])
    _flush(0.02, 1)
    leftovers += _batch_destroy([w.id for w in walkers])
    _flush(0.02, 1)
    leftovers += _batch_destroy([v.id for v in vehs])
    _flush(0.03, 2)

    # Individual fallback destruction
    for aid in list(leftovers):
        try:
            a = world.get_actor(aid)
            if a:
                a.destroy()
        except Exception:
            pass
    _flush(0.03, 2)

    # Summarize remaining actors
    actors2 = world.get_actors()
    rem_veh = sum(1 for a in actors2 if a.type_id.startswith('vehicle.') and a.id not in keep)
    rem_walk = sum(1 for a in actors2 if a.type_id.startswith('walker.pedestrian') and a.id not in keep)
    return rem_veh, rem_walk



APOLLO_CLEAR_URL = "http://127.0.0.1:9002/clear"  # If the container is not on the host network, change to the corresponding IP:PORT

def apollo_clear_prediction_planning(times=1, interval=0.0, timeout=2.0, verbose=True):
    url = APOLLO_CLEAR_URL
    payload = json.dumps({"times": times, "interval": interval}).encode("utf-8")
    req = urllib.request.Request(url, data=payload,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
        if verbose:
            print(f"[APOLLO-CLEAR] ok -> {url}")
        return True
    except Exception as e:
        if verbose:
            print(f"[APOLLO-CLEAR] failed -> {url}: {e}")
        return False


def run_all_actions_for_npcs(
    vehicle_list: Sequence[Any],
    all_action_sequences: Sequence[List[str]],
    action_trans: Callable[[Any, str], None],
    *,
    delay_s: float = 0.0,    # optional wait between each action (or use synchronous simulation's world.tick() instead)
) -> None:
    if len(vehicle_list) != len(all_action_sequences):
        raise ValueError("vehicle_list and all_action_sequences have different lengths, cannot match one-to-one.")

    for veh, seq in zip(vehicle_list, all_action_sequences):
        # For “the current vehicle”, execute its entire action sequence
        for action in seq:
            action_trans(veh, action)   # first argument is the vehicle, second is the action (string)
            if delay_s > 0:
                time.sleep(delay_s)
def action_trans(vehicle, action):
    """
    Example function that executes the corresponding operation on a vehicle based on the action name.
    """
    if action == 'break':
        vehicle.decelerate()
    elif action == 'accelerate':
        vehicle.accelerate()
    elif action == 'right_change_acc':
        vehicle.request_lane_change_accel('right')
    elif action == 'right_change_dec':
        vehicle.request_lane_change_decel('right')
    elif action == 'left_change_acc':
        vehicle.request_lane_change_accel('left')
    elif action == 'left_change_dec':
        vehicle.request_lane_change_decel('left')


def map_size(world):
    # Load .xodr file
    xodr_path = "Town01.xodr"
    tree = ET.parse(xodr_path)
    root = tree.getroot()

    # Initialize bounds
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')

    # Iterate over geometry information
    for road in root.findall('road'):
        for geometry in road.find('planView').findall('geometry'):
            x = float(geometry.get('x'))
            y = float(geometry.get('y'))
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
    return min_x, max_x , min_y, max_y

def append_jsonl(path: str, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
def _to_jsonable(x):
    try:
        import carla as _carla
    except Exception:
        _carla = None
    if _carla and isinstance(x, _carla.Transform):
        return {"location": _to_jsonable(x.location), "rotation": _to_jsonable(x.rotation)}
    if _carla and isinstance(x, _carla.Location):
        return {"x": float(x.x), "y": float(x.y), "z": float(x.z)}
    if _carla and isinstance(x, _carla.Rotation):
        return {"pitch": float(x.pitch), "yaw": float(x.yaw), "roll": float(x.roll)}
    if _carla and isinstance(x, _carla.Vector3D):
        return {"x": float(x.x), "y": float(x.y), "z": float(x.z)}
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if hasattr(x, "item") and callable(getattr(x, "item", None)):
        try: return x.item()
        except Exception: pass
    return x

def get_xy_speed(vehicle):
    """
    Get the X and Y speed components of a vehicle in m/s.
    """
    velocity = vehicle.get_velocity()
    x_speed = velocity.x  # Forward/Backward speed (m/s)
    y_speed = velocity.y  # Lateral speed (m/s)
    return x_speed, y_speed

def position_scaler(position, x_min, x_max, y_min, y_max):
    # Example values for min and max positions

    # Original positions
    npc_position_x, npc_position_y = position

    # Scaling to [0, 2]
    x_range = x_max - x_min
    y_range = y_max - y_min
    scaled_x = ((npc_position_x - x_min) / x_range) * 2
    # print(min_x)
    scaled_y = ((npc_position_y - y_min) / y_range) * 2

    return scaled_x, scaled_y

def state_encoder(ego_vehicle, vehicles, ego_vehicle_position, agent_positions, max_x_diff, max_y_diff):
    state = []
    # Compute speeds of all vehicles to avoid redundant calls
    vehicle_speeds = [get_xy_speed(vehicle) for vehicle in vehicles]
    ego_vel_x, ego_vel_y = get_xy_speed(ego_vehicle)

    for i, agent_position in enumerate(agent_positions):
        vel_x, vel_y = vehicle_speeds[i]

        # Compute positions and velocities of other agents
        other_agents = [pos for j, pos in enumerate(agent_positions) if j != i]
        other_vels = [vehicle_speeds[j] for j in range(len(vehicles)) if j != i]

        # Ensure at least two agents (otherwise pad with default values)
        while len(other_agents) < 2:
            other_agents.append((0, 0))
            other_vels.append((0, 0))

        agent_state = (
            agent_position[0] / max_x_diff,
            agent_position[1] / max_y_diff,
            vel_x,
            vel_y,
            other_agents[0][0] / max_x_diff,
            other_agents[0][1] / max_y_diff,
            other_vels[0][0],
            other_vels[0][1],
            other_agents[1][0] / max_x_diff,
            other_agents[1][1] / max_y_diff,
            other_vels[1][0],
            other_vels[1][1],
            ego_vehicle_position[0] / max_x_diff,
            ego_vehicle_position[1] / max_y_diff,
            ego_vel_x,
            ego_vel_y
        )
        state.append(agent_state)

    return state


def has_passed_destination(vehicle,
                          destination_location,
                          world_map,
                          near_dist_m: float = 6.0,
                          lat_band_m: float = 4.0,
                          pass_ahead_m: float = 2.0,
                          require_same_lane: bool = True,
                          near_speed_mps: float = None):
    “””
    near = straight-line distance to destination <= near_dist_m
           (optionally, if near_speed_mps is not None, also requires speed <= that threshold)
    passed = using the “road tangent at the destination waypoint” as reference, the vehicle is “ahead” of the destination (s >= pass_ahead_m)
             and lateral offset |d| <= lat_band_m
             (optionally: requires same road/lane as destination)
    “””
    import math
    import carla

    # --- Vehicle and destination positions ---
    veh_tf = vehicle.get_transform()
    veh_loc = veh_tf.location
    dx = veh_loc.x - destination_location.x
    dy = veh_loc.y - destination_location.y
    dist = math.hypot(dx, dy)

    # near determination
    near = dist <= near_dist_m
    if near and near_speed_mps is not None:
        vel = vehicle.get_velocity()
        speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
        near = near and (speed <= near_speed_mps)

    # --- Project using the “road tangent/normal at the destination” ---
    wp_dest = world_map.get_waypoint(destination_location,
                                     project_to_road=True,
                                     lane_type=carla.LaneType.Driving)
    if wp_dest is None:
        # Fallback: could not get destination waypoint, can only return near, set passed to False
        return near, False

    T = wp_dest.transform
    t_hat = T.get_forward_vector()   # road tangent (unit vector)
    n_hat = T.get_right_vector()     # road normal (unit vector)
    r = carla.Vector3D(veh_loc.x - T.location.x,
                       veh_loc.y - T.location.y,
                       (veh_loc.z - T.location.z) if hasattr(veh_loc, "z") else 0.0)

    # Longitudinal / lateral projection
    s = r.x * t_hat.x + r.y * t_hat.y + r.z * t_hat.z
    d = r.x * n_hat.x + r.y * n_hat.y + r.z * n_hat.z

    same_lane_ok = True
    if require_same_lane:
        wp_veh = world_map.get_waypoint(veh_loc,
                                        project_to_road=True,
                                        lane_type=carla.LaneType.Driving)
        same_lane_ok = (wp_veh is not None and
                        wp_veh.road_id == wp_dest.road_id and
                        wp_veh.lane_id == wp_dest.lane_id)

    passed = (s >= pass_ahead_m) and (abs(d) <= lat_band_m) and same_lane_ok
    return near, passed



import numpy as np
from scipy.spatial.distance import euclidean
import random

import math

def _extract_xy_list(pop: dict):
    """
    Extract [(x,y), (x,y), ...] from the population's position_info (or directly from a position dict),
    compatible with both surrounding_info and surrounding_transforms formats.
    """
    # pop may be {"position_info": {...}} or directly {...}
    pi = pop.get("position_info", pop)

    # Get ego coordinates
    ego_tf = pi["ego_transform"]
    ego_xy = (ego_tf.location.x, ego_tf.location.y)

    # Get surrounding coordinate list (compatible with both field names)
    if "surrounding_info" in pi:
        surr = pi["surrounding_info"]
        surr_xy = [(item["transform"].location.x, item["transform"].location.y) for item in surr]
    elif "surrounding_transforms" in pi:
        surr = pi["surrounding_transforms"]
        surr_xy = [(t.location.x, t.location.y) for t in surr]
    else:
        surr_xy = []

    # Vehicle count (if vehicle_num field is absent, use list length)
    vnum = pi.get("vehicle_num", len(surr_xy))

    return ego_xy, surr_xy, vnum

def _euclid2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def calculate_population_distance(pop1, pop2, alpha=1.0):
    “””
    Compute the “distance” between two populations (or their position_info):
      - EGO position difference
      - Average surrounding position difference (zipped to the minimum common count)
      - Vehicle count difference (weighted by alpha)
    Compatible with surrounding_info / surrounding_transforms.
    “””
    ego1, surr1, n1 = _extract_xy_list(pop1)
    ego2, surr2, n2 = _extract_xy_list(pop2)

    # EGO distance
    ego_dist = _euclid2(ego1, ego2)

    # Surrounding average distance (zipped in order; if order-independent is needed, use Hungarian matching)
    m = min(len(surr1), len(surr2))
    if m > 0:
        surr_dists = [_euclid2(a, b) for a, b in zip(surr1[:m], surr2[:m])]
        surr_mean = sum(surr_dists) / m
    else:
        surr_mean = 0.0

    # Relative vehicle count difference
    denom = max(max(n1, n2), 1)
    vnum_term = alpha * (abs(n1 - n2) / denom)

    # Final distance: position component uses the mean of (ego + surrounding_mean)
    pos_term = (ego_dist + surr_mean) / 2.0
    return pos_term + vnum_term



def average_population_distance(population, generation):
    """
    Compute the average distance between a population and an entire generation.
    """
    distances = [calculate_population_distance(population["position_info"], pop["position_info"]) for pop in generation]
    return np.mean(distances)

def min_population_distance(population, generation):
    """
    Compute the minimum distance between a population and an entire generation.
    """
    distances = [calculate_population_distance(population["position_info"], pop["position_info"]) for pop in generation]
    return min(distances)

def max_population_distance(population, generation):
    """
    Compute the maximum distance between a population and an entire generation.
    """
    distances = [calculate_population_distance(population["position_info"], pop["position_info"]) for pop in generation]
    return max(distances)

def parents_selection(fitness, population, population_size):
    processed_fitness = []
    for i in range(population_size):
        processed_fitness.append(
            [fitness["safety_violation"][i], fitness["diversity"][i], fitness["ART_trigger_time"][i]])
    rank, weight = non_dominated_sorting_initial(processed_fitness)
    parents = random.choices(population, weights=weight, k=population_size)
    return parents

def next_gen_selection(fitness, population, population_size):
    processed_fitness = []
    for i in range(population_size):
        processed_fitness.append(
            [fitness["safety_violation"][i+10], fitness["diversity"][i+10], fitness["ART_trigger_time"][i+10]])
    sorted_population = non_dominated_sorting_with_weights(processed_fitness, population)
    population = sorted_population[0:population_size]
    return population

def non_dominated_sorting_initial(solutions: List[List[float]]) -> (List[Set[int]], List[float]):
    def dominates(sol1: List[float], sol2: List[float]) -> bool:
        """Check if sol1 dominates sol2."""
        return all(x <= y for x, y in zip(sol1, sol2)) and any(x < y for x, y in zip(sol1, sol2))

    # Non-dominated sorting
    n = len(solutions)
    dominated_by = [set() for _ in range(n)]
    dominates_count = [0 for _ in range(n)]
    fronts = [[]]

    for i in range(n):
        for j in range(n):
            if i != j:
                if dominates(solutions[i], solutions[j]):
                    dominated_by[i].add(j)
                elif dominates(solutions[j], solutions[i]):
                    dominates_count[i] += 1
        if dominates_count[i] == 0:
            fronts[0].append(i)

    i = 0
    while fronts[i]:
        next_front = []
        for sol in fronts[i]:
            for dominated_sol in dominated_by[sol]:
                dominates_count[dominated_sol] -= 1
                if dominates_count[dominated_sol] == 0:
                    next_front.append(dominated_sol)
        i += 1
        fronts.append(next_front)

    fronts = [set(front) for front in fronts if front]

    # Assigning weights
    weights = [0] * n
    for i, front in enumerate(fronts):
        weight = 1 / (i + 1)
        for sol in front:
            weights[sol] = weight

    return fronts, weights

def non_dominated_sorting_with_weights(solutions: List[List[float]],population) -> List[List[float]]:
    def dominates(sol1: List[float], sol2: List[float]) -> bool:
        """Check if sol1 dominates sol2."""
        return all(x <= y for x, y in zip(sol1, sol2)) and any(x < y for x, y in zip(sol1, sol2))

    def calculate_crowding_distance(front_solutions: List[List[float]]) -> List[float]:
        """Calculate the crowding distance of each solution in the front."""
        if not front_solutions:
            return []

        size = len(front_solutions)
        distances = [0.0 for _ in range(size)]
        for m in range(len(front_solutions[0])):
            sorted_indices = sorted(range(size), key=lambda x: front_solutions[x][m])
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            for i in range(1, size - 1):
                distances[sorted_indices[i]] += (
                            front_solutions[sorted_indices[i + 1]][m] - front_solutions[sorted_indices[i - 1]][m])

        return distances

    # Non-dominated sorting
    n = len(solutions)
    dominated_by = [set() for _ in range(n)]
    dominates_count = [0 for _ in range(n)]
    fronts = [[]]

    for i in range(n):
        for j in range(n):
            if i != j:
                if dominates(solutions[i], solutions[j]):
                    dominated_by[i].add(j)
                elif dominates(solutions[j], solutions[i]):
                    dominates_count[i] += 1
        if dominates_count[i] == 0:
            fronts[0].append(i)

    i = 0
    while fronts[i]:
        next_front = []
        for sol in fronts[i]:
            for dominated_sol in dominated_by[sol]:
                dominates_count[dominated_sol] -= 1
                if dominates_count[dominated_sol] == 0:
                    next_front.append(dominated_sol)
        i += 1
        fronts.append(next_front)

    fronts = [set(front) for front in fronts if front]

    # Sorting within each front based on crowding distance
    sorted_population_indices = []
    for front in fronts:
        front_list = list(front)
        front_solutions = [solutions[i] for i in front_list]
        crowding_distances = calculate_crowding_distance(front_solutions)
        # Sort the front based on crowding distance (descending order)
        sorted_front_indices = sorted(front_list, key=lambda i: crowding_distances[front_list.index(i)], reverse=True)
        sorted_population_indices.extend(sorted_front_indices)

    sorted_population = [population[i] for i in sorted_population_indices]

    return sorted_population