import requests
import carla
from ptop.core.carla_controller import LaneKeepAndChangeController
import random
from websocket import create_connection, WebSocketException
import json
import threading
import math
import logging
import time
import numpy as np
import collections
import os

from ptop.utils.geometry import ego_local_sd, vec_norm, unit_vec, dot, spd_and_vec

max_search_distance_for_destination = 200  # Maximum BFS search distance
step_dist_for_destination = 2.0            # BFS step distance (meters)
max_search_distance_for_spawns = 50.0      # Used for multi-lane spawn point selection
step_for_spawns = 1.0

log = logging.getLogger(__name__)


EGO_FAULT_CLOSE_SPEED_MIN = 0.8   # m/s, minimum EGO approach speed along the collision normal
EGO_FAULT_RATIO          = 0.60   # Ratio threshold of EGO approach speed to total approach speed
IMPULSE_MIN              = 400.0  # Collision impulse lower bound; too small can be considered a scrape/false positive
REAR_END_BONUS           = 0.05   # Slightly relaxed ratio threshold for rear-end scenarios

def fetch_localization_variable(url="http://127.0.0.1:5000/var"):
    """
    Fetch the latest localization data from the container via HTTP GET request.
    :param url: Flask endpoint address, defaults to localhost 127.0.0.1:5000/var
    :return: A dictionary of localization data in JSON format, or None on failure.
    """
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # Raises an exception if the response status code is not 200
        data = response.json()
        return data
    except Exception as e:
        log.error("Error occurred while fetching variable data: %s", e)
        return None

def distance(loc1, loc2):
    """Simple Euclidean distance."""
    return math.sqrt(
        (loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2
    )

# ========= Added: spawn minimum spacing protection =========
SPAWN_GAP_VEH = 4.5   # Minimum gap between vehicles (meters)
SPAWN_GAP_PED = 2.5   # Minimum gap between pedestrians (meters)
SPAWN_GAP_EGO = 7.0   # Minimum gap between NPC and EGO (meters)

def _gap_ok(tf, accepted_tfs, min_gap):
    """Check if tf's 2D distance to all Transforms in accepted_tfs is >= min_gap."""
    for t in accepted_tfs:
        dx = tf.location.x - t.location.x
        dy = tf.location.y - t.location.y
        if math.hypot(dx, dy) < float(min_gap):
            return False
    return True
# ========================================

class MultiVehicleDemo:
    """
    Main functions:
      1) Spawn ego + N autonomous vehicles (with LaneKeepAndChangeController)
      2) Attach collision sensors to all vehicles
         - When ego collides => determine "ego actively hit" or "someone hit ego"
         - When other vehicles collide => emergency braking, but don't end episode
      3) In tick(), return (signals_list, ego_collision, self.collision, ego_cross_solid_line, ego_run_red_light)
      4) Provide set_destination() => find the farthest same-direction waypoint via BFS, saved as self.ego_destination
      5) Provide get_controller(idx) => get the controller of the idx-th autonomous vehicle
    """

    def __init__(self, world, external_ads, websocket_url="ws://localhost:8888/websocket",
                 gps_offset=carla.Vector3D(x=1.0, y=0.0, z=0.5)):
        self.world = world
        self.population_size = 10
        self.map = world.get_map()
        self.ego_vehicle = None
        self.multi_vehicle_collision_count = 0
        self.vehicles = []            # Store autonomous vehicles
        self.controllers = None       # N LaneKeepAndChangeController instances
        self.url = websocket_url
        self.gps_offset = gps_offset
        self.ws = None
        self.vehicle_num = None
        self.ws_thread = None
        self.ws_running = False
        self.ws_receive_buffer = []
        self.ego_spawning_point = None
        self.ego_destination = None   # Set via set_destination
        self.collision = False
        self.external_ads = external_ads
        self.count = 0
        self.turn_on = False
        self.modules = [
            'Localization',
            'Routing',
            'Prediction',
            'Planning',
            'Control'
        ]
        self.side_collision_count_vehicle = 0  # Side collision count
        self.rear_collision_count_vehicle = 0  # Rear-end collision count
        self.collision_count_obj = 0

        # Flag indicating whether ego actively hit another actor
        self.ego_collision = False

        # Map bounds
        self.map_bounds = self._compute_map_bounds()

        # Collision sensor list
        self.collision_sensors = []

        # ----- LaneInvasion related -----
        self.ego_cross_solid_line = False  # Whether EGO crossed a solid line
        self.lane_invasion_sensor_ego = None

        # ----- Red light violation detection related -----
        self.ego_run_red_light = False  # Whether EGO red light violation was detected

        if self.external_ads:
            self._connect_websocket()

    # ========== Basic Functions ==========
    # ---- Added: determine if “NPC rear-ended EGO” ----
    def _is_npc_rear_end(self, ego: "carla.Vehicle", npc: "carla.Vehicle") -> bool:
        """
        Returns True if and only if: the other vehicle is behind EGO, traveling in the same direction
        with a higher forward speed (approaching), and the lateral offset is small (roughly same lane).
        Thresholds can be adjusted as needed.
        """
        try:
            ego_tf = ego.get_transform()
            npc_tf = npc.get_transform()

            # Relative longitudinal/lateral in ego coordinate frame
            s_rel, d_rel = ego_local_sd(ego_tf, npc_tf.location)

            # Ego forward unit vector
            yaw = math.radians(ego_tf.rotation.yaw)
            fwd = carla.Vector3D(math.cos(yaw), math.sin(yaw), 0.0)

            v_e = ego.get_velocity()
            v_n = npc.get_velocity()
            v_e_f = v_e.x * fwd.x + v_e.y * fwd.y + v_e.z * fwd.z
            v_n_f = v_n.x * fwd.x + v_n.y * fwd.y + v_n.z * fwd.z
            dv_f  = v_n_f - v_e_f  # NPC's forward speed difference relative to EGO (>0 means approaching from behind)

            # Heading is close (same direction), and lateral offset is small (roughly same lane)
            dyaw = abs(((npc_tf.rotation.yaw - ego_tf.rotation.yaw + 180.0) % 360.0) - 180.0)
            lane_w = 3.5
            try:
                wp = self.map.get_waypoint(ego_tf.location)
                if wp and hasattr(wp, "lane_width"):
                    lane_w = float(wp.lane_width)
            except Exception:
                pass

            return (s_rel < -0.5) and (abs(d_rel) <= 0.4 * lane_w) and (dyaw <= 35.0) and (dv_f > 0.5)
        except Exception:
            return False

    def _connect_websocket(self):
        try:
            self.ws = create_connection(self.url)
            self.ws_running = True
            log.info("Connected to WebSocket server: %s", self.url)
            # Start a thread to receive messages
            self.ws_thread = threading.Thread(target=self._receive_messages, daemon=True)
            self.ws_thread.start()
        except WebSocketException as e:
            log.error("Unable to connect to WebSocket server: %s", e)
            self.ws = None

    def _receive_messages(self):
        while self.ws_running:
            try:
                result = self.ws.recv()
                if result:
                    self.ws_receive_buffer.append(result)
            except WebSocketException as e:
                log.error("Error while receiving WebSocket message: %s", e)
                self.ws_running = False
            except Exception as e:
                log.error("Unknown error: %s", e)
                self.ws_running = False

    def _compute_map_bounds(self):
        """
        Simply obtain the map x,y range via map.generate_waypoints(2.0).
        """
        wps = self.map.generate_waypoints(2.0)
        if not wps:
            log.warning("generate_waypoints is empty, no map data?")
            return (0, 0, 0, 0)

        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        for wp in wps:
            loc = wp.transform.location
            if loc.x < min_x: min_x = loc.x
            if loc.x > max_x: max_x = loc.x
            if loc.y < min_y: min_y = loc.y
            if loc.y > max_y: max_y = loc.y
        log.info("Map x range=(%.1f,%.1f), y range=(%.1f,%.1f)", min_x, max_x, min_y, max_y)
        return (min_x, max_x, min_y, max_y)

    def get_map_bounds(self):
        return self.map_bounds

    # ========== Vehicle Spawning Logic ==========

    def setup_vehicles(self, scenario_conf):
        “””
        1) Spawn EGO
        2) Spawn NPCs (cars/bicycles/pedestrians) in order of scenario_conf['surrounding_info']
           - If an NPC position conflicts: resample in place (shift along lane forward/backward & laterally) until success
           - If pedestrian fails: repeatedly resample from the navigation mesh until success
        3) After success, write back the “final successful position” to scenario_conf to keep the scenario representation consistent with reality
        “””
        world = self.world
        world_map = world.get_map()
        blueprint_library = world.get_blueprint_library()

        self.vehicle_num = int(scenario_conf["vehicle_num"])
        self.controllers = [None] * self.vehicle_num

        # ------- surrounding_info parsing -------
        surrounding = scenario_conf["surrounding_info"]

        # Two representations: list[{"transform","type"}] or dict{"transform":[...], "type":[...]}
        def _get_item(i):
            if isinstance(surrounding, list):
                return surrounding[i]["transform"], str(surrounding[i]["type"]).lower()
            else:
                return surrounding["transform"][i], str(surrounding["type"][i]).lower()

        def _set_item_transform(i, new_tf):
            if isinstance(surrounding, list):
                surrounding[i]["transform"] = new_tf
            else:
                surrounding["transform"][i] = new_tf

        n_to_spawn = min(self.vehicle_num,
                         len(surrounding) if isinstance(surrounding, list) else len(surrounding["transform"]))

        # ------- EGO -------
        self.ego_spawning_point = scenario_conf["ego_transform"]
        self.ego_vehicle = None
        if not getattr(self, "external_ads", False):
            try:
                bp_ego = blueprint_library.find("vehicle.tesla.model3")
            except Exception:
                veh_bps = blueprint_library.filter("vehicle.*")
                four_wheels = [bp for bp in veh_bps if bp.has_attribute("number_of_wheels")
                               and int(bp.get_attribute("number_of_wheels").as_int()) == 4]
                bp_ego = random.choice(four_wheels if four_wheels else veh_bps)
            if bp_ego.has_attribute("color"):
                bp_ego.set_attribute("color", "0,0,255")
            self.ego_vehicle = world.try_spawn_actor(bp_ego, self.ego_spawning_point)
        else:
            # External ADS: find the existing mkz_2017 and move it to ego_transform
            all_actors = world.get_actors()
            candidate_vehicles = all_actors.filter("vehicle.*")
            for v in candidate_vehicles:
                if "mkz_2017" in v.type_id:
                    self.ego_vehicle = v
                    break
            if not self.ego_vehicle:
                log.error("Could not find 'mkz_2017' as EGO.")
                return False
            self.ego_vehicle.set_transform(self.ego_spawning_point)

        if not self.ego_vehicle:
            log.error("EGO vehicle spawn failed.")
            return False

        # ------- Blueprint pools -------
        veh_bps_all = blueprint_library.filter("vehicle.*")
        car_bps = blueprint_library.filter("vehicle.tesla.model3") or veh_bps_all
        bike_bps = [bp for bp in veh_bps_all if ("bicycle" in bp.id.lower() or "bike" in bp.id.lower())]
        walker_bps = blueprint_library.filter("walker.pedestrian.*")

        def _pick(pool, fallback):
            if pool: return random.choice(pool)
            if fallback: return random.choice(fallback)
            return random.choice(veh_bps_all)

        # ------- Geometry / lane helper -------
        def _project_to_lane(tf, clip_ratio=0.45):
            wp = world_map.get_waypoint(tf.location, project_to_road=True,
                                        lane_type=carla.LaneType.Driving)
            if not wp:
                return tf, None
            lane_w = float(getattr(wp, "lane_width", 3.5))
            # Keep relative lateral offset, but clip to 0.45*lane_width
            center = wp.transform.location
            right = wp.transform.get_right_vector()
            dv = carla.Vector3D(tf.location.x - center.x, tf.location.y - center.y, tf.location.z - center.z)
            dlat = dv.x * right.x + dv.y * right.y + dv.z * right.z
            d_clip = float(np.clip(dlat, -clip_ratio * lane_w, clip_ratio * lane_w))
            loc = carla.Location(center.x + d_clip * right.x,
                                 center.y + d_clip * right.y,
                                 tf.location.z)
            # Aligning yaw with lane direction is more stable
            yaw = wp.transform.rotation.yaw
            return carla.Transform(loc, carla.Rotation(pitch=tf.rotation.pitch, yaw=yaw, roll=tf.rotation.roll)), lane_w

        def _lane_shift_candidates(tf0, max_forward=18.0, step_s=2.0, step_d=0.75, d_mul=0.45):
            """Sample candidate poses along lane centerline +/-s forward/backward, then +/-d laterally (closer ones prioritized)."""
            base_wp = world_map.get_waypoint(tf0.location, project_to_road=True,
                                             lane_type=carla.LaneType.Driving)
            if not base_wp:
                return [tf0]

            # Generate s offset sequence (0, +2, -2, +4, -4, ...)
            s_vals = [0.0]
            k = int(max_forward // step_s)
            for i in range(1, k + 1):
                s_vals += [i * step_s, -i * step_s]

            # Generate d offset sequence (0, +0.75, -0.75, +1.5, -1.5, ...)
            lane_w = float(getattr(base_wp, "lane_width", 3.5))
            d_max = d_mul * lane_w
            d_vals = [0.0]
            kd = max(1, int(d_max // step_d))
            for i in range(1, kd + 1):
                d_vals += [i * step_d, -i * step_d]

            cands = []
            for s in s_vals:
                # ---- Key fix: never call next/previous(0.0) ----
                if s > 0.0:
                    wps = base_wp.next(s)
                elif s < 0.0:
                    wps = base_wp.previous(-s)
                else:
                    wps = [base_wp]  # s == 0, use current waypoint directly

                if not wps:
                    continue
                wp = wps[0]
                center = wp.transform.location
                right = wp.transform.get_right_vector()
                yaw = wp.transform.rotation.yaw

                for d in d_vals:
                    loc = carla.Location(center.x + d * right.x,
                                         center.y + d * right.y,
                                         tf0.location.z)
                    cands.append(carla.Transform(
                        loc,
                        carla.Rotation(pitch=tf0.rotation.pitch, yaw=yaw, roll=tf0.rotation.roll)
                    ))
            return cands

        def _tick_flush():
            try:
                world.tick()
            except Exception:
                world.wait_for_tick()
            time.sleep(0.01)

        # ------- Containers -------
        if not hasattr(self, "vehicles"): self.vehicles = []
        if not hasattr(self, "pedestrians"): self.pedestrians = []

        # ------- Spawn NPCs one by one (resample on failure until success) -------
        spawned_vehicle_count = 0
        spawned_ped_count = 0

        log.info("vehicle number (requested): %s", self.vehicle_num)

        # Added: record successfully placed Transforms (separately for vehicles/pedestrians)
        veh_tfs = []
        ped_tfs = []

        for i in range(n_to_spawn):
            init_tf, npc_type = _get_item(i)
            actor = None

            try:
                if npc_type == "pedestrian":
                    if not walker_bps:
                        log.warning("No pedestrian blueprints, NPC[%d] skipped.", i)
                        continue
                    bp = random.choice(walker_bps)

                    # Try original position first (if minimum spacing is satisfied)
                    if self.ego_vehicle:
                        ego_loc_now = self.ego_vehicle.get_transform().location
                        if math.hypot(init_tf.location.x - ego_loc_now.x,
                                      init_tf.location.y - ego_loc_now.y) < SPAWN_GAP_EGO:
                            actor = None
                        elif not _gap_ok(init_tf, ped_tfs, SPAWN_GAP_PED):
                            actor = None
                        else:
                            actor = world.try_spawn_actor(bp, init_tf)
                    else:
                        # Rare case where ego has not spawned yet; fallback to only check against placed pedestrians
                        if _gap_ok(init_tf, ped_tfs, SPAWN_GAP_PED):
                            actor = world.try_spawn_actor(bp, init_tf)

                    if not actor:
                        # Resample from navigation mesh until success
                        attempts = 0
                        while actor is None:
                            loc = world.get_random_location_from_navigation()
                            if loc is None:
                                attempts += 1
                                if attempts % 10 == 0: _tick_flush()
                                continue
                            tf_try = carla.Transform(loc, init_tf.rotation)

                            # Distance constraint with EGO / already-placed pedestrians
                            ok_ego = True
                            if self.ego_vehicle:
                                ego_loc_now = self.ego_vehicle.get_transform().location
                                if math.hypot(tf_try.location.x - ego_loc_now.x,
                                              tf_try.location.y - ego_loc_now.y) < SPAWN_GAP_EGO:
                                    ok_ego = False
                            if ok_ego and _gap_ok(tf_try, ped_tfs, SPAWN_GAP_PED):
                                actor = world.try_spawn_actor(bp, tf_try)
                                attempts += 1
                                if actor:
                                    _set_item_transform(i, tf_try)
                                    ped_tfs.append(tf_try)
                                    break
                            else:
                                attempts += 1

                            if attempts % 10 == 0:
                                _tick_flush()
                    else:
                        # Original position succeeded: write back as well (for consistency)
                        _set_item_transform(i, init_tf)
                        ped_tfs.append(init_tf)

                    if actor:
                        self.pedestrians.append(actor)
                        spawned_ped_count += 1
                    else:
                        # Fallback: keep retrying (preserving original logic, but with spacing constraint added)
                        while actor is None:
                            loc = world.get_random_location_from_navigation()
                            if loc:
                                tf_try = carla.Transform(loc, init_tf.rotation)
                                ok_ego = True
                                if self.ego_vehicle:
                                    ego_loc_now = self.ego_vehicle.get_transform().location
                                    if math.hypot(tf_try.location.x - ego_loc_now.x,
                                                  tf_try.location.y - ego_loc_now.y) < SPAWN_GAP_EGO:
                                        ok_ego = False
                                if ok_ego and _gap_ok(tf_try, ped_tfs, SPAWN_GAP_PED):
                                    actor = world.try_spawn_actor(bp, tf_try)
                                    if actor:
                                        _set_item_transform(i, tf_try)
                                        self.pedestrians.append(actor)
                                        ped_tfs.append(tf_try)
                                        spawned_ped_count += 1
                                        break
                            _tick_flush()

                else:
                    # Vehicle / bicycle (fallback to four-wheel car), resample near lane until success
                    if npc_type == "bicycle":
                        bp = _pick(bike_bps, car_bps)
                    elif npc_type == "car":
                        bp = _pick(car_bps, veh_bps_all)
                    else:
                        bp = _pick(car_bps, veh_bps_all)

                    # First project/clip init_tf to the lane
                    tf0, _ = _project_to_lane(init_tf)

                    # Try the projected original position first (if minimum spacing is satisfied)
                    can_try = True
                    if self.ego_vehicle:
                        ego_loc_now = self.ego_vehicle.get_transform().location
                        if math.hypot(tf0.location.x - ego_loc_now.x,
                                      tf0.location.y - ego_loc_now.y) < SPAWN_GAP_EGO:
                            can_try = False
                    if can_try and not _gap_ok(tf0, veh_tfs, SPAWN_GAP_VEH):
                        can_try = False
                    actor = world.try_spawn_actor(bp, tf0) if can_try else None

                    if actor:
                        _set_item_transform(i, tf0)
                        veh_tfs.append(tf0)
                    else:
                        # Generate candidates along lane and keep trying; if a round fails, expand range and resample
                        attempts = 0
                        max_forward = 18.0
                        while actor is None:
                            candidates = _lane_shift_candidates(tf0, max_forward=max_forward,
                                                                step_s=2.0, step_d=0.75, d_mul=0.45)
                            for tf_try in candidates:
                                ok_ego = True
                                if self.ego_vehicle:
                                    ego_loc_now = self.ego_vehicle.get_transform().location
                                    if math.hypot(tf_try.location.x - ego_loc_now.x,
                                                  tf_try.location.y - ego_loc_now.y) < SPAWN_GAP_EGO:
                                        ok_ego = False
                                if ok_ego and _gap_ok(tf_try, veh_tfs, SPAWN_GAP_VEH):
                                    actor = world.try_spawn_actor(bp, tf_try)
                                    attempts += 1
                                    if actor:
                                        _set_item_transform(i, tf_try)
                                        veh_tfs.append(tf_try)
                                        break
                                else:
                                    attempts += 1
                                if attempts % 15 == 0:
                                    _tick_flush()
                            if actor:
                                break
                            # Expand search range and try another round
                            max_forward = min(max_forward + 12.0, 60.0)
                            if attempts > 200:
                                # Fallback: randomly pick from global spawn_points until success
                                sps = world_map.get_spawn_points()
                                random.shuffle(sps)
                                for tf_try in sps:
                                    ok_ego = True
                                    if self.ego_vehicle:
                                        ego_loc_now = self.ego_vehicle.get_transform().location
                                        if math.hypot(tf_try.location.x - ego_loc_now.x,
                                                      tf_try.location.y - ego_loc_now.y) < SPAWN_GAP_EGO:
                                            ok_ego = False
                                    if ok_ego and _gap_ok(tf_try, veh_tfs, SPAWN_GAP_VEH):
                                        actor = world.try_spawn_actor(bp, tf_try)
                                        attempts += 1
                                        if actor:
                                            _set_item_transform(i, tf_try)
                                            veh_tfs.append(tf_try)
                                            break
                                    else:
                                        attempts += 1
                                    if attempts % 15 == 0:
                                        _tick_flush()
                            if attempts > 400 and actor is None:
                                # Keep retrying (until success), tick every 30 attempts
                                tf_try = random.choice(world_map.get_spawn_points())
                                ok_ego = True
                                if self.ego_vehicle:
                                    ego_loc_now = self.ego_vehicle.get_transform().location
                                    if math.hypot(tf_try.location.x - ego_loc_now.x,
                                                  tf_try.location.y - ego_loc_now.y) < SPAWN_GAP_EGO:
                                        ok_ego = False
                                if ok_ego and _gap_ok(tf_try, veh_tfs, SPAWN_GAP_VEH):
                                    actor = world.try_spawn_actor(bp, tf_try)
                                    if actor:
                                        _set_item_transform(i, tf_try)
                                        veh_tfs.append(tf_try)
                                        break
                                _tick_flush()

                    if actor:
                        self.vehicles.append(actor)
                        spawned_vehicle_count += 1

            except Exception as e:
                log.error(“Error spawning NPC[%d]: %s”, i, e)
                # Force fallback logic that retries until success (vehicle as example)
                if npc_type == "pedestrian" and walker_bps:
                    bp = random.choice(walker_bps)
                    while True:
                        loc = world.get_random_location_from_navigation()
                        if loc:
                            tf_try = carla.Transform(loc, init_tf.rotation)
                            ok_ego = True
                            if self.ego_vehicle:
                                ego_loc_now = self.ego_vehicle.get_transform().location
                                if math.hypot(tf_try.location.x - ego_loc_now.x,
                                              tf_try.location.y - ego_loc_now.y) < SPAWN_GAP_EGO:
                                    ok_ego = False
                            if ok_ego and _gap_ok(tf_try, ped_tfs, SPAWN_GAP_PED):
                                a2 = world.try_spawn_actor(bp, tf_try)
                                if a2:
                                    _set_item_transform(i, tf_try)
                                    self.pedestrians.append(a2)
                                    ped_tfs.append(tf_try)
                                    spawned_ped_count += 1
                                    break
                        _tick_flush()
                else:
                    bp = _pick(car_bps, veh_bps_all)
                    tf0, _ = _project_to_lane(init_tf)
                    while True:
                        # Global random spawn_point
                        tf_try = random.choice(world_map.get_spawn_points())
                        ok_ego = True
                        if self.ego_vehicle:
                            ego_loc_now = self.ego_vehicle.get_transform().location
                            if math.hypot(tf_try.location.x - ego_loc_now.x,
                                          tf_try.location.y - ego_loc_now.y) < SPAWN_GAP_EGO:
                                ok_ego = False
                        if ok_ego and _gap_ok(tf_try, veh_tfs, SPAWN_GAP_VEH):
                            a2 = world.try_spawn_actor(bp, tf_try)
                            if a2:
                                _set_item_transform(i, tf_try)
                                self.vehicles.append(a2)
                                veh_tfs.append(tf_try)
                                spawned_vehicle_count += 1
                                break
                        _tick_flush()

        log.info("spawned vehicles (vehicle.*): %d", spawned_vehicle_count)
        log.info("spawned pedestrians: %d", spawned_ped_count)

        # Attach controllers to spawned “vehicles” (excluding pedestrians)

        for i, v in enumerate(self.vehicles):
            try:
                self.controllers[i] = LaneKeepAndChangeController(v)
            except Exception as e:
                log.warning("Controller creation failed veh[%d] id=%s: %s", i, v.id, e)

        return True

    def _is_valid_side_lane(self, wp, side_wp):
        """
        Check if the left/right lane is Driving type and has the same direction (same sign) as the current lane_id.
        """
        if not side_wp:
            return False
        if side_wp.lane_type != carla.LaneType.Driving:
            return False
        if wp.lane_id * side_wp.lane_id <= 0:
            return False
        return True

    def setup_vehicles_with_collision(self, scenario_conf):
        """
        Public interface:
        1) First call setup_vehicles
        2) If successful => _setup_collision_sensors
        """
        success = self.setup_vehicles(scenario_conf)
        if success:
            self._setup_collision_sensors()
        return success

    # ========== Collision Sensor Logic + LaneInvasion Sensor Logic ==========

    def _setup_collision_sensors(self):
        """
        Add collision sensors to ego vehicle + all autonomous vehicles.
        """
        blueprint_library = self.world.get_blueprint_library()
        collision_bp = blueprint_library.find('sensor.other.collision')

        # Ego collision sensor
        if self.ego_vehicle:
            collision_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
            sensor_ego = self.world.spawn_actor(collision_bp, collision_transform, attach_to=self.ego_vehicle)
            ### Added/modified: include this sensor reference in the callback
            sensor_ego.listen(lambda event, v=self.ego_vehicle, s=sensor_ego: self.collision_callback(event, v, s))
            self.collision_sensors.append(sensor_ego)
            log.info("Ego Vehicle %s collision sensor attached: %s", self.ego_vehicle.id, sensor_ego.id)

            # ----- Attach LaneInvasionSensor to ego vehicle -----
            lane_invasion_bp = blueprint_library.find('sensor.other.lane_invasion')
            lane_invasion_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.0))
            self.lane_invasion_sensor_ego = self.world.spawn_actor(
                lane_invasion_bp,
                lane_invasion_transform,
                attach_to=self.ego_vehicle
            )
            self.lane_invasion_sensor_ego.listen(self.lane_invasion_callback)
            log.info("Ego Vehicle %s lane invasion sensor attached: %s", self.ego_vehicle.id, self.lane_invasion_sensor_ego.id)

        # Other autonomous vehicles collision sensors
        for veh in self.vehicles:
            collision_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
            sensor = self.world.spawn_actor(collision_bp, collision_transform, attach_to=veh)
            ### Added/modified: include this sensor reference in the callback
            sensor.listen(lambda event, v=veh, s=sensor: self.collision_callback(event, v, s))
            self.collision_sensors.append(sensor)
            print(f"[INFO] Vehicle {veh.id} collision sensor attached: {sensor.id}")

    def lane_invasion_callback(self, event):
        """
        Triggered when the ego vehicle crosses lane markings. Checks if any solid LaneMarking is included.
        """
        for marking in event.crossed_lane_markings:
            if marking.type in [
                carla.LaneMarkingType.Solid,
                carla.LaneMarkingType.SolidSolid,
                carla.LaneMarkingType.SolidBroken,
                carla.LaneMarkingType.BrokenSolid
            ]:
                # Once a solid line type is detected => indicates crossing a solid line
                self.ego_cross_solid_line = True
                print("[INFO] EGO vehicle crossed a solid line!")
                break

    # ==================== Blame Attribution (this section is newly added/replaced) ====================

    # Thresholds (adjust as needed)
    EGO_FAULT_CLOSE_SPEED_MIN = 0.8   # m/s, minimum EGO approach speed along collision normal
    EGO_FAULT_RATIO          = 0.60   # Ratio threshold of EGO approach speed to total approach speed
    IMPULSE_MIN              = 400.0  # Collision impulse lower bound
    REAR_END_BONUS           = 0.05   # Relaxed ratio threshold for rear-end scenarios

    # ---- Helper: implemented as static methods for easy in-class invocation ----
    @staticmethod
    def _vec_norm(v: "carla.Vector3D") -> float:
        return math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)

    @staticmethod
    def _spd_and_vec(actor):
        v = actor.get_velocity()
        return MultiVehicleDemo._vec_norm(v), v

    @staticmethod
    def _unit_vec(a: "carla.Location", b: "carla.Location") -> "carla.Vector3D":
        dx, dy, dz = (b.x - a.x), (b.y - a.y), (b.z - a.z)
        n = math.sqrt(dx*dx + dy*dy + dz*dz) + 1e-9
        return carla.Vector3D(dx/n, dy/n, dz/n)

    @staticmethod
    def _dot(a: "carla.Vector3D", b: "carla.Vector3D") -> float:
        return a.x*b.x + a.y*b.y + a.z*b.z

    @staticmethod
    def _ego_local_sd(ego_tf: "carla.Transform", loc: "carla.Location"):
        yaw = math.radians(ego_tf.rotation.yaw)
        cy, sy = math.cos(yaw), math.sin(yaw)
        dx = loc.x - ego_tf.location.x
        dy = loc.y - ego_tf.location.y
        s =  dx * cy + dy * sy
        d = -dx * sy + dy * cy
        return s, d

    def _assign_blame_ego(self,
                          ego: "carla.Vehicle",
                          other: "carla.Actor",
                          world_map: "carla.Map",
                          event_normal_impulse: "carla.Vector3D"):
        “””
        Returns: (ego_fault: bool, why: str)
        Based on: comparing the ratio of “approach speed components” along the EGO->other direction + impulse and minimum speed thresholds.
        “””
        try:
            # Connecting unit vector (from EGO toward other)
            loc_e = ego.get_transform().location
            loc_o = other.get_transform().location
            n = self._unit_vec(loc_e, loc_o)

            # Speed components
            _, v_e = self._spd_and_vec(ego)
            if hasattr(other, "get_velocity"):
                _, v_o = self._spd_and_vec(other)
            else:
                v_o = carla.Vector3D(0.0, 0.0, 0.0)

            c_ego   = max(0.0, self._dot(v_e, n))     # EGO toward other
            c_other = max(0.0, -self._dot(v_o, n))    # Other toward EGO (equivalent to -v_o dot n)

            # Impulse magnitude
            J = self._vec_norm(event_normal_impulse)

            # Pose relationship (rear-end determination: other is directly ahead and EGO approach is greater)
            s_rel, _ = self._ego_local_sd(ego.get_transform(), loc_o)
            rear_end_like = (s_rel > 0.0 and c_ego > c_other)

            r = c_ego / (c_ego + c_other + 1e-9)
            thr = self.EGO_FAULT_RATIO - (self.REAR_END_BONUS if rear_end_like else 0.0)

            if J >= self.IMPULSE_MIN and c_ego >= self.EGO_FAULT_CLOSE_SPEED_MIN and r >= thr:
                reason = f"ego_fault: J={J:.1f}, c_ego={c_ego:.2f}, c_other={c_other:.2f}, r={r:.2f}, rear_end={rear_end_like}"
                return True, reason
            else:
                reason = f"non_ego_fault: J={J:.1f}, c_ego={c_ego:.2f}, c_other={c_other:.2f}, r={r:.2f}, rear_end={rear_end_like}"
                return False, reason
        except Exception as e:
            return False, f"non_ego_fault: exception {e}"

    def collision_callback(self, event, vehicle, sensor):
        """
        Only count collisions caused by EGO; otherwise only apply emergency braking without counting toward EGO collision metrics.
        """
        # If already counted, can return directly as needed (or continue processing for timer stop/destroy)
        # if self.collision_count_obj == 1 or self.multi_vehicle_collision_count == 1 or self.side_collision_count_vehicle == 1:
        #     return

        # other_actor = event.other_actor
        # is_vehicle = hasattr(other_actor, "type_id") and (str(other_actor.type_id).startswith("vehicle.") or str(other_actor.type_id).startswith("bicycle") or str(other_actor.type_id).startswith("bike"))

        if vehicle == self.ego_vehicle:
            # Only consider blame attribution when EGO is involved
            # ego_fault, why = _assign_blame_ego(self.ego_vehicle, other_actor, self.map, event.normal_impulse)
            self.collision = True
            self.ego_collision = True
            self.side_collision_count_vehicle = 1
            # if is_vehicle:
            #     if ego_fault:


                    #
                    # # Rough multi-vehicle or side/rear collision classification (can be refined)
                    # ego_transform = self.ego_vehicle.get_transform()
                    # ego_loc = ego_transform.location
                    # lane_width = self.map.get_waypoint(ego_loc).lane_width
                    # all_vehicles = self.world.get_actors().filter("vehicle.*")
                    #
                    # count_vehicles_in_lane = 0
                    # for v in all_vehicles:
                    #     if v.id == self.ego_vehicle.id:
                    #         continue
                    #     v_loc = v.get_transform().location
                    #     dist_2d = math.hypot(v_loc.x - ego_loc.x, v_loc.y - ego_loc.y)
                    #     if dist_2d < lane_width:
                    #         count_vehicles_in_lane += 1



                    # print(f"[INFO] EGO collided with vehicle (EGO at fault) | {why}")
                # else:
                    # Not EGO's fault: do not count toward EGO metrics
                    # print(f"[INFO] Vehicle collision, but not attributed to EGO | {why}")
            # else:
            #
            #     self.collision_count_obj = 1
            #     print(f"[INFO] EGO hit a stationary/non-vehicle object (EGO at fault) | {why}")

        else:
            # Non-EGO collision: only apply emergency braking (not counted toward EGO metrics)
            if vehicle in getattr(self, "vehicles", []):
                try:
                    idx = self.vehicles.index(vehicle)
                    controller = self.controllers[idx]
                    if controller:
                        controller.brake()
                except Exception:
                    pass

            # Apply braking directly
            try:
                cur = vehicle.get_control()
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=getattr(cur, "steer", 0.0)))
            except Exception:
                pass

        # Sensor one-shot: cleanup
        try:
            sensor.stop()
            sensor.destroy()
            print(f"[INFO] Collision sensor {sensor.id} destroyed (one-shot).")
        except Exception:
            pass

        if sensor in getattr(self, "collision_sensors", []):
            try:
                self.collision_sensors.remove(sensor)
            except Exception:
                pass

    # ========== Red Light Violation Detection Logic ==========

    def _detect_run_red_light(self):
        """
        Detect whether EGO ran a red light (with grace period + deceleration check).
        Same name and signature as the original function: no parameters, returns bool.
        Requires: self.ego_vehicle exists; carla is imported.
        Optional: if self.world exists, simulation time is used preferentially.
        """
        import math, time

        # ---------- Adjustable thresholds (can also be overridden externally by setting self._rl_*) ----------
        RED_STOP_WINDOW = getattr(self, "_rl_red_stop_window", 2.0)  # Red light grace period (seconds)
        STOP_SPEED_EPS = getattr(self, "_rl_stop_speed_eps", 0.2)  # Considered stopped (m/s)
        DECEL_DELTA_REQ = getattr(self, "_rl_decel_delta_req", 0.5)  # Minimum speed reduction before red-to-green (m/s)
        RECENT_GREEN_WINDOW = getattr(self, "_rl_recent_green_window", 3.0)  # Lookback window after red-to-green transition (seconds)

        def _now():
            # Prefer simulation time
            if hasattr(self, "world") and self.world is not None:
                try:
                    return self.world.get_snapshot().timestamp.elapsed_seconds
                except Exception:
                    pass
            return time.time()

        def _speed_of(vehicle):
            v = vehicle.get_velocity()
            return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

        def _reset_red_episode(_self):
            _self._rl_red_start_t = None
            _self._rl_v_at_red = None
            _self._rl_min_v_during_red = None

        # ---------- Initialize internal state ----------
        if not hasattr(self, "_rl_last_tl_state"):
            self._rl_last_tl_state = None
        if not hasattr(self, "_rl_red_start_t"):
            self._rl_red_start_t = None
            self._rl_v_at_red = None
            self._rl_min_v_during_red = None

        # ---------- Basic visibility check ----------
        if not getattr(self, "ego_vehicle", None):
            return False
        tlight = self.ego_vehicle.get_traffic_light()
        if tlight is None:
            # When no traffic light is visible, reset the episode once
            self._rl_last_tl_state = None
            _reset_red_episode(self)
            return False

        state = tlight.get_state()
        now = _now()
        speed = _speed_of(self.ego_vehicle)

        state_changed = (state != self._rl_last_tl_state)
        self._rl_last_tl_state = state

        # ================= Red Light Logic =================
        if state == carla.TrafficLightState.Red:
            if self._rl_red_start_t is None:
                # Just entered red light
                self._rl_red_start_t = now
                self._rl_v_at_red = speed
                self._rl_min_v_during_red = speed
            else:
                # Update “minimum speed” during red light
                if self._rl_min_v_during_red is None:
                    self._rl_min_v_during_red = speed
                else:
                    self._rl_min_v_during_red = min(self._rl_min_v_during_red, speed)

            # Rule 1: still not stopped after red light grace period => red light violation
            if (now - self._rl_red_start_t) >= RED_STOP_WINDOW and speed > STOP_SPEED_EPS:
                return True

            return False  # During red light but no violation yet

        # ================= Green Light Logic =================
        if state == carla.TrafficLightState.Green:
            if self._rl_red_start_t is not None:
                # Only make a determination within the short window after “just experienced red light”
                if (now - self._rl_red_start_t) <= RECENT_GREEN_WINDOW:
                    v_at_red = self._rl_v_at_red if self._rl_v_at_red is not None else speed
                    min_v = self._rl_min_v_during_red if self._rl_min_v_during_red is not None else speed
                    decel_amt = max(0.0, v_at_red - min_v)
                    slowed_enough = (decel_amt >= DECEL_DELTA_REQ) or (min_v <= STOP_SPEED_EPS)
                    if not slowed_enough:
                        _reset_red_episode(self)
                        return True
            _reset_red_episode(self)
            return False

        # ================= Yellow Light Logic (no violation check here, only maintain state) =================
        if state == carla.TrafficLightState.Yellow:
            # If transitioning from red->yellow, end the red light episode
            if state_changed and self._rl_red_start_t is not None:
                _reset_red_episode(self)
            return False

        # Other states (e.g., Off/Unknown), clear once to prevent stale state from affecting subsequent logic
        _reset_red_episode(self)
        return False

    # ========== tick & return signals ==========

    def tick(self):
        """
        Each frame:
         1) Execute LaneKeepAndChangeController.run_step() for all autonomous vehicles
         2) Check if EGO ran a red light
         3) Return (signals_list, self.ego_collision, self.collision, self.ego_cross_solid_line, self.ego_run_red_light)
        """
        signals_list = [None]*self.vehicle_num
        for i in range(self.vehicle_num):
            ctrl = self.controllers[i]
            if ctrl:
                control, signals = ctrl.run_step()
                self.vehicles[i].apply_control(control)
                signals_list[i] = signals
            else:
                signals_list[i] = None

        # If no red light violation has been recorded yet, check now
        if not self.ego_run_red_light:
            if self._detect_run_red_light():
                self.ego_run_red_light = True
                print("[INFO] EGO ran a red light!")

        return signals_list, self.ego_collision, self.collision, self.ego_cross_solid_line, self.ego_run_red_light

    # ========== Utility functions for the main script ==========

    def reconnect(self):
        """
        Closes the websocket connection and re-creates it so that data can be received again
        """
        self.ws.close()
        self.ws = create_connection(self.url)
        return

    def check_module_status(self, modules):
        """
        Checks if all modules in a provided list are enabled
        """
        module_status = self.get_module_status()
        for module, status in module_status.items():
            if not status and module in modules:
                log.warning("Warning: Apollo module {} is not running!!!".format(module))
                self.enable_module(module)
                time.sleep(1)

    def get_module_status(self):
        """
        Returns a dict where the key is the name of the module
        and value is a bool based on the module's current status
        """
        self.reconnect()
        data = json.loads(self.ws.recv())  # first recv => SimControlStatus
        while data["type"] != "HMIStatus":
            data = json.loads(self.ws.recv())
        # In practice, should parse data and return module status; this is just an example
        return {}

    def get_controller(self, idx):
        """
        Get the controller of the idx-th autonomous vehicle (0~N-1).
        """
        if idx < 0 or idx >= len(self.controllers):
            print(f"[WARN] get_controller: index {idx} out of range (0~{len(self.controllers)-1})!")
            return None
        return self.controllers[idx]

    def get_vehicle_positions(self):
        """
        Return a list of positions of all autonomous vehicles (excluding ego vehicle).
        """
        positions = []
        for v in self.vehicles:
            loc = v.get_location()
            positions.append(loc)
        return positions

    def destroy_all(self):
        """
        Destroy all sensors and vehicles at the end.
        """
        # # 1) Replace sensor callbacks with no-ops to prevent pending events from triggering logic
        # for s in self.collision_sensors:
        #     s.listen(lambda event: None)

        if self.lane_invasion_sensor_ego:
            self.lane_invasion_sensor_ego.listen(lambda event: None)

        # 2) In sync/async mode, tick or sleep to wait for the underlying system to fully clear callbacks
        for _ in range(3):
            self.world.wait_for_tick()

        # 3) Stop and destroy
        for s in self.collision_sensors:
            try:
                s.stop()
                s.destroy()
            except:
                pass
        self.collision_sensors.clear()

        if self.lane_invasion_sensor_ego:
            try:
                self.lane_invasion_sensor_ego.stop()
                self.lane_invasion_sensor_ego.destroy()
            except:
                pass
            self.lane_invasion_sensor_ego = None

        for _ in range(3):
            self.world.wait_for_tick()

        # Finally destroy vehicles
        for v in self.vehicles:
            try:
                v.destroy()
                self.world.wait_for_tick()
            except:
                pass
        self.vehicles.clear()

        # If ego_vehicle still exists
        if not self.external_ads and self.ego_vehicle:
            try:
                self.ego_vehicle.destroy()
            except:
                pass
            self.ego_vehicle = None

        # State reset
        self.collision = False
        self.ego_collision = False
        self.multi_vehicle_collision_count = 0
        self.rear_collision_count_vehicle = 0
        self.side_collision_count_vehicle = 0
        self.collision_count_obj = 0
        self.ego_cross_solid_line = 0
        self.ego_run_red_light = False
        self.world.wait_for_tick()
        # self.world.tick()

    def enable_module(self, module):
        """
        module is the name of the Apollo 5.0 module as seen in the "Module Controller" tab of Dreamview
        """
        self.ws.send(
            json.dumps({"type": "HMIAction", "action": "START_MODULE", "value": module})
        )
        return

    def disable_module(self, module):
        """
        module is the name of the Apollo 5.0 module as seen in the "Module Controller" tab of Dreamview
        """
        self.ws.send(
            json.dumps({"type": "HMIAction", "action": "STOP_MODULE", "value": module})
        )
        return

    # ========== Set Destination Example ==========

    def set_destination(self):
        """
        Perform a simple BFS on the ego vehicle's current lane to find the farthest same-direction waypoint => self.ego_destination.
        If WebSocket is available, can send RoutingRequest (optional).
        """
        if not self.ego_vehicle:
            print("[ERROR] ego_vehicle not spawned, cannot set_destination.")
            return

        # 1) Get the waypoint corresponding to ego's position
        ego_loc = self.ego_vehicle.get_location()
        start_wp = self.map.get_waypoint(ego_loc, lane_type=carla.LaneType.Driving)
        if not start_wp:
            print("[ERROR] ego_vehicle waypoint is empty, cannot set_destination.")
            return

        import collections
        queue = collections.deque()
        visited = set()
        queue.append((start_wp, 0.0))
        same_direction_wps = []

        init_lane_id = start_wp.lane_id
        init_lane_sign = 1 if init_lane_id >= 0 else -1

        while queue:
            cur_wp, dist_so_far = queue.popleft()
            if cur_wp in visited:
                continue
            visited.add(cur_wp)

            same_direction_wps.append((cur_wp, dist_so_far))

            if dist_so_far > max_search_distance_for_destination:
                continue

            nxt_wps = cur_wp.next(step_dist_for_destination)
            for nxt_wp in nxt_wps:
                nxt_lane_sign = 1 if nxt_wp.lane_id >= 0 else -1
                if nxt_lane_sign == init_lane_sign:
                    dist_increment = cur_wp.transform.location.distance(nxt_wp.transform.location)
                    new_dist = dist_so_far + dist_increment
                    if new_dist <= (max_search_distance_for_destination + step_dist_for_destination):
                        queue.append((nxt_wp, new_dist))

        if not same_direction_wps:
            print("[WARNING] No same-direction waypoints found => set_destination failed")
            return

        # Find the farthest point
        furthest_wp, furthest_dist = max(same_direction_wps, key=lambda x: x[1])
        self.ego_destination = furthest_wp.transform.location
        print(f"[INFO] set_destination: target point (x={self.ego_destination.x:.2f}, y={self.ego_destination.y:.2f}), dist={furthest_dist:.1f}m")

        # If you have WebSocket => send RoutingRequest (optional)
        apollo_data = fetch_localization_variable()
        if self.ws and apollo_data is not None and 'position' in apollo_data:
            try:
                yaw_deg = self.ego_vehicle.get_transform().rotation.yaw
                yaw_rad = math.radians(yaw_deg)

                msg = {
                    "type": "SendRoutingRequest",
                    "start": {
                        "x": apollo_data['position']['x'],
                        "y": apollo_data['position']['y'],
                        "z": apollo_data['position']['z'],
                        "heading": -yaw_rad,
                    },
                    "end": {
                        "x": self.ego_destination.x,
                        "y": -self.ego_destination.y,
                        "z": apollo_data['position']['z'],
                    },
                    "waypoint": "[]",
                }
                self.ws.send(json.dumps(msg))
                print("[INFO] Routing request sent:", json.dumps(msg))
            except WebSocketException as e:
                print(f"[ERROR] WebSocket error while sending RoutingRequest: {e}")
            except Exception as e:
                print(f"[ERROR] Internal error in set_destination: {e}")

    def close_connection(self):
        """
        Close the WebSocket connection at the end, if one exists.
        """
        self.ws_running = False
        if self.ws:
            try:
                self.ws.close()
                print("[INFO] WebSocket connection closed.")
            except Exception as e:
                print(f"[ERROR] Error while closing WebSocket connection: {e}")
