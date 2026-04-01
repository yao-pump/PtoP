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

max_search_distance_for_destination = 200  # BFS搜寻的最大距离
step_dist_for_destination = 2.0            # BFS步进距离(米)
max_search_distance_for_spawns = 50.0      # 用于多车道spawn选点
step_for_spawns = 1.0

log = logging.getLogger(__name__)


EGO_FAULT_CLOSE_SPEED_MIN = 0.8   # m/s，EGO 沿碰撞法线方向的最小逼近速度
EGO_FAULT_RATIO          = 0.60   # EGO 逼近速度占双方总逼近速度的比例阈值
IMPULSE_MIN              = 400.0  # 碰撞冲量下限，过小可视作擦碰/假阳性
REAR_END_BONUS           = 0.05   # 追尾情形下，略放宽比例阈值

# -------- 工具：向量/速度/单位向量/点积 --------
def _vec_norm(v):
    return math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)

def _spd_and_vec(actor):
    v = actor.get_velocity()
    return _vec_norm(v), v

def _unit_vec(a: "carla.Location", b: "carla.Location"):
    dx, dy, dz = (b.x - a.x), (b.y - a.y), (b.z - a.z)
    n = math.sqrt(dx*dx + dy*dy + dz*dz) + 1e-9
    return carla.Vector3D(dx/n, dy/n, dz/n)

def _dot(a: "carla.Vector3D", b: "carla.Vector3D"):
    return a.x*b.x + a.y*b.y + a.z*b.z

# 若你的文件里没有 ego_local_sd，就复制你已有的实现过来
def _ego_local_sd(ego_tf: "carla.Transform", loc: "carla.Location"):
    yaw = math.radians(ego_tf.rotation.yaw)
    cy, sy = math.cos(yaw), math.sin(yaw)
    dx = loc.x - ego_tf.location.x
    dy = loc.y - ego_tf.location.y
    s =  dx * cy + dy * sy
    d = -dx * sy + dy * cy
    return s, d

def _assign_blame_ego(ego: "carla.Vehicle",
                      other: "carla.Actor",
                      world_map: "carla.Map",
                      event_normal_impulse: "carla.Vector3D") -> (bool, str):
    """
    返回: (ego_fault: bool, why: str)
    判定依据：
      1) 取 ego->other 的单位向量 n，计算两者沿 ±n 的逼近速度分量 c_ego、c_other；
      2) 比例 r = c_ego / (c_ego + c_other)；
      3) 满足：冲量够大 & c_ego 超过最小值 & r >= 阈值（追尾略放宽） => 归因 EGO。
    """
    try:
        # 位置/连线单位向量（从 EGO 指向对方）
        loc_e = ego.get_transform().location
        loc_o = other.get_transform().location
        n = _unit_vec(loc_e, loc_o)

        # 速度与沿连线的“逼近分量”
        spd_e, v_e = _spd_and_vec(ego)
        if hasattr(other, "get_velocity"):
            spd_o, v_o = _spd_and_vec(other)
        else:
            spd_o, v_o = 0.0, carla.Vector3D(0.0, 0.0, 0.0)

        # EGO 朝向对方的速度分量（>0 代表在朝 other 方向运动）
        c_ego   = max(0.0, _dot(v_e, n))
        # OTHER 朝向 EGO 的速度分量：other 朝向 -n，等价于 -(v_o·n)
        c_other = max(0.0, -_dot(v_o, n))

        # 冲量强度
        J = _vec_norm(event_normal_impulse)

        # 位姿关系（用于识别追尾/侧撞）
        s_rel, d_rel = _ego_local_sd(ego.get_transform(), loc_o)  # s>0: 对方在 EGO 前方
        rear_end_like = (s_rel > 0.0 and c_ego > c_other)  # 更像 EGO 追尾前车
        # 侧向擦碰时 d_rel 很大，可提高 IMPULSE_MIN 或直接降低权重（此处保持简单）

        # 比例与阈值
        r = c_ego / (c_ego + c_other + 1e-9)
        thr = EGO_FAULT_RATIO - (REAR_END_BONUS if rear_end_like else 0.0)

        # 判定
        if J >= IMPULSE_MIN and c_ego >= EGO_FAULT_CLOSE_SPEED_MIN and r >= thr:
            reason = f"ego_fault: J={J:.1f}, c_ego={c_ego:.2f}, c_other={c_other:.2f}, r={r:.2f}, rear_end={rear_end_like}"
            return True, reason
        else:
            reason = f"non_ego_fault: J={J:.1f}, c_ego={c_ego:.2f}, c_other={c_other:.2f}, r={r:.2f}, rear_end={rear_end_like}"
            return False, reason
    except Exception as e:
        # 出错时保守不归因于 EGO
        return False, f"non_ego_fault: exception {e}"
def fetch_localization_variable(url="http://127.0.0.1:5000/var"):
    """
    通过 HTTP GET 请求获取容器中最新定位数据
    :param url: Flask 接口地址，默认使用本机 127.0.0.1:5000/var
    :return: 返回 JSON 格式的定位数据字典，或 None（失败时）
    """
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # 如果响应状态码不是200，会抛出异常
        data = response.json()
        return data
    except Exception as e:
        print("获取变量数据时发生错误:", e)
        return None

def distance(loc1, loc2):
    """简单的欧几里得距离"""
    return math.sqrt(
        (loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2
    )

# ========= 新增：spawn 最小间距保护 =========
SPAWN_GAP_VEH = 4.5   # 车辆之间的最小间距（米）
SPAWN_GAP_PED = 2.5   # 行人之间的最小间距（米）
SPAWN_GAP_EGO = 7.0   # NPC 与 EGO 的最小间距（米）

def _gap_ok(tf, accepted_tfs, min_gap):
    """tf 是否与 accepted_tfs 中所有 Transform 的2D距离均≥min_gap"""
    for t in accepted_tfs:
        dx = tf.location.x - t.location.x
        dy = tf.location.y - t.location.y
        if math.hypot(dx, dy) < float(min_gap):
            return False
    return True
# ========================================

class MultiVehicleDemo:
    """
    主要功能:
      1) 生成 ego + N辆自动车(挂 LaneKeepAndChangeController)
      2) 为所有车辆附加碰撞传感器
         - ego 车碰撞时 => 判断"ego主动撞" or "别人撞ego"
         - 其他车碰撞时 => 紧急刹车，但不结束
      3) 在 tick() 中, 返回 (signals_list, ego_collision, self.collision, ego_cross_solid_line, ego_run_red_light)
      4) 提供 set_destination() => 通过 BFS 寻找同向最远路点, 保存为 self.ego_destination
      5) 提供 get_controller(idx) => 获取第 idx 辆自动车的控制器
    """

    def __init__(self, world, external_ads, websocket_url="ws://localhost:8888/websocket",
                 gps_offset=carla.Vector3D(x=1.0, y=0.0, z=0.5)):
        self.world = world
        self.population_size = 10
        self.map = world.get_map()
        self.ego_vehicle = None
        self.multi_vehicle_collision_count = 0
        self.vehicles = []            # 保存自动车
        self.controllers = None       # N个 LaneKeepAndChangeController
        self.url = websocket_url
        self.gps_offset = gps_offset
        self.ws = None
        self.vehicle_num = None
        self.ws_thread = None
        self.ws_running = False
        self.ws_receive_buffer = []
        self.ego_spawning_point = None
        self.ego_destination = None   # 通过 set_destination 设置
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
        self.side_collision_count_vehicle = 0  # 侧方碰撞数
        self.rear_collision_count_vehicle = 0  # 追尾碰撞数
        self.collision_count_obj = 0

        # 标记ego是否主动撞别人
        self.ego_collision = False

        # 地图边界
        self.map_bounds = self._compute_map_bounds()

        # 碰撞传感器列表
        self.collision_sensors = []

        # ----- LaneInvasion 相关 -----
        self.ego_cross_solid_line = False  # EGO是否压实线
        self.lane_invasion_sensor_ego = None

        # ----- 闯红灯检测相关 -----
        self.ego_run_red_light = False  # 是否检测到EGO闯红灯

        if self.external_ads:
            self._connect_websocket()

    # ========== 基础函数 ==========
    # ---- 新增：判定是否为“NPC 追尾 EGO” ----
    def _is_npc_rear_end(self, ego: "carla.Vehicle", npc: "carla.Vehicle") -> bool:
        """
        返回 True 当且仅当：对方车辆在 EGO 后方、与 EGO 同向且沿车道前进速度更大（逼近），
        且横向偏差不大（基本同车道）。阈值可按需调整。
        """
        try:
            ego_tf = ego.get_transform()
            npc_tf = npc.get_transform()

            # ego 坐标系下的相对纵/横向
            s_rel, d_rel = self._ego_local_sd(ego_tf, npc_tf.location)

            # ego 前向单位向量
            yaw = math.radians(ego_tf.rotation.yaw)
            fwd = carla.Vector3D(math.cos(yaw), math.sin(yaw), 0.0)

            v_e = ego.get_velocity()
            v_n = npc.get_velocity()
            v_e_f = v_e.x * fwd.x + v_e.y * fwd.y + v_e.z * fwd.z
            v_n_f = v_n.x * fwd.x + v_n.y * fwd.y + v_n.z * fwd.z
            dv_f  = v_n_f - v_e_f  # NPC 相对 EGO 的前向速度差（>0 表示从后往前逼近）

            # 航向接近（同向），并且横向偏差小（基本在同一车道）
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
            print(f"[INFO] 已连接到WebSocket服务器: {self.url}")
            # 启动一个线程来接收消息
            self.ws_thread = threading.Thread(target=self._receive_messages, daemon=True)
            self.ws_thread.start()
        except WebSocketException as e:
            print(f"[ERROR] 无法连接到WebSocket服务器: {e}")
            self.ws = None

    def _receive_messages(self):
        while self.ws_running:
            try:
                result = self.ws.recv()
                if result:
                    self.ws_receive_buffer.append(result)
            except WebSocketException as e:
                print(f"[ERROR] WebSocket接收消息时出错: {e}")
                self.ws_running = False
            except Exception as e:
                print(f"[ERROR] 未知错误: {e}")
                self.ws_running = False

    def _compute_map_bounds(self):
        """
        简单地通过 map.generate_waypoints(2.0) 获取地图x,y范围
        """
        wps = self.map.generate_waypoints(2.0)
        if not wps:
            print("[WARN] generate_waypoints为空,地图无数据?")
            return (0, 0, 0, 0)

        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        for wp in wps:
            loc = wp.transform.location
            if loc.x < min_x: min_x = loc.x
            if loc.x > max_x: max_x = loc.x
            if loc.y < min_y: min_y = loc.y
            if loc.y > max_y: max_y = loc.y
        print(f"[INFO] 地图x范围=({min_x:.1f},{max_x:.1f}), y范围=({min_y:.1f},{max_y:.1f})")
        return (min_x, max_x, min_y, max_y)

    def get_map_bounds(self):
        return self.map_bounds

    # ========== 车辆生成逻辑 ==========

    def setup_vehicles(self, scenario_conf):
        """
        1) 生成 EGO
        2) 按 scenario_conf['surrounding_info'] 的顺序生成 NPC（车/自行车/行人）
           - 若某 NPC 位置冲突：就地重采样（沿车道前后&横向微移），直到生成成功
           - 行人失败：从导航网格反复重采样，直到成功
        3) 成功后把“最终成功位置”回写到 scenario_conf 中，保证场景表示与实际一致
        """
        world = self.world
        world_map = world.get_map()
        blueprint_library = world.get_blueprint_library()

        self.vehicle_num = int(scenario_conf["vehicle_num"])
        self.controllers = [None] * self.vehicle_num

        # ------- surrounding_info 解析 -------
        surrounding = scenario_conf["surrounding_info"]

        # 两类表示：list[{"transform","type"}] 或 dict{"transform":[...], "type":[...]}
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
            # 外部 ADS：找到已存在的 mkz_2017 并移动到 ego_transform
            all_actors = world.get_actors()
            candidate_vehicles = all_actors.filter("vehicle.*")
            for v in candidate_vehicles:
                if "mkz_2017" in v.type_id:
                    self.ego_vehicle = v
                    break
            if not self.ego_vehicle:
                print("[ERROR] 未找到 'mkz_2017' 作为 EGO。")
                return False
            self.ego_vehicle.set_transform(self.ego_spawning_point)

        if not self.ego_vehicle:
            print("[ERROR] EGO 车辆生成失败。")
            return False

        # ------- 蓝图池 -------
        veh_bps_all = blueprint_library.filter("vehicle.*")
        car_bps = blueprint_library.filter("vehicle.tesla.model3") or veh_bps_all
        bike_bps = [bp for bp in veh_bps_all if ("bicycle" in bp.id.lower() or "bike" in bp.id.lower())]
        walker_bps = blueprint_library.filter("walker.pedestrian.*")

        def _pick(pool, fallback):
            if pool: return random.choice(pool)
            if fallback: return random.choice(fallback)
            return random.choice(veh_bps_all)

        # ------- 几何/车道辅助 -------
        def _project_to_lane(tf, clip_ratio=0.45):
            wp = world_map.get_waypoint(tf.location, project_to_road=True,
                                        lane_type=carla.LaneType.Driving)
            if not wp:
                return tf, None
            lane_w = float(getattr(wp, "lane_width", 3.5))
            # 保持相对横向偏移，但裁剪到 0.45*lane_width
            center = wp.transform.location
            right = wp.transform.get_right_vector()
            dv = carla.Vector3D(tf.location.x - center.x, tf.location.y - center.y, tf.location.z - center.z)
            dlat = dv.x * right.x + dv.y * right.y + dv.z * right.z
            d_clip = float(np.clip(dlat, -clip_ratio * lane_w, clip_ratio * lane_w))
            loc = carla.Location(center.x + d_clip * right.x,
                                 center.y + d_clip * right.y,
                                 tf.location.z)
            # yaw 对齐车道更稳
            yaw = wp.transform.rotation.yaw
            return carla.Transform(loc, carla.Rotation(pitch=tf.rotation.pitch, yaw=yaw, roll=tf.rotation.roll)), lane_w

        def _lane_shift_candidates(tf0, max_forward=18.0, step_s=2.0, step_d=0.75, d_mul=0.45):
            """沿车道中心线前后 ±s，再横向 ±d 采样候选位姿（优先近的）"""
            base_wp = world_map.get_waypoint(tf0.location, project_to_road=True,
                                             lane_type=carla.LaneType.Driving)
            if not base_wp:
                return [tf0]

            # 生成 s 偏移序列（0, +2, -2, +4, -4, ...）
            s_vals = [0.0]
            k = int(max_forward // step_s)
            for i in range(1, k + 1):
                s_vals += [i * step_s, -i * step_s]

            # 生成 d 偏移序列（0, +0.75, -0.75, +1.5, -1.5, ...）
            lane_w = float(getattr(base_wp, "lane_width", 3.5))
            d_max = d_mul * lane_w
            d_vals = [0.0]
            kd = max(1, int(d_max // step_d))
            for i in range(1, kd + 1):
                d_vals += [i * step_d, -i * step_d]

            cands = []
            for s in s_vals:
                # ---- 关键修复：绝不调用 next/previous(0.0) ----
                if s > 0.0:
                    wps = base_wp.next(s)
                elif s < 0.0:
                    wps = base_wp.previous(-s)
                else:
                    wps = [base_wp]  # s == 0，直接使用当前 waypoint

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

        # ------- 容器 -------
        if not hasattr(self, "vehicles"): self.vehicles = []
        if not hasattr(self, "pedestrians"): self.pedestrians = []

        # ------- 逐个生成 NPC（失败则重采样直至成功） -------
        spawned_vehicle_count = 0
        spawned_ped_count = 0

        print('vehicle number (requested):', self.vehicle_num)

        # 新增：记录已成功放置的 Transform（分别对车辆/行人）
        veh_tfs = []
        ped_tfs = []

        for i in range(n_to_spawn):
            init_tf, npc_type = _get_item(i)
            actor = None

            try:
                if npc_type == "pedestrian":
                    if not walker_bps:
                        print(f"[WARN] 无行人蓝图，NPC[{i}] 跳过。")
                        continue
                    bp = random.choice(walker_bps)

                    # 先试一次原位（若满足最小间距）
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
                        # 极少数情况下 ego 尚未 spawn，降级只检查与已放置行人
                        if _gap_ok(init_tf, ped_tfs, SPAWN_GAP_PED):
                            actor = world.try_spawn_actor(bp, init_tf)

                    if not actor:
                        # 导航网格重采样直到成功
                        attempts = 0
                        while actor is None:
                            loc = world.get_random_location_from_navigation()
                            if loc is None:
                                attempts += 1
                                if attempts % 10 == 0: _tick_flush()
                                continue
                            tf_try = carla.Transform(loc, init_tf.rotation)

                            # 与 EGO / 已放置行人 的距离约束
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
                        # 原位成功：也回写（保持一致）
                        _set_item_transform(i, init_tf)
                        ped_tfs.append(init_tf)

                    if actor:
                        self.pedestrians.append(actor)
                        spawned_ped_count += 1
                    else:
                        # 兜底：继续强刷（保留你的原有逻辑，但也加上间距约束）
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
                    # 车辆 / 自行车（回退为四轮车），采用车道附近重采样，直到成功
                    if npc_type == "bicycle":
                        bp = _pick(bike_bps, car_bps)
                    elif npc_type == "car":
                        bp = _pick(car_bps, veh_bps_all)
                    else:
                        bp = _pick(car_bps, veh_bps_all)

                    # 先对 init_tf 做车道投影裁剪
                    tf0, _ = _project_to_lane(init_tf)

                    # 先试一次投影后的原位（若满足最小间距）
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
                        # 沿车道生成候选并不断尝试；若一轮不成，扩大范围/再次采样
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
                            # 扩大搜索范围再来一轮
                            max_forward = min(max_forward + 12.0, 60.0)
                            if attempts > 200:
                                # 兜底：从全局 spawn_points 随机抽直到成功
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
                                # 继续强刷（直到成功），但每 30 次给下 tick
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
                print(f"[ERROR] 生成 NPC[{i}] 出错: {e}")
                # 强制进入“直到成功”为止的兜底逻辑（车辆为例）
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
                        # 全局随机 spawn_point
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

        print("spawned vehicles (vehicle.*):", spawned_vehicle_count)
        print("spawned pedestrians:", spawned_ped_count)

        # 为已生成的“车辆”（不含行人）挂上控制器

        for i, v in enumerate(self.vehicles):
            try:
                self.controllers[i] = LaneKeepAndChangeController(v)
            except Exception as e:
                print(f"[WARN] 控制器创建失败 veh[{i}] id={v.id}: {e}")

        return True

    def _is_valid_side_lane(self, wp, side_wp):
        """
        判断左右车道是否为 Driving 且与本车道lane_id同向(同正负)
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
        对外接口：
        1) 先 setup_vehicles
        2) 若成功 => _setup_collision_sensors
        """
        success = self.setup_vehicles(scenario_conf)
        if success:
            self._setup_collision_sensors()
        return success

    # ========== 碰撞传感器逻辑 + LaneInvasion传感器逻辑 ==========

    def _setup_collision_sensors(self):
        """
        给自车(ego) + 所有自动车都加碰撞传感器
        """
        blueprint_library = self.world.get_blueprint_library()
        collision_bp = blueprint_library.find('sensor.other.collision')

        # ego 碰撞传感器
        if self.ego_vehicle:
            collision_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
            sensor_ego = self.world.spawn_actor(collision_bp, collision_transform, attach_to=self.ego_vehicle)
            ### 新增/修改行：回调中带上本sensor引用
            sensor_ego.listen(lambda event, v=self.ego_vehicle, s=sensor_ego: self.collision_callback(event, v, s))
            self.collision_sensors.append(sensor_ego)
            print(f"[INFO] Ego Vehicle {self.ego_vehicle.id} 碰撞传感器已附加: {sensor_ego.id}")

            # ----- 给Ego车附加 LaneInvasionSensor -----
            lane_invasion_bp = blueprint_library.find('sensor.other.lane_invasion')
            lane_invasion_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.0))
            self.lane_invasion_sensor_ego = self.world.spawn_actor(
                lane_invasion_bp,
                lane_invasion_transform,
                attach_to=self.ego_vehicle
            )
            self.lane_invasion_sensor_ego.listen(self.lane_invasion_callback)
            print(f"[INFO] Ego Vehicle {self.ego_vehicle.id} 车道侵入传感器已附加: {self.lane_invasion_sensor_ego.id}")

        # 其它自动车碰撞传感器
        for veh in self.vehicles:
            collision_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
            sensor = self.world.spawn_actor(collision_bp, collision_transform, attach_to=veh)
            ### 新增/修改行：回调中带上本sensor引用
            sensor.listen(lambda event, v=veh, s=sensor: self.collision_callback(event, v, s))
            self.collision_sensors.append(sensor)
            print(f"[INFO] 车辆 {veh.id} 碰撞传感器已附加: {sensor.id}")

    def lane_invasion_callback(self, event):
        """
        当ego车辆跨越车道线时触发。判断是否包含实线LaneMarking。
        """
        for marking in event.crossed_lane_markings:
            if marking.type in [
                carla.LaneMarkingType.Solid,
                carla.LaneMarkingType.SolidSolid,
                carla.LaneMarkingType.SolidBroken,
                carla.LaneMarkingType.BrokenSolid
            ]:
                # 一旦检测到包含实线类型 => 表示压实线
                self.ego_cross_solid_line = True
                print("[INFO] EGO车辆压实线！")
                break

    # ==================== 责任判定（仅这一块是新增/替换） ====================

    # 阈值（按需调整）
    EGO_FAULT_CLOSE_SPEED_MIN = 0.8   # m/s，EGO 沿碰撞法线方向的最小逼近速度
    EGO_FAULT_RATIO          = 0.60   # EGO 逼近速度占双方总逼近速度的比例阈值
    IMPULSE_MIN              = 400.0  # 碰撞冲量下限
    REAR_END_BONUS           = 0.05   # 追尾情形下放宽比例阈值

    # ---- 辅助：做成静态方法，便于类内调用 ----
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
        """
        返回: (ego_fault: bool, why: str)
        依据：沿 EGO→对方 的连线方向，比较双方“逼近速度分量”的占比 + 冲量与最小速度门槛。
        """
        try:
            # 连线单位向量（从 EGO 指向对方）
            loc_e = ego.get_transform().location
            loc_o = other.get_transform().location
            n = self._unit_vec(loc_e, loc_o)

            # 速度分量
            _, v_e = self._spd_and_vec(ego)
            if hasattr(other, "get_velocity"):
                _, v_o = self._spd_and_vec(other)
            else:
                v_o = carla.Vector3D(0.0, 0.0, 0.0)

            c_ego   = max(0.0, self._dot(v_e, n))     # EGO 朝向对方
            c_other = max(0.0, -self._dot(v_o, n))    # 对方朝向 EGO（相当于 -v_o·n）

            # 冲量强度
            J = self._vec_norm(event_normal_impulse)

            # 位姿关系（追尾判别：对方在正前且 EGO 逼近更大）
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
        只统计由 EGO 原因导致的碰撞；否则仅紧急制动不计入 EGO 碰撞指标。
        """
        # 已统计过可按需直接 return（或继续处理以停表/销毁）
        # if self.collision_count_obj == 1 or self.multi_vehicle_collision_count == 1 or self.side_collision_count_vehicle == 1:
        #     return

        # other_actor = event.other_actor
        # is_vehicle = hasattr(other_actor, "type_id") and (str(other_actor.type_id).startswith("vehicle.") or str(other_actor.type_id).startswith("bicycle") or str(other_actor.type_id).startswith("bike"))

        if vehicle == self.ego_vehicle:
            # 仅当 EGO 参与时考虑归因
            # ego_fault, why = _assign_blame_ego(self.ego_vehicle, other_actor, self.map, event.normal_impulse)
            self.collision = True
            self.ego_collision = True
            self.side_collision_count_vehicle = 1
            # if is_vehicle:
            #     if ego_fault:


                    #
                    # # 多车 or 侧方/追尾 的粗判（可更精细）
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



                    # print(f"[INFO] EGO 与车辆碰撞（EGO 责任） | {why}")
                # else:
                    # 非 EGO 责任：不计入 EGO 指标
                    # print(f"[INFO] 车辆碰撞，但不归因于 EGO | {why}")
            # else:
            #
            #     self.collision_count_obj = 1
            #     print(f"[INFO] EGO 撞上静止/非车辆物体（EGO 责任） | {why}")

        else:
            # 非 EGO 的碰撞：只做紧急制动（不计入 EGO 指标）
            if vehicle in getattr(self, "vehicles", []):
                try:
                    idx = self.vehicles.index(vehicle)
                    controller = self.controllers[idx]
                    if controller:
                        controller.brake()
                except Exception:
                    pass

            # 直接下刹车
            try:
                cur = vehicle.get_control()
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=getattr(cur, "steer", 0.0)))
            except Exception:
                pass

        # 传感器一次性：收尾
        try:
            sensor.stop()
            sensor.destroy()
            print(f"[INFO] 碰撞传感器 {sensor.id} 已销毁 (一次性).")
        except Exception:
            pass

        if sensor in getattr(self, "collision_sensors", []):
            try:
                self.collision_sensors.remove(sensor)
            except Exception:
                pass

    # ========== 闯红灯检测逻辑 ==========

    def _detect_run_red_light(self):
        """
        检测 EGO 是否闯红灯（有宽限+减速过程判断）。
        与原函数同名同签名：无参，返回 bool。
        需：self.ego_vehicle 存在；carla 已导入。
        可选：若 self.world 存在则优先使用仿真时间。
        """
        import math, time

        # ---------- 可调阈值（也可在外部提前设 self._rl_* 来覆盖） ----------
        RED_STOP_WINDOW = getattr(self, "_rl_red_stop_window", 2.0)  # 红灯宽限期（秒）
        STOP_SPEED_EPS = getattr(self, "_rl_stop_speed_eps", 0.2)  # 视为已停（m/s）
        DECEL_DELTA_REQ = getattr(self, "_rl_decel_delta_req", 0.5)  # 红转绿前至少降速（m/s）
        RECENT_GREEN_WINDOW = getattr(self, "_rl_recent_green_window", 3.0)  # 红转绿后的追溯窗口（秒）

        def _now():
            # 优先仿真时间
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

        # ---------- 初始化内部状态 ----------
        if not hasattr(self, "_rl_last_tl_state"):
            self._rl_last_tl_state = None
        if not hasattr(self, "_rl_red_start_t"):
            self._rl_red_start_t = None
            self._rl_v_at_red = None
            self._rl_min_v_during_red = None

        # ---------- 基础可见性检查 ----------
        if not getattr(self, "ego_vehicle", None):
            return False
        tlight = self.ego_vehicle.get_traffic_light()
        if tlight is None:
            # 看不到红绿灯时，重置一次 episode
            self._rl_last_tl_state = None
            _reset_red_episode(self)
            return False

        state = tlight.get_state()
        now = _now()
        speed = _speed_of(self.ego_vehicle)

        state_changed = (state != self._rl_last_tl_state)
        self._rl_last_tl_state = state

        # ================= 红灯逻辑 =================
        if state == carla.TrafficLightState.Red:
            if self._rl_red_start_t is None:
                # 刚进入红灯
                self._rl_red_start_t = now
                self._rl_v_at_red = speed
                self._rl_min_v_during_red = speed
            else:
                # 红灯期间更新“最低速度”
                if self._rl_min_v_during_red is None:
                    self._rl_min_v_during_red = speed
                else:
                    self._rl_min_v_during_red = min(self._rl_min_v_during_red, speed)

            # 规则1：红灯超过宽限期仍未停下 => 闯红灯
            if (now - self._rl_red_start_t) >= RED_STOP_WINDOW and speed > STOP_SPEED_EPS:
                return True

            return False  # 红灯中但尚未违规

        # ================= 绿灯逻辑 =================
        if state == carla.TrafficLightState.Green:
            if self._rl_red_start_t is not None:
                # 仅在“刚经历过红灯”的短窗口内做一次判定
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

        # ================= 黄灯逻辑（此处不判违规，仅维护状态） =================
        if state == carla.TrafficLightState.Yellow:
            # 若从红->黄，结束红灯 episode
            if state_changed and self._rl_red_start_t is not None:
                _reset_red_episode(self)
            return False

        # 其它状态（如 Off/Unknown），清空一次以免脏状态影响后续
        _reset_red_episode(self)
        return False

    # ========== tick & 返回 signals ==========

    def tick(self):
        """
        每帧:
         1) 让所有自动车执行 LaneKeepAndChangeController.run_step()
         2) 检查 EGO 是否闯红灯
         3) 返回 (signals_list, self.ego_collision, self.collision, self.ego_cross_solid_line, self.ego_run_red_light)
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

        # 若尚未记录过闯红灯，则检测一下
        if not self.ego_run_red_light:
            if self._detect_run_red_light():
                self.ego_run_red_light = True
                print("[INFO] EGO 发生闯红灯！")

        return signals_list, self.ego_collision, self.collision, self.ego_cross_solid_line, self.ego_run_red_light

    # ========== 提供给主脚本的工具函数 ==========

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
        # 实际中应解析 data 并返回模块状态，这里仅做示例
        return {}

    def get_controller(self, idx):
        """
        获取第 idx 辆自动车(0~N-1)的控制器
        """
        if idx < 0 or idx >= len(self.controllers):
            print(f"[WARN] get_controller: 索引 {idx} 超出范围(0~{len(self.controllers)-1})!")
            return None
        return self.controllers[idx]

    def get_vehicle_positions(self):
        """
        返回所有自动车(不包含ego车)的位置list
        """
        positions = []
        for v in self.vehicles:
            loc = v.get_location()
            positions.append(loc)
        return positions

    def destroy_all(self):
        """
        结束时销毁所有传感器与车辆
        """
        # # 1) 先把传感器回调改为空函数，防止还没处理完的事件再调用逻辑
        # for s in self.collision_sensors:
        #     s.listen(lambda event: None)

        if self.lane_invasion_sensor_ego:
            self.lane_invasion_sensor_ego.listen(lambda event: None)

        # 2) 同步/异步模式下，tick 或 sleep 等待底层彻底清空回调
        for _ in range(3):
            self.world.wait_for_tick()

        # 3) 停止并销毁
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

        # 最后再销毁车辆
        for v in self.vehicles:
            try:
                v.destroy()
                self.world.wait_for_tick()
            except:
                pass
        self.vehicles.clear()

        # 如果还有 ego_vehicle
        if not self.external_ads and self.ego_vehicle:
            try:
                self.ego_vehicle.destroy()
            except:
                pass
            self.ego_vehicle = None

        # 状态重置
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

    # ========== 设定目的地示例 ==========

    def set_destination(self):
        """
        在ego车当前车道上做一个简单的 BFS, 找同向的最远路点 => self.ego_destination
        如果有websocket, 可发送RoutingRequest(可选)
        """
        if not self.ego_vehicle:
            print("[ERROR] ego_vehicle未生成, 无法set_destination.")
            return

        # 1) 获取ego位置对应的waypoint
        ego_loc = self.ego_vehicle.get_location()
        start_wp = self.map.get_waypoint(ego_loc, lane_type=carla.LaneType.Driving)
        if not start_wp:
            print("[ERROR] ego_vehicle waypoint为空, 无法set_destination.")
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
            print("[WARNING] 未找到同向的waypoints => set_destination失败")
            return

        # 找最远点
        furthest_wp, furthest_dist = max(same_direction_wps, key=lambda x: x[1])
        self.ego_destination = furthest_wp.transform.location
        print(f"[INFO] set_destination: 目标点 (x={self.ego_destination.x:.2f}, y={self.ego_destination.y:.2f}), dist={furthest_dist:.1f}m")

        # 如果你有WebSocket => 发送RoutingRequest(可选)
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
                print(f"[ERROR] 发送RoutingRequest时WebSocket错误: {e}")
            except Exception as e:
                print(f"[ERROR] set_destination内部错误: {e}")

    def close_connection(self):
        """
        若有 websocket 连接, 在结束时关闭
        """
        self.ws_running = False
        if self.ws:
            try:
                self.ws.close()
                print("[INFO] WebSocket连接已关闭。")
            except Exception as e:
                print(f"[ERROR] 关闭WebSocket连接时出错: {e}")
