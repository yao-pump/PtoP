#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主程序（完整可运行骨架）：
- GA 采样 + 代末 SVGD 微调（对 NPC 初始 (ds, dd, dyaw)）
- ART（sb_art.py）在线操控 NPC（自适应策略 + 组合动作）
- MLP surrogate：输入 = 起点特征，标签 = episode 内 NPC 与 EGO 最近距离时刻的 hazard（min-distance 方案）
- 在线每回合 1-2 epoch 微调；可选 EMA（若 surrogate 提供 ema_update）
"""

import os
import cv2
import math
import time
import shutil
import random
import numpy as np
from queue import Queue
import json
from datetime import datetime
import torch
import torch.nn as nn

import carla

from utility import (
    has_passed_destination, action_trans, apollo_clear_prediction_planning, purge_npcs
)
from world import MultiVehicleDemo
from offline_searcher import CombinedGA
from npc_surrogate_mlp import NPCHazardMLPSurrogate
from npc_svgd_runtime import RuntimeNPCSVGD
from replay_buffer import NearMissReplay
from ART_fuzzer import (
    ARTSelectorScenario, update_adaptive_policy, filter_triggerables, build_actions_for
)

# ====================== 全局超参数 ======================
TIME_STEP = 0.05
population_size = 10  # 每代 10 个有效回合

# —— Traffic Manager —— #
KEEP_ALIVE_PERIOD = 30  # TM 续权周期（tick）

# —— 录像 —— #
RECORDING = True

# —— 早退/兜底 —— #
EPISODE_MAX_SECONDS = 180.0
NO_PROGRESS_SECONDS = 15
PROGRESS_THRESH = 2.0
STARTUP_STEPS = 500

# —— SVGD 门控（代际末触发，不在回合内） —— #
NEARMISS_SAVE_TAU = 0.5
SVGD_TOP_CASES = 5
SVGD_STEPS_PER_CASE = 8
SVGD_EPS = 0.08         # 步长更小
SVGD_BETA = 3        # 排斥更强
SVGD_GRAD_EPS = 0.35    # 核更窄

# —— KING 规划器参数 —— #
REPLAN_STRIDE = 5     # 每多少 tick 重新规划一次
H = 25                # 规划地平线步数
DT_PLAN = 0.10        # 规划步长（可与 CARLA tick 不同）
N_OPT = 30            # 每次滚动优化迭代轮数
LR = 5e-2             # Adam 学习率
TAU_SOFTMIN = 1.0     # 软最小距离温度
ACC_LIMIT = 3.0       # m/s^2
STEER_LIMIT = 0.6     # rad (~34°)
W_U = 1e-3            # 控制幅度正则
W_DU = 5e-3           # 控制平滑正则
V_MIN, V_MAX = 0.0, 25.0  # 速度上下限
K_ATTACK = 3      # 参与对抗优化的 NPC 数

# —— SVGD 搜索盒 —— #
DS_LIM = 25.0
DD_LIM = 4.5
DYAW_LIM = 20.0
MIN_SEP = 3.5

# —— 生成数量最低门槛（实际生成太少直接回滚） —— #
MIN_NPC = 20

# —— ART 步长 —— #
SBART_STEPS_BASE = {
    "break": 10,
    "accelerate": 4,
    "left_change_acc": 10,
    "left_change_dec": 10,
    "right_change_acc": 10,
    "right_change_dec": 10,
}
DEFAULT_STEPS = 6


def to_state(actor: carla.Actor):
    tf = actor.get_transform()
    vel = actor.get_velocity()
    v = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
    return np.array([tf.location.x, tf.location.y,
                     math.radians(tf.rotation.yaw), v], dtype=np.float32)
# ====================== 几何/辅助 ======================
def constant_velocity_rollout(x0_np, H, dt):
    x = torch.tensor(x0_np, dtype=torch.float32)
    xs = [x[None, :]]
    for _ in range(H):
        x = x.clone()
        x[0] = x[0] + x[3]*torch.cos(x[2]) * dt
        x[1] = x[1] + x[3]*torch.sin(x[2]) * dt
        # yaw, v 恒定
        xs.append(x[None, :])
    return torch.cat(xs, dim=0)  # [H+1, 4]

def _yaw_to_unit(yaw_deg: float):
    r = math.radians(yaw_deg)
    return math.cos(r), math.sin(r)

def ego_local_sd(ego_tf: carla.Transform, pt: carla.Location):
    dx = pt.x - ego_tf.location.x
    dy = pt.y - ego_tf.location.y
    cy, sy = _yaw_to_unit(ego_tf.rotation.yaw)
    s =  dx * cy + dy * sy
    d = -dx * sy + dy * cy
    return s, d

# ====================== Episode 记录 ======================
class EpisodeRecorder:
    def __init__(self, world_map):
        self.map = world_map
        self.frames = []  # [{"ego": {...}, "npcs": {id: {...}}}]

    @staticmethod
    def _vel_of(actor):
        v = actor.get_velocity()
        return (v.x, v.y, v.z)

    def log(self, ego: carla.Actor, vehicles: list):
        ego_tf = ego.get_transform()
        ego_vel = self._vel_of(ego)
        npcs = {}
        for v in vehicles:
            if v.id == ego.id:
                continue
            npcs[v.id] = {"tf": v.get_transform(), "vel": self._vel_of(v)}
        self.frames.append({"ego": {"tf": ego_tf, "vel": ego_vel}, "npcs": npcs})

# ====================== min-distance hazard 标签 ======================

def _closing_speed_at(ego_tf, ego_vel, npc_tf, npc_vel):
    rx = npc_tf.location.x - ego_tf.location.x
    ry = npc_tf.location.y - ego_tf.location.y
    rnorm = math.hypot(rx, ry) + 1e-6
    ux, uy = rx / rnorm, ry / rnorm
    vrelx = npc_vel[0] - ego_vel[0]
    vrely = npc_vel[1] - ego_vel[1]
    v_close = -(vrelx * ux + vrely * uy)  # 正值=在靠近
    return max(0.0, v_close)

def find_min_distance_window(frames, npc_id, win=2):
    d_min = float('inf'); t_star = -1
    for t in range(len(frames)):
        npcs = frames[t]["npcs"]
        if npc_id not in npcs:
            continue
        ego_tf = frames[t]["ego"]["tf"]; npc_tf = npcs[npc_id]["tf"]
        dx = npc_tf.location.x - ego_tf.location.x
        dy = npc_tf.location.y - ego_tf.location.y
        d = math.hypot(dx, dy)
        if d < d_min:
            d_min = d; t_star = t
    if t_star < 0:
        return -1, [], float('inf')
    i0 = max(0, t_star - win); i1 = min(len(frames) - 1, t_star + win)
    return t_star, list(range(i0, i1 + 1)), d_min

def hazard_from_min_distance(frames, world_map, npc_id,
                             D0=6.0, v0=0.5, sigma_v=1.0,
                             w_dist=0.9, w_close=0.7, use_heading=False, w_head=0.3, win=2):
    if not frames:
        return 0.0
    t_star, idxs, d_min = find_min_distance_window(frames, npc_id, win=win)
    if t_star < 0 or not idxs:
        return 0.0

    # 碰撞近似：任一窗口帧距离 <1.5m 判 1
    collided = False
    v_closes = []
    heads = []
    for t in idxs:
        ego_k = frames[t]["ego"]; npcs_k = frames[t]["npcs"]
        if npc_id not in npcs_k:
            continue
        npc_k = npcs_k[npc_id]
        dx = npc_k["tf"].location.x - ego_k["tf"].location.x
        dy = npc_k["tf"].location.y - ego_k["tf"].location.y
        if math.hypot(dx, dy) <= 1.5:
            collided = True
        v_closes.append(_closing_speed_at(ego_k["tf"], ego_k["vel"], npc_k["tf"], npc_k["vel"]))
        if use_heading:
            dyaw = npc_k["tf"].rotation.yaw - ego_k["tf"].rotation.yaw
            while dyaw >= 180: dyaw -= 360
            while dyaw < -180: dyaw += 360
            heads.append((1.0 + math.cos(math.radians(dyaw))) * 0.5)

    if collided:
        return 1.0

    s_dist = math.exp(-min(d_min, 40.0) / D0)
    v_bar = float(np.mean(v_closes)) if v_closes else 0.0
    s_close = 1.0 / (1.0 + math.exp(-(v_bar - v0) / max(sigma_v, 1e-6)))
    s_head = (float(np.mean(heads)) if (use_heading and heads) else 0.0)

    prod = 1.0
    for w, s in ((w_dist, s_dist), (w_close, s_close)):
        prod *= (1.0 - w * max(0.0, min(1.0, s)))
    if use_heading:
        prod *= (1.0 - w_head * max(0.0, min(1.0, s_head)))
    y_i = 1.0 - prod
    return max(0.0, min(1.0, y_i))

# ====================== 数据集与训练 ======================

def build_initial_pose_dataset_minDist(rec, world_map, surrogate_mlp,
                                       D0=6.0, v0=0.5, sigma_v=1.0,
                                       w_dist=0.9, w_close=0.7, use_heading=False, w_head=0.3, win=2):
    F = rec.frames
    if not F:
        return np.zeros((0, 8), np.float32), np.zeros((0,), np.float32), []
    t0 = 0
    ego_tf0 = F[t0]["ego"]["tf"]
    Xs, Ys, META = [], [], []
    for nid, npc in F[t0]["npcs"].items():
        npc_tf0 = npc["tf"]
        feats = surrogate_mlp._build_feats(world_map, ego_tf0, npc_tf0)
        y_i = hazard_from_min_distance(F, world_map, nid,
                                       D0=D0, v0=v0, sigma_v=sigma_v,
                                       w_dist=w_dist, w_close=w_close,
                                       use_heading=use_heading, w_head=w_head, win=win)
        Xs.append(feats); Ys.append(y_i)
        ds0, dd0 = ego_local_sd(ego_tf0, npc_tf0.location)
        dyaw0 = npc_tf0.rotation.yaw - ego_tf0.rotation.yaw
        while dyaw0 >= 180: dyaw0 -= 360
        while dyaw0 < -180: dyaw0 += 360
        META.append({"npc_id": nid, "ds0": float(ds0), "dd0": float(dd0), "dyaw0": float(dyaw0)})
    if not Xs:
        return np.zeros((0, 8), np.float32), np.zeros((0,), np.float32), []
    return np.asarray(Xs, np.float32), np.asarray(Ys, np.float32), META


def train_mlp_initial_pose_minDist(surrogate_mlp: NPCHazardMLPSurrogate,
                                   world_map, rec: EpisodeRecorder,
                                   epochs=1, batch=256, lr=1e-3,
                                   D0=6.0, v0=0.5, sigma_v=1.0,
                                   w_dist=0.9, w_close=0.7, use_heading=False, w_head=0.3, win=2):
    X, Y, META = build_initial_pose_dataset_minDist(rec, world_map, surrogate_mlp,
                                                    D0=D0, v0=v0, sigma_v=sigma_v,
                                                    w_dist=w_dist, w_close=w_close,
                                                    use_heading=use_heading, w_head=w_head, win=win)
    if X.shape[0] == 0:
        return 0, 0.0, 0, (X, Y, META)
    device = surrogate_mlp.device
    model = surrogate_mlp.model
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = torch.nn.BCELoss(reduction='mean')

    idx = np.arange(len(Y))
    total_steps, running = 0, 0.0
    for _ in range(epochs):
        np.random.shuffle(idx)
        for i0 in range(0, len(idx), batch):
            sel = idx[i0:i0 + batch]
            x_t = torch.tensor(X[sel], device=device)
            y_t = torch.tensor(Y[sel], device=device)
            pred = model(x_t).squeeze(-1)
            loss = bce(pred, y_t)
            opt.zero_grad(); loss.backward(); opt.step()
            total_steps += 1; running += float(loss.item())
    model.eval()
    avg_loss = running / max(total_steps, 1)
    return total_steps, avg_loss, len(Y), (X, Y, META)


def push_nearmiss_initial_to_replay(replay: NearMissReplay, dataset_tuple, min_tau=NEARMISS_SAVE_TAU, top_p=0.3, max_k=64):
    X, Y, META = dataset_tuple
    if len(Y) == 0:
        return 0
    order = np.argsort(-Y)
    m = max(1, int(len(order) * top_p))
    cnt = 0
    for k in order[:m]:
        if float(Y[k]) < min_tau:
            break
        replay.add_many([{
            "ds": float(META[k]["ds0"]),
            "dd": float(META[k]["dd0"]),
            "dyaw": float(META[k]["dyaw0"]),
            "F": float(Y[k])
        }])
        cnt += 1
        if cnt >= max_k:
            break
    return cnt
# ====================== 可微 Kinematic Bicycle ======================
class KinematicBicycle(nn.Module):
    def __init__(self, L=2.7):
        super().__init__()
        self.L = L

    def forward(self, x0, U, dt):
        """
        x0: [4]=(x,y,yaw,v), U:[H,2]=(a,delta)
        return X:[H+1,4]
        """
        X = [x0[None, :]]
        x = x0
        for t in range(U.shape[0]):
            a = torch.clamp(U[t, 0], -ACC_LIMIT, ACC_LIMIT)
            delta = torch.clamp(U[t, 1], -STEER_LIMIT, STEER_LIMIT)
            v = torch.clamp(x[3], V_MIN, V_MAX)
            x_next = torch.zeros_like(x)
            x_next[0] = x[0] + v*torch.cos(x[2]) * dt
            x_next[1] = x[1] + v*torch.sin(x[2]) * dt
            x_next[2] = x[2] + (v/self.L)*torch.tan(delta) * dt
            x_next[3] = torch.clamp(v + a*dt, V_MIN, V_MAX)
            X.append(x_next[None, :])
            x = x_next
        return torch.cat(X, dim=0)

def softmin_distance(npc_traj, ego_traj, tau=1.0):
    diff = npc_traj[:, :2] - ego_traj[:, :2]
    d = torch.sqrt(torch.sum(diff*diff, dim=1) + 1e-9)  # [H+1]
    return -tau * torch.logsumexp(-d / max(tau, 1e-6), dim=0)

def control_regularizer(U):
    loss_u = (U**2).mean()
    dU = U[1:] - U[:-1]
    loss_du = (dU**2).mean()
    return W_U*loss_u + W_DU*loss_du
# ====================== KING 规划器 ======================
class KingPlanner:
    def __init__(self, horizon=H, dt=DT_PLAN, n_opt=N_OPT, lr=LR, device="cpu"):
        self.horizon = horizon
        self.dt = dt
        self.n_opt = n_opt
        self.device = torch.device(device)
        self.model = KinematicBicycle().to(self.device)
        self.prev_plan = {}  # veh_id -> tensor[H,2]

    def plan_once(self, ego_x0, npc_x0, veh_id):
        ego_traj = constant_velocity_rollout(ego_x0, self.horizon, self.dt).to(self.device)
        npc_x0_t = torch.tensor(npc_x0, dtype=torch.float32, device=self.device)

        # warm-start
        if veh_id in self.prev_plan and self.prev_plan[veh_id].shape[0] == self.horizon:
            U0 = torch.zeros((self.horizon, 2), dtype=torch.float32, device=self.device)
            U0[:-1] = self.prev_plan[veh_id][1:].detach()
        else:
            U0 = torch.zeros((self.horizon, 2), dtype=torch.float32, device=self.device)

        U = nn.Parameter(U0.clone())
        opt = torch.optim.Adam([U], lr=LR)

        for _ in range(self.n_opt):
            opt.zero_grad()
            npc_traj = self.model(npc_x0_t, U, self.dt)
            J = softmin_distance(npc_traj, ego_traj, tau=TAU_SOFTMIN) + control_regularizer(U)
            J.backward()
            opt.step()

        self.prev_plan[veh_id] = U.detach()
        u0 = self.prev_plan[veh_id][0].detach().cpu().numpy()
        a = float(np.clip(u0[0], -ACC_LIMIT, ACC_LIMIT))
        delta = float(np.clip(u0[1], -STEER_LIMIT, STEER_LIMIT))
        return a, delta

def apply_king_control(actor: carla.Vehicle, a: float, delta: float):
    steer_cmd = float(np.clip(delta / STEER_LIMIT, -1.0, 1.0))
    if a >= 0:
        throttle = float(np.clip(a / ACC_LIMIT, 0.0, 1.0))
        brake = 0.0
    else:
        throttle = 0.0
        brake = float(np.clip(-a / ACC_LIMIT, 0.0, 1.0))
    ctrl = carla.VehicleControl(throttle=throttle, brake=brake, steer=steer_cmd)
    actor.apply_control(ctrl)

# ====================== JSON 序列化工具 ======================

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


def append_jsonl(path: str, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ====================== 主程序 ======================

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    COLLIDED_JSONL = "collided_scenarios.jsonl"

    # 交通灯全绿
    # for actor in world.get_actors():
    #     if isinstance(actor, carla.TrafficLight):
    #         actor.set_state(carla.TrafficLightState.Green)
    #         actor.freeze(False)

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(42)

    external_ads = True
    demo = MultiVehicleDemo(world, external_ads)
    world_map = world.get_map()

    surrogate = NPCHazardMLPSurrogate(ckpt_path="mlp_frozen.pt")
    replay = NearMissReplay(capacity=50000)

    art = ARTSelectorScenario()

    camera = None
    image_queue = None

    Test_buget = 400
    number_game = 1
    side_total = 0
    timeout_total = 0
    red_light_total = 0
    obj_collison_total = 0
    cross_solid_line_total = 0
    GA = CombinedGA(carla_map=world_map)
    GA.sample_initial_population()
    scenario_confs = GA.population

    gen_records = []  # {idx, peak_F, collided, pos_info}
    planner = KingPlanner(device="cuda" if torch.cuda.is_available() else "cpu")
    while number_game <= Test_buget:
        abnormal_case = False
        idx_in_pop = (number_game - 1) % population_size
        scenario_conf = scenario_confs[idx_in_pop]["position_info"]

        success = demo.setup_vehicles_with_collision(scenario_conf)
        if not success:
            print("[ERROR] 车辆生成失败，跳过（不计入有效回合）。")
            scenario_confs[idx_in_pop] = GA.resample()
            continue

        actual_n = len(demo.vehicles)
        demo.vehicle_num = actual_n
        if actual_n < MIN_NPC:
            print(f"[WARN] only {actual_n} NPC spawned (<{MIN_NPC}), 回滚重采样。")
            demo.destroy_all()
            scenario_confs[idx_in_pop] = GA.resample()
            continue

        print(f"[INFO] 请求 {scenario_conf.get('vehicle_num','?')}，实际生成 {actual_n} (有效回合序号 {number_game})")

        controllers = []
        controller_by_actor_id = {}
        for i, v in enumerate(demo.vehicles):
            ctrl = demo.get_controller(i)
            controllers.append(ctrl)
            controller_by_actor_id[v.id] = ctrl

        if not demo.external_ads:
            demo.ego_vehicle.set_autopilot(True, tm.get_port())
            tm.vehicle_percentage_speed_difference(demo.ego_vehicle, 5)

        # 所有 NPC 上 TM
        ego_id = demo.ego_vehicle.id if demo.ego_vehicle is not None else -1
        for v in demo.vehicles:
            if v.id != ego_id:
                v.set_autopilot(True, tm.get_port())
                tm.vehicle_percentage_speed_difference(v, random.randint(-10, 20))
                tm.ignore_signs_percentage(v, 0)
                tm.ignore_lights_percentage(v, 0)
                tm.ignore_walkers_percentage(v, 0)
                tm.distance_to_leading_vehicle(v, 2.5)

        # 录像
        if RECORDING:
            camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '640')
            camera_bp.set_attribute('image_size_y', '360')
            camera_bp.set_attribute('fov', '90')
            spectator = world.get_spectator()
            image_queue = Queue()

            def camera_callback(image, image_queue):
                array = np.frombuffer(image.raw_data, dtype=np.uint8)
                array = array.reshape((image.height, image.width, 4))
                image_queue.put(array)

            camera = world.spawn_actor(
                camera_bp,
                carla.Transform(spectator.get_transform().location, spectator.get_transform().rotation)
            )
            camera.listen(lambda data: camera_callback(data, image_queue))
            save_dir = f"recording/episode_{number_game}"
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = f"recording/episode_{number_game}"
            camera = None
            image_queue = None

        wall_start = time.monotonic()
        progress_anchor_loc = None
        progress_anchor_t = wall_start
        timeout_cnt = 0
        start_loc = None

        # ART 运行期状态
        override_state = {}   # veh_id -> {"steps": int, "act": str}

        # === 选择对抗车（按前向距离 |s| 最小） ===
        ego_tf0 = demo.ego_vehicle.get_transform()
        scored = []
        for v in demo.vehicles:
            if v.id == ego_id: continue
            s, _ = ego_local_sd(ego_tf0, v.get_transform().location)
            scored.append((abs(s), v))
        scored.sort(key=lambda x: x[0])
        attack_list = [v for _, v in scored[:min(K_ATTACK, len(scored))]]

        # 回放记录器
        rec = EpisodeRecorder(world_map)

        for mod in demo.modules:
            demo.enable_module(mod)

        # ========== 主循环 ==========
        for step in range(50000):
            world.wait_for_tick()

            if time.monotonic() - wall_start > EPISODE_MAX_SECONDS:
                print(f"[EARLY-EXIT] episode wall-clock > {EPISODE_MAX_SECONDS}s, stop.")
                break

            # 启动阶段
            if step < STARTUP_STEPS:
                ego_vel = demo.ego_vehicle.get_velocity()
                speed_ego = math.sqrt(ego_vel.x ** 2 + ego_vel.y ** 2 + ego_vel.z ** 2)
                if speed_ego > 5:
                    print('[WARN] EGO 启动阶段速度异常偏高，结束该轮。')
                    abnormal_case = True
            if step == STARTUP_STEPS and demo.external_ads:
                start_loc = demo.ego_vehicle.get_location()
                progress_anchor_loc = start_loc
                progress_anchor_t = time.monotonic()
                demo.set_destination()

            if step > STARTUP_STEPS:
                # TM 续权（排除正在接管的 NPC）
                if step % KEEP_ALIVE_PERIOD == 0:
                    for v in demo.vehicles:
                        if v.id == ego_id or v.id in override_state:
                            continue
                        v.set_autopilot(True, tm.get_port())
                        tm.vehicle_percentage_speed_difference(v, random.randint(-10, 20))
                        tm.ignore_signs_percentage(v, 0)
                        tm.ignore_lights_percentage(v, 0)
                        tm.ignore_walkers_percentage(v, 0)
                        tm.distance_to_leading_vehicle(v, 2.5)

                # 执行接管中的车辆
                dead_ids = []
                for vid, st in list(override_state.items()):
                    v = world.get_actor(vid)
                    if v is None:
                        dead_ids.append(vid); continue
                    ctrl = controller_by_actor_id.get(vid)
                    if st["steps"] > 0 and ctrl is not None:
                        action_trans(ctrl, st["act"])
                        st["steps"] -= 1
                    else:
                        v.set_autopilot(True, tm.get_port())
                        dead_ids.append(vid)
                for vid in dead_ids:
                    override_state.pop(vid, None)

                # 环境统计（若碰撞，立即早退）
                signals_list, ego_collision, all_collision, cross_solid_line, red_light = demo.tick()
                if ego_collision or all_collision:
                    print("[EARLY-EXIT] collision detected. Ending episode immediately.")
                    break

                # 非阻塞录像
                if RECORDING and image_queue is not None:
                    try:
                        frame = image_queue.get_nowait()
                        cv2.imwrite(os.path.join(save_dir, f"frame_{step}.png"), frame)
                        spectator = world.get_spectator()
                        camera.set_transform(spectator.get_transform())
                    except Exception:
                        pass

                # 无进展超时
                ego_loc = demo.ego_vehicle.get_location()
                if progress_anchor_loc is not None:
                    moved = math.hypot(ego_loc.x - progress_anchor_loc.x, ego_loc.y - progress_anchor_loc.y)
                    if moved >= PROGRESS_THRESH:
                        progress_anchor_loc = ego_loc
                        progress_anchor_t = time.monotonic()
                    elif time.monotonic() - progress_anchor_t > NO_PROGRESS_SECONDS:
                        print(f"[EARLY-EXIT] no progress for {NO_PROGRESS_SECONDS}s (Δ={moved:.2f}m).")
                        timeout_cnt = 1
                        break

                if demo.ego_destination is not None:
                    near_dest, pass_dest = has_passed_destination(demo.ego_vehicle, demo.ego_destination, world_map)
                    if step != 0 and demo.external_ads and near_dest:
                        print('Arrive, episode end')
                        break
                    else:
                        if pass_dest:
                            for mod in demo.modules: demo.enable_module(mod)
                            print('pass destination error'); abnormal_case = True

                # 摄像机跟随
                spectator = world.get_spectator()
                trans = demo.ego_vehicle.get_transform()
                loc = trans.location
                yaw_deg = trans.rotation.yaw
                yaw_rad = math.radians(yaw_deg)
                offset_x = -10.0; offset_z = 5.0
                cam_x = loc.x + offset_x * math.cos(yaw_rad)
                cam_y = loc.y + offset_x * math.sin(yaw_rad)
                cam_z = loc.z + offset_z
                spectator.set_transform(carla.Transform(
                    carla.Location(cam_x, cam_y, cam_z),
                    carla.Rotation(pitch=-20, yaw=yaw_deg)
                ))

                # 记录一帧
                rec.log(demo.ego_vehicle, demo.vehicles)

                # === KING：每 REPLAN_STRIDE tick 规划一次 ===
                if (step % REPLAN_STRIDE) == 0 and attack_list:
                    ego_x0 = to_state(demo.ego_vehicle)
                    for v in attack_list:
                        try:
                            a, delta = planner.plan_once(ego_x0, to_state(v), v.id)
                            apply_king_control(v, a, delta)
                        except Exception as e:
                            print(f"[WARN] 规划异常 veh {v.id}: {e}")
                            v.set_autopilot(True, tm.get_port())

        # episode 尾：清空 apollo 的 prediction/planning 缓存（若有）
        try:
            apollo_clear_prediction_planning(times=3, interval=0.05)
        except Exception as e:
            print(f"[WARN] apollo_clear_prediction_planning failed: {e}")

        for _ in range(300):
            world.wait_for_tick()

        if RECORDING and camera is not None:
            try:
                camera.stop(); camera.destroy()
            except Exception:
                pass

        # 结束检查
        end_loc = demo.ego_vehicle.get_location()
        if start_loc is not None:
            distance_to_start = math.dist([start_loc.x, start_loc.y], [end_loc.x, end_loc.y])
        else:
            distance_to_start = 0.0
        print('Moved distance: ', distance_to_start)
        if distance_to_start < 1:
            print("[WARNING] EGO 起点与终点距离 < 1m，判定异常，不计入本代。")
            abnormal_case = True

        # 统计 + GA 演化（仅对有效回合）
        if not abnormal_case:
            side_total += demo.side_collision_count_vehicle
            obj_collison_total = demo.collision_count_obj
            timeout_total += timeout_cnt
            red_light_total += demo.ego_run_red_light
            cross_solid_line_total += demo.ego_cross_solid_line

            # === 回合结束：训练 MLP（起点 -> 回合级 min-dist hazard） ===
            steps_done, avg_loss, n_samples, dataset_tuple = train_mlp_initial_pose_minDist(
                surrogate, world_map, rec,
                epochs=2, batch=256, lr=1e-3,
                D0=6.0, v0=0.5, sigma_v=1.0,
                w_dist=0.9, w_close=0.7, use_heading=False, w_head=0.3, win=2
            )
            if hasattr(surrogate, 'ema_update'):
                try:
                    surrogate.ema_update(tau=0.05)
                except Exception:
                    pass

            # 以该回合的最高标签作“峰值”用于代内排序
            peak_y = float(dataset_tuple[1].max()) if n_samples > 0 else 0.0
            print(f"[EP-END] train-steps={steps_done}, loss≈{avg_loss:.4f}, n={n_samples}, peak_F={peak_y:.3f}")

            # 记录（含是否发生车辆侧碰）
            jsonable_conf = _to_jsonable(scenario_conf)
            collided_record = {
                "ts": datetime.now().isoformat(),
                "episode": number_game,
                "side_collision": demo.side_collision_count_vehicle,
                "object collision": demo.collision_count_obj,
                "timeout": timeout_cnt,
                "red_light": demo.ego_run_red_light,
                "cross_solid": demo.ego_cross_solid_line,
                "scenario_conf": jsonable_conf
            }
            append_jsonl(COLLIDED_JSONL, collided_record)
            print(f"[SAVED] scenario appended to {COLLIDED_JSONL}")
            if demo.side_collision_count_vehicle == 0 and RECORDING and os.path.isdir(save_dir):
                shutil.rmtree(save_dir)

            # —— 整代结束：SVGD 注入 + 下一代 ——
            if (number_game % population_size) == 0:
                # 把“本代历史 + 当前回合”一起排名，取 TOP_CASES 做 SVGD
                ranked = sorted(
                    gen_records + [{"idx": number_game, "peak_F": peak_y, "collided": bool(demo.side_collision_count_vehicle), "pos_info": scenario_conf}],
                    key=lambda r: r["peak_F"], reverse=True
                )
                top_cases = ranked[:min(SVGD_TOP_CASES, len(ranked))]

                new_population = []
                for r in top_cases:
                    pos_info = r["pos_info"]
                    # —— 这里实际调用 RuntimeNPCSVGD —— #
                    svgd_runner = RuntimeNPCSVGD(
                        world_map,
                        surrogate=surrogate,
                        top_k=5,                     # 不必等于场景内 NPC 数；内部会 min(N, top_k)
                        steps=SVGD_STEPS_PER_CASE,
                        epsilon=SVGD_EPS,
                        beta=SVGD_BETA,
                        grad_eps=SVGD_GRAD_EPS,
                        ds_lim=DS_LIM, dd_lim=DD_LIM, dyaw_lim=DYAW_LIM,
                        min_sep=MIN_SEP
                    )
                    svgd_runner.refine_position_info(pos_info)   # <--- 关键调用
                    new_population.append({"position_info": pos_info})
                print(f"{len(new_population)} scenario(s) refined by SVGD.")

                # 填满一代
                for _ in range(population_size - len(new_population)):
                    new_population.append(GA.resample())
                scenario_confs = new_population
                gen_records.clear()
                print("[GEN] end. New generation sampled.")
            else:
                gen_records.append({
                    "idx": number_game,
                    "peak_F": peak_y,
                    "collided": bool(demo.side_collision_count_vehicle),
                    "pos_info": scenario_conf
                })

            number_game += 1
        else:
            if RECORDING and os.path.isdir(save_dir):
                shutil.rmtree(save_dir)
            scenario_confs[idx_in_pop] = GA.resample()
            print(f"[INFO] 本轮异常，已回滚重采样（不计入有效回合）。")

        demo.destroy_all()

        keep_ids = {demo.ego_vehicle.id} if demo.ego_vehicle else set()
        rem_veh, rem_walk = purge_npcs(world, client, tm=tm, keep_actor_ids=keep_ids,
                                       include_walkers=True, hard_teleport=True)
        print(f"[PURGE] left vehicles={rem_veh}, walkers={rem_walk}")

    # ---- 结束统计 ----
    print('Side collision (ego×vehicle):', side_total)
    print('Ego vehicle object collision:', obj_collison_total)
    print('Time Out:', timeout_total)
    print('red light:', red_light_total)
    demo.destroy_all(); demo.close_connection()
    print("Cleanup done.")


if __name__ == "__main__":
    main()
