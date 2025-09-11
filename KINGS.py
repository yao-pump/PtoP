#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KING baseline（随机初始场景 + 可微自行车模型滚动优化）：
- 随机选 EGO + N 个 NPC 的初始 Transform（均来自 map.get_spawn_points）
- 前 K_ATTACK 个 NPC（按与 EGO 的前向距离 |s| 最近）作为“对抗车”，其余 NPC 由 Traffic Manager 控制
- 每 REPLAN_STRIDE tick：用可微 Kinematic Bicycle + 常速 EGO 预测，在短时域内优化对抗车控制序列 U=[a,delta]
- 目标：最小化与 EGO 的软最小距离（soft-min），并加控制幅度/平滑正则；仅执行第一步（MPC）

依赖你现有的：
  - world.MultiVehicleDemo（已支持 setup_vehicles_with_collision / 碰撞统计/闯红灯/压实线等）
  - utility.purge_npcs（回合末清场）
  - utility.has_passed_destination（是否到达）
"""

import os
import cv2
import math
import time
import random
import json
from datetime import datetime
from queue import Queue

import numpy as np
import torch
import torch.nn as nn
import carla

# ---- 可选依赖：若没有对应函数则提供空实现 ----
try:
    from utility import has_passed_destination, apollo_clear_prediction_planning, purge_npcs
except Exception:
    def has_passed_destination(*args, **kwargs): return (False, False)
    def apollo_clear_prediction_planning(*args, **kwargs): pass
    def purge_npcs(world, client, tm=None, keep_actor_ids=None, include_walkers=True, hard_teleport=True): return 0, 0

from world import MultiVehicleDemo

# ====================== 全局超参数 ======================
TIME_STEP = 0.05
STARTUP_STEPS = 500

# —— Traffic Manager —— #
TM_PORT = 8000
KEEP_ALIVE_PERIOD = 30  # TM 续权周期（tick）

# —— 录像 —— #
RECORDING = False

# —— 早退/兜底 —— #
EPISODE_MAX_SECONDS = 180.0
NO_PROGRESS_SECONDS = 15
PROGRESS_THRESH = 2.0

# —— 随机场景 —— #
NPC_NUM = 20      # 每回合 NPC 总数
MIN_NPC = 20      # 生成失败过多时跳回合
K_ATTACK = 3      # 参与对抗优化的 NPC 数

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

# —— 实验轮数 —— #
TEST_BUDGET = 400

# ====================== 几何/辅助 ======================
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

def to_state(actor: carla.Actor):
    tf = actor.get_transform()
    vel = actor.get_velocity()
    v = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
    return np.array([tf.location.x, tf.location.y,
                     math.radians(tf.rotation.yaw), v], dtype=np.float32)

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

# ====================== 随机场景生成 ======================
def random_scenario(world_map: carla.Map, npc_num=NPC_NUM):
    sps = list(world_map.get_spawn_points())
    random.shuffle(sps)
    if len(sps) < npc_num + 1:
        raise RuntimeError(f"spawn_points 不足：需要 {npc_num+1}，仅有 {len(sps)}")
    ego_tf = sps[0]
    surrounding = [{"transform": sps[i+1], "type": "car"} for i in range(npc_num)]
    return {"vehicle_num": npc_num, "ego_transform": ego_tf, "surrounding_info": surrounding}

# ====================== JSON 工具（记录场景） ======================
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
    world_map = world.get_map()
    COLLIDED_JSONL = "king_random_scenarios.jsonl"

    # 交通灯置绿（可选）
    for actor in world.get_actors():
        if isinstance(actor, carla.TrafficLight):
            actor.set_state(carla.TrafficLightState.Green)
            actor.freeze(True)

    tm = client.get_trafficmanager(TM_PORT)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(42)

    external_ads = True
    demo = MultiVehicleDemo(world, external_ads)

    camera = None
    image_queue = None

    number_game = 1
    side_total = 0
    timeout_total = 0
    red_light_total = 0
    obj_collision_total = 0
    cross_solid_total = 0

    planner = KingPlanner(device="cuda" if torch.cuda.is_available() else "cpu")

    while number_game <= TEST_BUDGET:
        abnormal_case = False

        # === 随机场景 ===
        scenario_conf = random_scenario(world_map, npc_num=NPC_NUM)

        # === 生成与碰撞传感器 ===
        success = demo.setup_vehicles_with_collision(scenario_conf)
        if not success:
            print("[ERROR] 车辆生成失败，跳过回合。")
            continue

        actual_n = len(demo.vehicles)
        demo.vehicle_num = actual_n
        if actual_n < MIN_NPC:
            print(f"[WARN] only {actual_n} NPC spawned (<{MIN_NPC}), 跳回合。")
            demo.destroy_all()
            continue

        # === TM 挂到非对抗车上 ===
        ego_id = demo.ego_vehicle.id if demo.ego_vehicle is not None else -1
        for v in demo.vehicles:
            if v.id != ego_id:
                v.set_autopilot(True, tm.get_port())
                tm.vehicle_percentage_speed_difference(v, random.randint(-10, 20))
                tm.ignore_signs_percentage(v, 0)
                tm.ignore_lights_percentage(v, 0)
                tm.ignore_walkers_percentage(v, 0)
                tm.distance_to_leading_vehicle(v, 2.5)

        # === 录像 ===
        if RECORDING:
            camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '1280')
            camera_bp.set_attribute('image_size_y', '720')
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
            save_dir = f"recording/king_episode_{number_game}"
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = f"recording/king_episode_{number_game}"
            camera = None
            image_queue = None

        # === 选择对抗车（按前向距离 |s| 最小） ===
        ego_tf0 = demo.ego_vehicle.get_transform()
        scored = []
        for v in demo.vehicles:
            if v.id == ego_id: continue
            s, _ = ego_local_sd(ego_tf0, v.get_transform().location)
            scored.append((abs(s), v))
        scored.sort(key=lambda x: x[0])
        attack_list = [v for _, v in scored[:min(K_ATTACK, len(scored))]]

        # 对抗车退出 TM
        for v in attack_list:
            v.set_autopilot(False)

        # === 回合状态 ===
        wall_start = time.monotonic()
        progress_anchor_loc = None
        progress_anchor_t = wall_start
        timeout_cnt = 0
        start_loc = None

        # === 启动各模块（若使用 Apollo） ===
        for mod in demo.modules:
            demo.enable_module(mod)

        # ========== 主循环 ==========
        for step in range(100000):
            world.wait_for_tick()

            if time.monotonic() - wall_start > EPISODE_MAX_SECONDS:
                print(f"[EARLY-EXIT] wall-clock > {EPISODE_MAX_SECONDS}s")
                break

            # 启动阶段
            if step < STARTUP_STEPS:
                ego_vel = demo.ego_vehicle.get_velocity()
                speed_ego = math.sqrt(ego_vel.x**2 + ego_vel.y**2 + ego_vel.z**2)
                if speed_ego > 5:
                    print('[WARN] 启动阶段 EGO 速度异常，结束该轮。')
                    abnormal_case = True
            if step == STARTUP_STEPS and demo.external_ads:
                start_loc = demo.ego_vehicle.get_location()
                progress_anchor_loc = start_loc
                progress_anchor_t = time.monotonic()
                demo.set_destination()

            if step > STARTUP_STEPS:
                # TM 续权（非对抗车）
                if step % KEEP_ALIVE_PERIOD == 0:
                    for v in demo.vehicles:
                        if v.id == ego_id or v in attack_list:
                            continue
                        v.set_autopilot(True, tm.get_port())

                # tick & 早退
                signals_list, ego_collision, all_collision, cross_solid_line, red_light = demo.tick()
                if ego_collision or all_collision:
                    print("[EARLY-EXIT] 碰撞，结束回合。")
                    break

                # 录像帧
                if RECORDING and image_queue is not None:
                    try:
                        frame = image_queue.get_nowait()
                        cv2.imwrite(os.path.join(save_dir, f"frame_{step}.png"), frame)
                        spectator = world.get_spectator()
                        camera.set_transform(spectator.get_transform())
                    except Exception:
                        pass

                # 无进展兜底
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

                # 到达检测（若用 Apollo 路由）
                if demo.ego_destination is not None:
                    near_dest, pass_dest = has_passed_destination(demo.ego_vehicle, demo.ego_destination, world_map)
                    if step != 0 and demo.external_ads and near_dest:
                        print('Arrive, episode end')
                        break
                    elif pass_dest:
                        for mod in demo.modules: demo.enable_module(mod)
                        print('pass destination error'); abnormal_case = True

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

        # episode 尾：Apollo 清缓存（可选）
        try:
            apollo_clear_prediction_planning(times=3, interval=0.05)
        except Exception:
            pass

        # 额外 tick 稳定一下
        for _ in range(60):
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
            print("[WARNING] EGO 起点与终点距离 < 1m，判定异常，不计入统计。")
            abnormal_case = True

        # 统计 & 记录
        if not abnormal_case:
            side_total += demo.side_collision_count_vehicle
            obj_collision_total += demo.collision_count_obj
            timeout_total += timeout_cnt
            red_light_total += int(demo.ego_run_red_light)
            cross_solid_total += int(demo.ego_cross_solid_line)

            # 记录场景（含指标）
            jsonable_conf = _to_jsonable(scenario_conf)
            record = {
                "ts": datetime.now().isoformat(),
                "episode": number_game,
                "side_collision": int(demo.side_collision_count_vehicle),
                "object_collision": int(demo.collision_count_obj),
                "timeout": int(timeout_cnt),
                "red_light": int(demo.ego_run_red_light),
                "cross_solid": int(demo.ego_cross_solid_line),
                "scenario_conf": jsonable_conf
            }
            append_jsonl(COLLIDED_JSONL, record)
            print(f"[SAVED] scenario appended to {COLLIDED_JSONL}")

            # 若无碰撞且录制目录存在，可删除
            if demo.side_collision_count_vehicle == 0 and RECORDING and os.path.isdir(save_dir):
                try:
                    import shutil; shutil.rmtree(save_dir)
                except Exception:
                    pass

            number_game += 1
        else:
            # 异常回合清理录像
            if RECORDING and os.path.isdir(save_dir):
                try:
                    import shutil; shutil.rmtree(save_dir)
                except Exception:
                    pass
            print(f"[INFO] 本轮异常，已跳过统计。")

        # 清场
        demo.destroy_all()
        keep_ids = {demo.ego_vehicle.id} if demo.ego_vehicle else set()
        rem_veh, rem_walk = purge_npcs(world, client, tm=tm, keep_actor_ids=keep_ids,
                                       include_walkers=True, hard_teleport=True)
        print(f"[PURGE] left vehicles={rem_veh}, walkers={rem_walk}")

    # ---- 汇总 ----
    print('=== KING baseline summary ===')
    print('Side collision (ego×vehicle):', side_total)
    print('Ego object collision:', obj_collision_total)
    print('Timeouts:', timeout_total)
    print('Red lights:', red_light_total)
    print('Cross solid lines:', cross_solid_total)

    demo.destroy_all(); demo.close_connection()
    print("Cleanup done.")

if __name__ == "__main__":
    main()
