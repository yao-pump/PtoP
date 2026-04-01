#!/usr/bin/env python3
# -*- coding: utf-8 -*-
“””
Main program (fully runnable skeleton, supporting dual-object “vehicle + pedestrian” control, with pedestrian NaN command fix):
- GA sampling + end-of-generation SVGD refinement (for NPC initial (ds, dd, dyaw))
- ART online NPC manipulation (adaptive strategy + combined actions)
- MLP surrogate: input = initial pose features, label = hazard at the closest NPC-EGO distance moment within the episode (min-distance scheme)
- Online fine-tuning 1-2 epochs per episode; optional EMA
- Added: WalkerPlanner (adversarial pedestrian control, integrator + soft minimum distance)
- Fixed: pedestrian control issued every tick & fallback sanitization, gradient/parameter clipping to prevent NaN propagation
“””

import os
import cv2
import math
import time
import shutil
import random
import logging
import numpy as np
from queue import Queue
import torch
import torch.nn as nn
import carla

from ptop.utils.utility import (
    has_passed_destination, action_trans, apollo_clear_prediction_planning, purge_npcs, _to_jsonable, append_jsonl
)
from ptop.utils.geometry import yaw_to_unit, ego_local_sd

logger = logging.getLogger(__name__)

from ptop.optimization.surrogate_mlp import NPCHazardMLPSurrogate
from ptop.core.world import MultiVehicleDemo
from ptop.optimization.seed_generator import seed_generator

from ptop.optimization.svgd_runtime import RuntimeNPCSVGD


K_ATTACK = 3      # Number of adversarial NPCs (up to K each for vehicles and pedestrians)

# —— Vehicle planner parameters —— #
REPLAN_STRIDE = 5
H = 25
DT_PLAN = 0.10
N_OPT = 30
LR = 5e-2
TAU_SOFTMIN = 1.0
ACC_LIMIT = 3.0
STEER_LIMIT = 0.6
W_U = 1e-3
W_DU = 5e-3
V_MIN, V_MAX = 0.0, 25.0

# —— Pedestrian planner parameters —— #
V_W_MAX = 2.2
W_W = 1e-3
W_DW = 5e-3
LR_WALKER = 5e-3          # FIX: more stable learning rate
U_CLAMP_ABS = 4.0         # FIX: clamp U after each step to prevent explosion
GRAD_CLIP_NORM = 10.0     # FIX: gradient clipping

# ====================== Geometry / Utilities ======================

def to_state(actor: carla.Actor):
    tf = actor.get_transform()
    vel = actor.get_velocity()
    v = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
    return np.array([tf.location.x, tf.location.y,
                     math.radians(tf.rotation.yaw), v], dtype=np.float32)

def to_state_walker(actor: carla.Actor):
    """Returns (x, y, yaw, v) for a pedestrian as well; yaw is approximated by velocity direction or facing."""
    tf = actor.get_transform()
    vel = actor.get_velocity()
    vx, vy, vz = vel.x, vel.y, vel.z
    v = math.sqrt(vx*vx + vy*vy + vz*vz)
    if abs(vx) + abs(vy) > 1e-6:
        yaw = math.atan2(vy, vx)
    else:
        yaw = math.radians(tf.rotation.yaw)
    return np.array([tf.location.x, tf.location.y, yaw, v], dtype=np.float32)

def constant_velocity_rollout(x0_np, H, dt):
    x = torch.tensor(x0_np, dtype=torch.float32)
    xs = [x[None, :]]
    for _ in range(H):
        x = x.clone()
        x[0] = x[0] + x[3]*torch.cos(x[2]) * dt
        x[1] = x[1] + x[3]*torch.sin(x[2]) * dt
        xs.append(x[None, :])
    return torch.cat(xs, dim=0)  # [H+1, 4]

# ====================== Differentiable Kinematic Bicycle (Vehicle) ======================
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

# ====================== Differentiable Integrator (Pedestrian) ======================
class WalkerIntegrator(nn.Module):
    def __init__(self, vmax=V_W_MAX):
        super().__init__()
        self.vmax = vmax

    def forward(self, x0, U, dt):
        """
        x0: [4]=(x,y,theta,v)
        U : [H,2] raw trainable parameters (unbounded), internally squashed via tanh to [-1,1] then multiplied by vmax to get (vx,vy)
        return X:[H+1,4] (only x,y are updated; theta comes from velocity direction; v is speed magnitude)
        """
        X = [x0[None, :]]
        x = x0
        for t in range(U.shape[0]):
            uv = torch.tanh(U[t]) * self.vmax  # (vx, vy)
            vx, vy = uv[0], uv[1]
            x_next = torch.zeros_like(x)
            x_next[0] = x[0] + vx * dt
            x_next[1] = x[1] + vy * dt
            theta = torch.atan2(vy, vx + 1e-9)
            v = torch.sqrt(vx*vx + vy*vy + 1e-12)
            x_next[2] = theta
            x_next[3] = torch.clamp(v, 0.0, self.vmax)
            X.append(x_next[None, :])
            x = x_next
        return torch.cat(X, dim=0)

# ====================== Shared Cost Functions ======================
def softmin_distance(npc_traj, ego_traj, tau=1.0):
    diff = npc_traj[:, :2] - ego_traj[:, :2]
    d = torch.sqrt(torch.sum(diff*diff, dim=1) + 1e-9)  # [H+1]
    return -tau * torch.logsumexp(-d / max(tau, 1e-6), dim=0)

def control_regularizer(U):
    loss_u = (U**2).mean()
    dU = U[1:] - U[:-1]
    loss_du = (dU**2).mean()
    return W_U*loss_u + W_DU*loss_du

def control_regularizer_walker(U):
    loss_u = (torch.tanh(U)**2).mean()
    dU = U[1:] - U[:-1]
    loss_du = (torch.tanh(dU)**2).mean()
    return W_W*loss_u + W_DW*loss_du

# ====================== Vehicle Planner ======================
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
            if not torch.isfinite(J):
                J = (U**2).mean()  # Prevent NaN
            J.backward()
            torch.nn.utils.clip_grad_norm_([U], GRAD_CLIP_NORM)  # Clip even single parameter
            opt.step()
            with torch.no_grad():
                U.data.clamp_(-U_CLAMP_ABS, U_CLAMP_ABS)

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

# ====================== Pedestrian Planner ======================
class WalkerPlanner:
    def __init__(self, horizon=H, dt=DT_PLAN, n_opt=N_OPT, lr=LR_WALKER, device="cpu", vmax=V_W_MAX):
        self.horizon = horizon
        self.dt = dt
        self.n_opt = n_opt
        self.device = torch.device(device)
        self.model = WalkerIntegrator(vmax=vmax).to(self.device)
        self.prev_plan = {}  # walker_id -> tensor[H,2]

        self.lr = lr

    def plan_once(self, ego_x0, walker_x0, walker_id):
        ego_traj = constant_velocity_rollout(ego_x0, self.horizon, self.dt).to(self.device)
        w_x0_t = torch.tensor(walker_x0, dtype=torch.float32, device=self.device)

        if walker_id in self.prev_plan and self.prev_plan[walker_id].shape[0] == self.horizon:
            U0 = torch.zeros((self.horizon, 2), dtype=torch.float32, device=self.device)
            U0[:-1] = self.prev_plan[walker_id][1:].detach()
        else:
            U0 = torch.zeros((self.horizon, 2), dtype=torch.float32, device=self.device)

        U = nn.Parameter(U0.clone())
        opt = torch.optim.Adam([U], lr=self.lr)

        for _ in range(self.n_opt):
            opt.zero_grad()
            traj = self.model(w_x0_t, U, self.dt)
            J = softmin_distance(traj, ego_traj, tau=TAU_SOFTMIN) + control_regularizer_walker(U)
            if not torch.isfinite(J):
                J = (torch.tanh(U)**2).mean()  # Prevent NaN
            J.backward()
            torch.nn.utils.clip_grad_norm_([U], GRAD_CLIP_NORM)         # FIX: gradient clipping
            opt.step()
            with torch.no_grad():
                if not torch.isfinite(U).all():
                    U.data.zero_()                                      # FIX: immediate rollback
                U.data.clamp_(-U_CLAMP_ABS, U_CLAMP_ABS)                # FIX: clamp

        self.prev_plan[walker_id] = U.detach()
        with torch.no_grad():
            u0 = torch.tanh(self.prev_plan[walker_id][0]).cpu().numpy() * V_W_MAX
        vx, vy = float(u0[0]), float(u0[1])
        # Return values may still need sanitization in the outer loop (see main loop)
        return vx, vy

def _sanitize_vec_towards_ego(walker: carla.Actor, ego: carla.Actor, wanted_speed=1.4):
    wl = walker.get_transform().location
    el = ego.get_location()
    dx, dy = (el.x - wl.x), (el.y - wl.y)
    n = math.hypot(dx, dy)
    if n < 1e-6:
        return 0.0, 0.0
    vx, vy = (dx / n) * wanted_speed, (dy / n) * wanted_speed
    return vx, vy

def _clean_v(vx, vy, walker: carla.Actor, ego: carla.Actor):
    """FIX: unified sanitization: non-finite / too large / too small => safe velocity vector pointing toward EGO"""
    if (not math.isfinite(vx)) or (not math.isfinite(vy)) or abs(vx) > 1e3 or abs(vy) > 1e3 or (abs(vx)+abs(vy) < 1e-3):
        return _sanitize_vec_towards_ego(walker, ego, wanted_speed=min(1.6, V_W_MAX))
    # Clamp (extra safety)
    sp = math.hypot(vx, vy)
    if sp > V_W_MAX + 1e-6:
        vx, vy = vx / sp * V_W_MAX, vy / sp * V_W_MAX
    return vx, vy

def apply_walker_control(actor: carla.Actor, vx: float, vy: float):
    # If still not clean, stop immediately
    if not (math.isfinite(vx) and math.isfinite(vy)):
        vx, vy = 0.0, 0.0
    speed = float(np.hypot(vx, vy))
    if speed < 1e-3:
        direction = carla.Vector3D(0.0, 0.0, 0.0)
        speed_cmd = 0.0
    else:
        dir_x = vx / speed
        dir_y = vy / speed
        direction = carla.Vector3D(dir_x, dir_y, 0.0)
        speed_cmd = float(np.clip(speed, 0.0, V_W_MAX))
    ctrl = carla.WalkerControl(direction=direction, speed=speed_cmd)
    actor.apply_control(ctrl)

def try_stop_walker_ai(demo: MultiVehicleDemo, walker: carla.Actor):
    """Attempt to stop the pedestrian AI controller, compatible with multiple wrapper structures."""
    try:
        if hasattr(demo, "walker_controller_by_id"):
            ctrl = getattr(demo, "walker_controller_by_id").get(walker.id, None)
            if ctrl:
                try: ctrl.stop()
                except Exception: pass
                return
        if hasattr(demo, "walker_controllers"):
            wcs = getattr(demo, "walker_controllers")
            for item in wcs:
                ctrl, w = None, None
                try:
                    if isinstance(item, tuple) and len(item) == 2:
                        w, ctrl = item
                    else:
                        ctrl = item
                        w = getattr(ctrl, 'actor', None) or getattr(ctrl, 'parent', None)
                        if w is None and hasattr(ctrl, 'get_actor'):
                            w = ctrl.get_actor()
                except Exception:
                    pass
                if w is not None and hasattr(w, 'id') and w.id == walker.id:
                    try: ctrl.stop()
                    except Exception: pass
                    return
    except Exception:
        pass

# ====================== Global Hyperparameters ======================
TIME_STEP = 0.05
population_size = 10

KEEP_ALIVE_PERIOD = 30

RECORDING = True

EPISODE_MAX_SECONDS = 180.0
NO_PROGRESS_SECONDS = 15
PROGRESS_THRESH = 2.0
STARTUP_STEPS = 300

NEARMISS_SAVE_TAU = 0.5
SVGD_TOP_CASES = 5
SVGD_STEPS_PER_CASE = 3
SVGD_EPS = 0.08
SVGD_BETA = 3
SVGD_GRAD_EPS = 0.35

DS_LIM = 25.0
DD_LIM = 4.5
DYAW_LIM = 20.0
MIN_SEP = 3.5

MIN_NPC = 20

SBART_STEPS_BASE = {
    "break": 10,
    "accelerate": 4,
    "left_change_acc": 10,
    "left_change_dec": 10,
    "right_change_acc": 10,
    "right_change_dec": 10,
}
DEFAULT_STEPS = 6

# ====================== Episode Recorder ======================
class EpisodeRecorder:
    def __init__(self, world_map):
        self.map = world_map
        self.frames = []

    @staticmethod
    def _vel_of(actor):
        v = actor.get_velocity()
        return (v.x, v.y, v.z)

    def log(self, ego: carla.Actor, actors: list):
        ego_tf = ego.get_transform()
        ego_vel = self._vel_of(ego)
        npcs = {}
        for a in actors:
            if a.id == ego.id:
                continue
            npcs[a.id] = {"tf": a.get_transform(), "vel": self._vel_of(a)}
        self.frames.append({"ego": {"tf": ego_tf, "vel": ego_vel}, "npcs": npcs})

# ====================== Hazard Computation / Training (unchanged) ======================
def _closing_speed_at(ego_tf, ego_vel, npc_tf, npc_vel):
    rx = npc_tf.location.x - ego_tf.location.x
    ry = npc_tf.location.y - ego_tf.location.y
    rnorm = math.hypot(rx, ry) + 1e-6
    ux, uy = rx / rnorm, ry / rnorm
    vrelx = npc_vel[0] - ego_vel[0]
    vrely = npc_vel[1] - ego_vel[1]
    v_close = -(vrelx * ux + vrely * uy)
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

# ====================== Main Program ======================
def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    COLLIDED_JSONL = "collided_scenarios.jsonl"

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(42)

    external_ads = True
    demo = MultiVehicleDemo(world, external_ads)
    world_map = world.get_map()

    surrogate = NPCHazardMLPSurrogate(ckpt_path="mlp_frozen.pt")


    camera = None
    image_queue = None

    Test_buget = 400
    number_game = 1
    side_total = 0
    timeout_total = 0
    red_light_total = 0
    obj_collison_total = 0
    cross_solid_line_total = 0
    seed_gen = seed_generator(carla_map=world_map, candidate_size=20)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    planner = KingPlanner(device=device)
    walker_planner = WalkerPlanner(device=device)

    abnormal_count = 0

    while number_game <= Test_buget:
        abnormal_case = False

        if abnormal_count > 2:
            scenario_conf = seed_gen.sample_seed_random()
        else:
            scenario_conf = seed_gen.sample_seed()

        # —— SVGD —— #

        pos_info = scenario_conf["position_info"]
        svgd_runner = RuntimeNPCSVGD(
            world_map,
            surrogate=surrogate,
            top_k=5,
            steps=SVGD_STEPS_PER_CASE,
            epsilon=SVGD_EPS,
            beta=SVGD_BETA,
            grad_eps=SVGD_GRAD_EPS,
            ds_lim=DS_LIM, dd_lim=DD_LIM, dyaw_lim=DYAW_LIM,
            min_sep=MIN_SEP
        )
        svgd_runner.refine_position_info(pos_info)
        logger.info("seed is refined by SVGD.")

        success = demo.setup_vehicles_with_collision(pos_info)
        if not success:
            logger.error("Vehicle/pedestrian spawn failed, skipping (not counted as a valid episode).")
            continue

        actual_n = len(demo.vehicles) + len(demo.pedestrians)
        if actual_n < MIN_NPC:
            logger.warning("only %d NPC spawned (<%d), rolling back and resampling.", actual_n, MIN_NPC)
            demo.destroy_all()
            continue

        logger.info("Requested %s, actually spawned %d (valid episode number %d)", pos_info.get('vehicle_num','?'), actual_n, number_game)

        # Record vehicle controllers (if any)
        controllers = []
        controller_by_actor_id = {}
        for i, v in enumerate(demo.vehicles):
            ctrl = demo.get_controller(i)
            controllers.append(ctrl)
            controller_by_actor_id[v.id] = ctrl

        if not demo.external_ads:
            demo.ego_vehicle.set_autopilot(True, tm.get_port())
            tm.vehicle_percentage_speed_difference(demo.ego_vehicle, 5)

        # Non-adversarial vehicles registered with Traffic Manager
        ego_id = demo.ego_vehicle.id if demo.ego_vehicle is not None else -1
        for v in demo.vehicles:
            if v.id != ego_id:
                v.set_autopilot(True, tm.get_port())
                tm.vehicle_percentage_speed_difference(v, random.randint(-10, 20))
                tm.ignore_signs_percentage(v, 0)
                tm.ignore_lights_percentage(v, 0)
                tm.ignore_walkers_percentage(v, 0)
                tm.distance_to_leading_vehicle(v, 2.5)

        # Recording
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

        # === Select adversarial targets (vehicles + pedestrians), by smallest |s| ===
        ego_tf0 = demo.ego_vehicle.get_transform()
        veh_scored = []
        for v in demo.vehicles:
            if v.id == ego_id: continue
            s, _ = ego_local_sd(ego_tf0, v.get_transform().location)
            veh_scored.append((abs(s), v))
        veh_scored.sort(key=lambda x: x[0])
        attack_vehicles = [v for _, v in veh_scored[:min(K_ATTACK, len(veh_scored))]]

        ped_scored = []
        for w in demo.pedestrians:
            s, _ = ego_local_sd(ego_tf0, w.get_transform().location)
            ped_scored.append((abs(s), w))
        ped_scored.sort(key=lambda x: x[0])
        attack_walkers = [w for _, w in ped_scored[:min(K_ATTACK, len(ped_scored))]]

        # Remove adversarial vehicles from Traffic Manager
        for v in attack_vehicles:
            v.set_autopilot(False)

        # Adversarial pedestrians: try to stop AI controller, clear control once
        for w in attack_walkers:
            try_stop_walker_ai(demo, w)
            w.apply_control(carla.WalkerControl(direction=carla.Vector3D(0.0, 0.0, 0.0), speed=0.0))

        # Pedestrian control cache from previous frame
        last_walker_cmd = {w.id: (0.0, 0.0) for w in attack_walkers}

        wall_start = time.monotonic()
        progress_anchor_loc = None
        progress_anchor_t = wall_start
        timeout_cnt = 0
        start_loc = None

        # Episode recorder
        rec = EpisodeRecorder(world_map)

        for mod in demo.modules:
            demo.enable_module(mod)

        # ========== Main Loop ==========
        for step in range(100000):
            world.wait_for_tick()

            if time.monotonic() - wall_start > EPISODE_MAX_SECONDS:
                logger.info("EARLY-EXIT: wall-clock > %ss", EPISODE_MAX_SECONDS)
                break

            # Startup phase
            if step < STARTUP_STEPS:
                ego_vel = demo.ego_vehicle.get_velocity()
                speed_ego = math.sqrt(ego_vel.x ** 2 + ego_vel.y ** 2 + ego_vel.z ** 2)
                if speed_ego > 5:
                    logger.warning("Abnormal EGO speed during startup phase, ending this episode.")
                    abnormal_case = True
            if step == STARTUP_STEPS and demo.external_ads:
                start_loc = demo.ego_vehicle.get_location()
                progress_anchor_loc = start_loc
                progress_anchor_t = time.monotonic()
                demo.set_destination()

            if step > STARTUP_STEPS:
                # TM keep-alive (non-adversarial vehicles)
                if step % KEEP_ALIVE_PERIOD == 0:
                    for v in demo.vehicles:
                        if v.id == ego_id or v in attack_vehicles:
                            continue
                        v.set_autopilot(True, tm.get_port())

                # tick & early exit
                signals_list, ego_collision, all_collision, cross_solid_line, red_light = demo.tick()
                if ego_collision or all_collision:
                    logger.info("EARLY-EXIT: Collision, ending episode.")
                    break

                # Recording frame
                if RECORDING and image_queue is not None:
                    try:
                        frame = image_queue.get_nowait()
                        cv2.imwrite(os.path.join(save_dir, f"frame_{step}.png"), frame)
                        spectator = world.get_spectator()
                        camera.set_transform(spectator.get_transform())
                    except Exception:
                        pass

                # Log replay data (vehicles + pedestrians)
                try:
                    all_npcs = list(demo.vehicles) + list(demo.pedestrians)
                    rec.log(demo.ego_vehicle, all_npcs)
                except Exception:
                    pass

                # No-progress fallback
                ego_loc = demo.ego_vehicle.get_location()
                if progress_anchor_loc is not None:
                    moved = math.hypot(ego_loc.x - progress_anchor_loc.x, ego_loc.y - progress_anchor_loc.y)
                    if moved >= PROGRESS_THRESH:
                        progress_anchor_loc = ego_loc
                        progress_anchor_t = time.monotonic()
                    elif time.monotonic() - progress_anchor_t > NO_PROGRESS_SECONDS:
                        logger.info("EARLY-EXIT: no progress for %ds (delta=%.2fm).", NO_PROGRESS_SECONDS, moved)
                        timeout_cnt = 1
                        break

                # Camera follow
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

                # Arrival detection
                if demo.ego_destination is not None:
                    near_dest, pass_dest = has_passed_destination(demo.ego_vehicle, demo.ego_destination, world_map)
                    if step != 0 and demo.external_ads and near_dest:
                        logger.info("Arrive, episode end")
                        break
                    elif pass_dest:
                        for mod in demo.modules: demo.enable_module(mod)
                        logger.warning("pass destination error")
                        abnormal_case = True

                # === Planning and Control ===
                if (step % REPLAN_STRIDE) == 0 and (attack_vehicles or attack_walkers):
                    ego_x0 = to_state(demo.ego_vehicle)

                    # Vehicles: re-plan + apply control
                    for v in attack_vehicles:
                        try:
                            a, delta = planner.plan_once(ego_x0, to_state(v), v.id)
                            apply_king_control(v, a, delta)
                        except Exception as e:
                            logger.warning("Vehicle planning exception veh %s: %s", v.id, e)
                            v.set_autopilot(True, tm.get_port())

                    # Pedestrians: re-plan + sanitize + apply control
                    for w in attack_walkers:
                        try:
                            vx, vy = walker_planner.plan_once(ego_x0, to_state_walker(w), w.id)
                            vx, vy = _clean_v(vx, vy, w, demo.ego_vehicle)       # FIX: sanitize
                            last_walker_cmd[w.id] = (vx, vy)
                            apply_walker_control(w, vx, vy)
                        except Exception as e:
                            logger.warning("Pedestrian planning exception walker %s: %s", w.id, e)
                else:
                    # Non-planning step: re-apply previous frame's pedestrian control (with re-sanitization)
                    for w in attack_walkers:
                        try:
                            vx0, vy0 = last_walker_cmd.get(w.id, (0.0, 0.0))
                            vx, vy = _clean_v(vx0, vy0, w, demo.ego_vehicle)     # FIX: re-sanitize
                            last_walker_cmd[w.id] = (vx, vy)
                            apply_walker_control(w, vx, vy)
                        except Exception:
                            pass

                # Debug: print pedestrian control and position every 20 ticks
                if (step % 20) == 0 and attack_walkers:
                    for w in attack_walkers:
                        try:
                            vx, vy = last_walker_cmd.get(w.id, (0.0, 0.0))
                            loc = w.get_transform().location
                            logger.debug("step=%d id=%s cmd=(%.2f,%.2f) pos=(%.1f,%.1f)", step, w.id, vx, vy, loc.x, loc.y)
                        except Exception:
                            pass

        # End of episode: clear Apollo prediction/planning cache (if applicable)
        try:
            apollo_clear_prediction_planning(times=3, interval=0.05)
        except Exception as e:
            logger.warning("apollo_clear_prediction_planning failed: %s", e)

        for _ in range(150):
            world.wait_for_tick()

        if RECORDING and camera is not None:
            try:
                camera.stop(); camera.destroy()
            except Exception:
                pass

        # End-of-episode check
        end_loc = demo.ego_vehicle.get_location()
        if start_loc is not None:
            distance_to_start = math.dist([start_loc.x, start_loc.y], [end_loc.x, end_loc.y])
        else:
            distance_to_start = 0.0
        logger.info("Moved distance: %s", distance_to_start)
        if distance_to_start < 1:
            logger.warning("EGO start-to-end distance < 1m, classified as abnormal, not counted in this generation.")
            abnormal_case = True

        seed_gen.executed_seed_set.append(scenario_conf)

        # Statistics
        if not abnormal_case:
            abnormal_count = 0
            side_total += demo.side_collision_count_vehicle
            obj_collison_total = demo.collision_count_obj
            timeout_total += timeout_cnt
            red_light_total += demo.ego_run_red_light
            cross_solid_line_total += demo.ego_cross_solid_line

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

            peak_y = float(dataset_tuple[1].max()) if n_samples > 0 else 0.0
            logger.info("EP-END: train-steps=%d, loss=%.4f, n=%d, peak_F=%.3f", steps_done, avg_loss, n_samples, peak_y)

            jsonable_conf = _to_jsonable(pos_info)
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
            logger.info("SAVED: scenario appended to %s", COLLIDED_JSONL)
            if demo.side_collision_count_vehicle == 0 and RECORDING and os.path.isdir(save_dir):
                shutil.rmtree(save_dir)

            number_game += 1
        else:
            abnormal_count += 1
            if RECORDING and os.path.isdir(save_dir):
                shutil.rmtree(save_dir)
            logger.info("This episode was abnormal, rolled back and resampled (not counted as a valid episode).")

        demo.destroy_all()

        keep_ids = {demo.ego_vehicle.id} if demo.ego_vehicle else set()
        rem_veh, rem_walk = purge_npcs(world, client, tm=tm, keep_actor_ids=keep_ids,
                                       include_walkers=True, hard_teleport=True)
        logger.info("PURGE: left vehicles=%d, walkers=%d", rem_veh, rem_walk)

    # ---- Final Statistics ----
    logger.info("Side collision (ego x vehicle): %d", side_total)
    logger.info("Ego vehicle object collision: %d", obj_collison_total)
    logger.info("Time Out: %d", timeout_total)
    logger.info("red light: %d", red_light_total)
    demo.destroy_all(); demo.close_connection()
    logger.info("Cleanup done.")

if __name__ == "__main__":
    main()
