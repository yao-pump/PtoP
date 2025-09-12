#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOSAT 主程序（完整可运行骨架）：
- 个体=场景；染色体=单车行为序列；基因=原子机动 或 模体机动
- 多目标：min METTC；max DFP、VOA、AEDF（对齐 MOSAT）
- NSGA-II：帕累托分级 + 拥挤度距离；交叉/变异；停滞重启
- 连续仿真执行：多个候选场景按队列交替注入，无需重置模拟器/重连 ADS
- 与已有 CARLA 辅助模块直接对接（MultiVehicleDemo, action_trans 等）

参考：MOSAT: Finding Safety Violations of ADS using Multi-objective Genetic Algorithm（ESEC/FSE'22）
"""

import os
import math
import time
import random
import shutil
import json
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import carla

# ====== 复用你工程里已有的工具 ====== #
from utility import (
    has_passed_destination, action_trans, apollo_clear_prediction_planning,
    purge_npcs
)
from world import MultiVehicleDemo

# ====================== 全局超参数 ======================
TIME_STEP = 0.05

# —— 连续仿真/早退 —— #
EPISODE_MAX_SECONDS = 120.0
NO_PROGRESS_SECONDS = 12
PROGRESS_THRESH = 2.0
STARTUP_STEPS = 300

# —— 种群规模/演化 —— #
POP_SIZE = 10
MAX_GENERATIONS = 20
P_CROSS = 0.4     # 交叉概率（个体层面）
P_MUT = 0.3       # 变异概率（安全违规场景内：双点交叉或洗牌）
STAGNATION_RESTART = 3  # 连续3代停滞则重启

# —— 染色体长度/车辆数限制 —— #
GENES_PER_CHROM_MIN = 4
GENES_PER_CHROM_MAX = 10
NPC_PER_SCEN_MIN = 4
NPC_PER_SCEN_MAX = 18
INIT_DIST_MAX = 50.0  # NPC 初始相对 EGO 距离上限（米）

# —— 模体时间片（与 MOSAT 设定一致/近似） —— #
ATOMIC_DT = 1.0
MOTIF_DT = 4.0

# —— DFP/VOA 采样步长 —— #
DFP_SAMPLE_DT = 0.1
VOA_SAMPLE_DT = 1.0

# —— 记录 & 回放存档 —— #
RECORD_JSONL = "mosat_violations.jsonl"
ARCHIVE_DIR = "mosat_replays"  # 留给你将来落地回放

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

# ====================== 原子机动定义 ======================
# 统一用 action_trans(ctrl, act_name) 执行，名称沿用你项目中的 ACT 集
ATOMIC_ACTIONS = [
    "accelerate", "break", "left_change_acc", "left_change_dec",
    "right_change_acc", "right_change_dec"
]

@dataclass
class AtomicGene:
    """一个原子机动 + 持续时间（秒）"""
    act: str
    dur: float = ATOMIC_DT

# ====================== 模体（Motif）定义（FSM 风格） ======================
# 4 种模体：ahead / side_front / behind / side_behind
# 根据 NPC 与 EGO 的相对构型（同/异车道、前后左右）生成一段动作序列

class MotifGene:
    def __init__(self, motif_type: str):
        assert motif_type in ["ahead", "side_front", "behind", "side_behind"]
        self.motif_type = motif_type
        self.dur = MOTIF_DT

    def plan(self, world_map, ego: carla.Actor, npc: carla.Actor) -> List[AtomicGene]:
        # 轻量近似：通过相对位置与车道判断，产出 1~3 个原子动作序列
        ego_wp = world_map.get_waypoint(ego.get_location())
        npc_wp = world_map.get_waypoint(npc.get_location())
        same_lane = (ego_wp.road_id == npc_wp.road_id and ego_wp.lane_id == npc_wp.lane_id)
        ex, ey = ego.get_location().x, ego.get_location().y
        nx, ny = npc.get_location().x, npc.get_location().y
        dx, dy = nx - ex, ny - ey
        cy, sy = _yaw_to_unit(ego.get_transform().rotation.yaw)
        front = (dx*cy + dy*sy) > 0  # 在 EGO 前方？

        if self.motif_type == "ahead" and same_lane and front:
            # 三择一：减速/制动/左右变道并回位（近似为一次变道）
            choice = random.choice([
                [AtomicGene("left_change_dec")],
                [AtomicGene("right_change_dec")],
                [AtomicGene("break")]
            ])
            return choice

        if self.motif_type == "side_front" and (not same_lane) and front:
            # 先并入 EGO 车道，再：减速或制动或再换道
            first = [AtomicGene(random.choice(["left_change_dec","right_change_dec"]))]
            second = [AtomicGene(random.choice(["break","left_change_dec","right_change_dec"]))]
            return first + second

        if self.motif_type == "behind" and same_lane and (not front):
            # 加速逼近 -> 邻道超越（加速）
            return [AtomicGene(random.choice(["left_change_acc","right_change_acc"])),
                    AtomicGene("accelerate")]

        if self.motif_type == "side_behind" and (not same_lane) and (not front):
            # 加速切到 EGO 前侧
            return [AtomicGene("accelerate"),
                    AtomicGene(random.choice(["left_change_acc","right_change_acc"]))]

        # 默认回退：轻微加减速
        return [AtomicGene(random.choice(["accelerate","break"]))]

# ====================== 染色体/个体表示 ======================
@dataclass
class Gene:
    kind: str  # "atomic" | "motif"
    payload: object  # AtomicGene 或 MotifGene

@dataclass
class Chromosome:
    """一辆 NPC 的机动序列"""
    genes: List[Gene] = field(default_factory=list)

@dataclass
class ScenarioIndividual:
    """一个场景 = 若干 NPC 的染色体 + 初始相对位姿"""
    chroms: List[Chromosome] = field(default_factory=list)
    # 初始相对位置（相对 EGO 起点）：[(ds, dd, dyaw), ...]，与 chroms 对齐
    init_rel: List[Tuple[float,float,float]] = field(default_factory=list)
    # 适应度（多目标）：(METTC, -DFP, -VOA, -AEDF)  —— 注意排序方向
    fitness: Optional[Tuple[float,float,float,float]] = None
    # 标记是否发生安全违规（碰撞/红灯等），用于变异策略
    violated: bool = False

# ====================== 随机初始化 ======================
def random_atomic_gene() -> Gene:
    return Gene("atomic", AtomicGene(act=random.choice(ATOMIC_ACTIONS)))

def random_motif_gene() -> Gene:
    return Gene("motif", MotifGene(random.choice(["ahead","side_front","behind","side_behind"])))

def random_chromosome() -> Chromosome:
    L = random.randint(GENES_PER_CHROM_MIN, GENES_PER_CHROM_MAX)
    genes = []
    for _ in range(L):
        if random.random() < 0.5:
            genes.append(random_atomic_gene())
        else:
            genes.append(random_motif_gene())
    return Chromosome(genes)

def random_individual(npcs: int) -> ScenarioIndividual:
    chroms = [random_chromosome() for _ in range(npcs)]
    init_rel = []
    for _ in range(npcs):
        ds = random.uniform(5.0, INIT_DIST_MAX) * (1 if random.random()<0.6 else -1)  # 前后
        dd = random.uniform(-4.0, 4.0)  # 左右
        dyaw = random.uniform(-15.0, 15.0)
        init_rel.append((ds, dd, dyaw))
    return ScenarioIndividual(chroms=chroms, init_rel=init_rel)

# ====================== NSGA-II：帕累托分级 + 拥挤度 ======================
def dominates(a, b):
    """a 是否支配 b（全部不劣，且至少一项严格更好）—— 目标向量： (METTC, -DFP, -VOA, -AEDF)"""
    better_or_equal = all(x <= y for x, y in zip(a, b))
    strictly_better = any(x < y for x, y in zip(a, b))
    return better_or_equal and strictly_better

def fast_non_dominated_sort(pop):
    S = defaultdict(list)
    n = defaultdict(int)
    fronts = [[]]
    for p in pop:
        S[p] = []
        n[p] = 0
        for q in pop:
            if p is q: continue
            if dominates(p.fitness, q.fitness):
                S[p].append(q)
            elif dominates(q.fitness, p.fitness):
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        Q = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    Q.append(q)
        i += 1
        fronts.append(Q)
    if not fronts[-1]:
        fronts.pop()
    return fronts

def crowding_distance(front):
    m = len(front[0].fitness)
    N = len(front)
    dist = {ind: 0.0 for ind in front}
    for k in range(m):
        front.sort(key=lambda ind: ind.fitness[k])
        dist[front[0]] = dist[front[-1]] = float('inf')
        fmin = front[0].fitness[k]
        fmax = front[-1].fitness[k]
        if fmax == fmin:
            continue
        for i in range(1, N-1):
            prev = front[i-1].fitness[k]
            nxt = front[i+1].fitness[k]
            dist[front[i]] += (nxt - prev) / (fmax - fmin)
    return dist

def nsga2_select(pop, k):
    fronts = fast_non_dominated_sort(pop)
    new_pop = []
    for F in fronts:
        if len(new_pop) + len(F) <= k:
            new_pop.extend(F)
        else:
            d = crowding_distance(F)
            F.sort(key=lambda ind: d[ind], reverse=True)
            new_pop.extend(F[:(k - len(new_pop))])
            break
    return new_pop

# ====================== 交叉/变异 ======================
def uniform_crossover(p1: ScenarioIndividual, p2: ScenarioIndividual):
    c1 = ScenarioIndividual()
    c2 = ScenarioIndividual()
    for (ch1, ch2) in zip(p1.chroms, p2.chroms):
        if random.random() < 0.5:
            c1.chroms.append(ch1); c2.chroms.append(ch2)
        else:
            c1.chroms.append(ch2); c2.chroms.append(ch1)
    # 初始相对位姿也统一交叉
    for (r1, r2) in zip(p1.init_rel, p2.init_rel):
        if random.random() < 0.5:
            c1.init_rel.append(r1); c2.init_rel.append(r2)
        else:
            c1.init_rel.append(r2); c2.init_rel.append(r1)
    return c1, c2

def gene_mutation_min_ettc(ind: ScenarioIndividual, target_ch_idx: int, gene_idx: int):
    """对最小 ETTC 的基因做‘替换/参数扰动’"""
    chrom = ind.chroms[target_ch_idx]
    g = chrom.genes[gene_idx]
    if g.kind == "atomic":
        # 原子基因：动作替换 或 时长微调
        if random.random() < 0.5:
            g.payload.act = random.choice(ATOMIC_ACTIONS)
        else:
            g.payload.dur = max(0.5, min(3.0, g.payload.dur + random.uniform(-0.3, 0.3)))
    else:
        # 模体基因：换另一个模体 或 退化成一个原子动作
        if random.random() < 0.5:
            g.payload = MotifGene(random.choice(["ahead","side_front","behind","side_behind"]))
        else:
            chrom.genes[gene_idx] = random_atomic_gene()

def two_point_crossover_inside(ind: ScenarioIndividual):
    """在同一场景内随机挑两条染色体，做双点交换"""
    if len(ind.chroms) < 2: return
    i, j = random.sample(range(len(ind.chroms)), 2)
    c1, c2 = ind.chroms[i], ind.chroms[j]
    if len(c1.genes) < 3 or len(c2.genes) < 3: return
    p1 = random.randint(1, len(c1.genes)-2)
    p2 = random.randint(1, len(c2.genes)-2)
    c1.genes[p1:], c2.genes[p2:] = c2.genes[p2:], c1.genes[p1:]

def shuffle_inside(ind: ScenarioIndividual):
    """逐条染色体随机洗牌基因顺序"""
    for c in ind.chroms:
        random.shuffle(c.genes)

# ====================== 记录器（采集帧数据用于指标计算） ======================
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
            if v.id == ego.id: continue
            npcs[v.id] = {"tf": v.get_transform(), "vel": self._vel_of(v)}
        self.frames.append({"ego": {"tf": ego_tf, "vel": ego_vel}, "npcs": npcs})

# ====================== 多目标指标 ======================
def _speed_of(vel): return math.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)

def compute_METTC(frames) -> float:
    """最小 ETTC（近似实现：逐帧估计，取最小）"""
    if not frames: return float('inf')
    min_ettc = float('inf')
    for f in frames:
        ego_tf = f["ego"]["tf"]; ego_vel = f["ego"]["vel"]
        v_ego = _speed_of(ego_vel)
        cy, sy = _yaw_to_unit(ego_tf.rotation.yaw)
        for npc_id, nk in f["npcs"].items():
            # 估算两直线轨迹交点到 EGO 的时间
            ex, ey = ego_tf.location.x, ego_tf.location.y
            nx, ny = nk["tf"].location.x, nk["tf"].location.y
            # 简化：若方向接近且前向距离缩小，则估一个相遇时间
            dx, dy = nx - ex, ny - ey
            dist = math.hypot(dx, dy) + 1e-6
            if v_ego < 0.2: continue
            ettc = dist / v_ego
            min_ettc = min(min_ettc, ettc)
    return max(0.0, min_ettc)

def compute_DFP(frames, world_map) -> float:
    """最大路线偏离：planned 路径近似为按车道中心线匀速前进"""
    if not frames: return 0.0
    # 取起始 EGO 位姿，沿车道中心线投影后与实际轨迹的最大偏差
    start_tf = frames[0]["ego"]["tf"]
    max_dev = 0.0
    for f in frames:
        ego_tf = f["ego"]["tf"]
        wp = world_map.get_waypoint(ego_tf.location, project_to_road=True)
        center = wp.transform.location
        dev = math.hypot(ego_tf.location.x - center.x, ego_tf.location.y - center.y)
        max_dev = max(max_dev, dev)
    return max_dev

def compute_VOA(frames) -> float:
    """最大加加速度（jerk）近似：1s 速度差分的二阶差分绝对值最大"""
    if not frames: return 0.0
    # 取每 1s 一个速度样本
    t_step = max(1, int(1.0 / TIME_STEP))
    v = []
    for i in range(0, len(frames), t_step):
        v.append(_speed_of(frames[i]["ego"]["vel"]))
    if len(v) < 3: return 0.0
    jerk = [abs((v[t] - v[t-1]) - (v[t-1]-v[t-2])) for t in range(2, len(v))]
    return max(jerk) if jerk else 0.0

def compute_AEDF(frames, archive_trajs) -> float:
    """与历史‘已发现违规’场景的 NPC 轨迹欧氏距离平均值（越大越多样）"""
    if not frames or not archive_trajs: return 0.0
    # 取当前场景中第一辆 NPC 的轨迹点（相对 EGO 起点）
    ego0 = frames[0]["ego"]["tf"].location
    cur_pts = []
    for f in frames:
        npcs = f["npcs"]
        if not npcs: continue
        any_npc = next(iter(npcs.values()))
        loc = any_npc["tf"].location
        cur_pts.append((loc.x - ego0.x, loc.y - ego0.y))
    if not cur_pts: return 0.0
    cur = np.array(cur_pts)
    dists = []
    for ref in archive_trajs:
        m = min(len(cur), len(ref))
        if m == 0: continue
        d = np.linalg.norm(cur[:m] - ref[:m], axis=1).mean()
        dists.append(d)
    return float(np.mean(dists)) if dists else 0.0

# ====================== 连续仿真执行器 ======================
class ScenarioExecutor:
    def __init__(self, world, demo: MultiVehicleDemo):
        self.world = world
        self.demo = demo
        self.world_map = world.get_map()
        self.archive_trajs = []  # 违规样本 NPC 轨迹（用于 AEDF）

    def _spawn_from_rel(self, ego: carla.Actor, rel_list: List[Tuple[float,float,float]]):
        ego_tf = ego.get_transform()
        cy, sy = _yaw_to_unit(ego_tf.rotation.yaw)
        spawns = []
        for (ds, dd, dyaw) in rel_list:
            # 局部坐标 -> 世界坐标
            x = ego_tf.location.x + ds*cy - dd*sy
            y = ego_tf.location.y + ds*sy + dd*cy
            loc = carla.Location(x=x, y=y, z=ego_tf.location.z)
            rot = carla.Rotation(yaw=ego_tf.rotation.yaw + dyaw)
            spawns.append(carla.Transform(loc, rot))
        ok = self.demo.setup_vehicles_with_collision({"spawn_points": spawns})
        return ok

    def _apply_chromosomes(self, chroms: List[Chromosome], controller_by_id: Dict[int, object]):
        """执行基因序列（按时间片循环）。模体在 plan() 时展开为原子序列。"""
        expanded: Dict[int, List[AtomicGene]] = {}
        for i, v in enumerate(self.demo.vehicles):
            if v.id == self.demo.ego_vehicle.id: continue
            seq: List[AtomicGene] = []
            for g in chroms[i].genes:
                if g.kind == "atomic":
                    seq.append(g.payload)
                else:
                    seq.extend(g.payload.plan(self.world_map, self.demo.ego_vehicle, v))
            expanded[v.id] = seq

        # 同步推进：每步取各车当前基因的一个原子动作，按其 dur/TIME_STEP 执行
        override = {vid: {"idx": 0, "left": (seq[0].dur if seq else 0.0)} for vid, seq in expanded.items()}
        last_tick = time.monotonic()
        while True:
            self.world.wait_for_tick()
            now = time.monotonic()
            dt = max(TIME_STEP, now - last_tick)
            last_tick = now

            finished = 0
            for vid, st in list(override.items()):
                seq = expanded[vid]
                if not seq: finished += 1; continue
                idx = st["idx"]
                if idx >= len(seq):
                    finished += 1; continue
                act = seq[idx].act
                ctrl = controller_by_id.get(vid)
                if ctrl: action_trans(ctrl, act)
                st["left"] -= dt
                if st["left"] <= 0:
                    st["idx"] += 1
                    if st["idx"] < len(seq):
                        st["left"] = seq[st["idx"]].dur
            if finished == len(expanded):
                break

    def run(self, individual: ScenarioIndividual) -> Tuple[bool, EpisodeRecorder, Dict]:
        """返回：(是否违规, 记录器, 统计)"""
        demo = self.demo
        world = self.world
        rec = EpisodeRecorder(self.world_map)

        # 重置环境（不重启 ADS，仅清场）
        demo.destroy_all()
        world.wait_for_tick()

        # 生成车辆
        success = demo.setup_vehicles_with_collision({})  # 先只生成 EGO
        if not success or demo.ego_vehicle is None:
            return False, rec, {"spawn_ok": False}

        # 将 NPC 按相对位姿生成
        if not self._spawn_from_rel(demo.ego_vehicle, individual.init_rel):
            return False, rec, {"spawn_ok": False}

        ego_id = demo.ego_vehicle.id
        controllers = []
        controller_by_actor_id = {}
        for i, v in enumerate(demo.vehicles):
            ctrl = demo.get_controller(i)
            controllers.append(ctrl)
            controller_by_actor_id[v.id] = ctrl

        # 上 TM
        tm = world.get_client().get_trafficmanager(8000)
        for v in demo.vehicles:
            if v.id != ego_id:
                v.set_autopilot(True, tm.get_port())
                tm.vehicle_percentage_speed_difference(v, random.randint(-10, 15))
                tm.distance_to_leading_vehicle(v, 2.5)

        # 开始计时
        wall_start = time.monotonic()
        progress_anchor_loc = None
        progress_anchor_t = wall_start

        # 预热
        for step in range(STARTUP_STEPS):
            world.wait_for_tick()

        # 目的地（若需要）
        if demo.external_ads:
            demo.set_destination()
            progress_anchor_loc = demo.ego_vehicle.get_location()
            progress_anchor_t = time.monotonic()

        violated = False
        violation_reason = ""

        # 执行基因序列
        self._apply_chromosomes(individual.chroms, controller_by_actor_id)

        # 主循环：采集状态 + 违规判定 + 早退
        for step in range(50000):
            world.wait_for_tick()
            if time.monotonic() - wall_start > EPISODE_MAX_SECONDS:
                violation_reason = "timeout"
                break

            # 统计 + 碰撞监听
            signals_list, ego_collision, all_collision, cross_solid_line, red_light = demo.tick()
            if ego_collision or all_collision:
                violated = True
                violation_reason = "collision"
                break
            if red_light:
                violated = True
                violation_reason = "red_light"
                break

            # 进度超时
            ego_loc = demo.ego_vehicle.get_location()
            if progress_anchor_loc is not None:
                moved = math.hypot(ego_loc.x - progress_anchor_loc.x, ego_loc.y - progress_anchor_loc.y)
                if moved >= PROGRESS_THRESH:
                    progress_anchor_loc = ego_loc
                    progress_anchor_t = time.monotonic()
                elif time.monotonic() - progress_anchor_t > NO_PROGRESS_SECONDS:
                    violation_reason = "no_progress"
                    break

            # 记录
            rec.log(demo.ego_vehicle, demo.vehicles)

            # 到达即结束
            if demo.ego_destination is not None:
                near_dest, pass_dest = has_passed_destination(demo.ego_vehicle, demo.ego_destination, self.world_map)
                if near_dest: break

        # 清 Apollo 缓存
        try:
            apollo_clear_prediction_planning(times=2, interval=0.05)
        except Exception:
            pass

        # 保存违规轨迹到档案（用于 AEDF）
        if violated and rec.frames:
            ego0 = rec.frames[0]["ego"]["tf"].location
            traj = []
            for f in rec.frames:
                npcs = f["npcs"]
                if not npcs: continue
                any_npc = next(iter(npcs.values()))
                loc = any_npc["tf"].location
                traj.append([loc.x - ego0.x, loc.y - ego0.y])
            self.archive_trajs.append(np.array(traj, dtype=np.float32))

        # 清场（保留 EGO 以连续运行可继续实现；此处简单清空）
        keep_ids = {demo.ego_vehicle.id} if demo.ego_vehicle else set()
        purge_npcs(world, world.get_client(), tm=None, keep_actor_ids=keep_ids,
                   include_walkers=True, hard_teleport=True)

        return violated, rec, {"spawn_ok": True, "reason": violation_reason}

# ====================== MOSAT 搜索器 ======================
class MOSATSearchEngine:
    def __init__(self, world, demo):
        self.world = world
        self.demo = demo
        self.executor = ScenarioExecutor(world, demo)
        self.generation = 1
        self.archive = []  # 违规场景归档（用于 AEDF）
        self.stagnation = 0
        self.last_front_signature = None

    def _evaluate(self, ind: ScenarioIndividual, rec: EpisodeRecorder, violated: bool):
        # 计算四目标
        METTC = compute_METTC(rec.frames)
        DFP = compute_DFP(rec.frames, self.world.get_map())
        VOA = compute_VOA(rec.frames)
        AEDF = compute_AEDF(rec.frames, self.executor.archive_trajs)
        # NSGA-II 统一按“越小越好”排序：对最大化项取负
        ind.fitness = (METTC, -DFP, -VOA, -AEDF)
        ind.violated = violated

    def initial_population(self) -> List[ScenarioIndividual]:
        n = random.randint(NPC_PER_SCEN_MIN, NPC_PER_SCEN_MAX)
        return [random_individual(n) for _ in range(POP_SIZE)]

    def variation(self, parents: List[ScenarioIndividual]) -> List[ScenarioIndividual]:
        children: List[ScenarioIndividual] = []

        # —— 按个体层面交叉 —— #
        crossed_pool = []
        for p in parents:
            if random.random() < P_CROSS:
                crossed_pool.append(p)
        random.shuffle(crossed_pool)
        for i in range(0, len(crossed_pool)-1, 2):
            c1, c2 = uniform_crossover(crossed_pool[i], crossed_pool[i+1])
            children.extend([c1, c2])

        # —— 逐个体应用“内变异” —— #
        for p in parents:
            child = random_individual(len(p.chroms))  # 先拷一份结构
            child.chroms = [Chromosome([Gene(g.kind, g.payload) for g in ch.genes]) for ch in p.chroms]
            child.init_rel = list(p.init_rel)

            if not p.violated:
                # 未违规：对“最小 ETTC 基因”做定向变异
                # 这里用启发式：挑随机染色体 & 基因位点进行一次变异
                ch_idx = random.randrange(len(child.chroms))
                gi = random.randrange(len(child.chroms[ch_idx].genes))
                gene_mutation_min_ettc(child, ch_idx, gi)
            else:
                # 已违规：增加多样性 —— 双点交叉或洗牌
                if random.random() < P_MUT:
                    two_point_crossover_inside(child)
                else:
                    shuffle_inside(child)
            children.append(child)
        return children

    def restart_if_needed(self, pop: List[ScenarioIndividual]) -> List[ScenarioIndividual]:
        # 若连续若干代的第一前沿（按 fitness 排序）未变化，则重启
        fronts = fast_non_dominated_sort(pop)
        top = sorted([tuple(ind.fitness) for ind in fronts[0]])
        sig = tuple(top)
        if sig == self.last_front_signature:
            self.stagnation += 1
        else:
            self.stagnation = 0
            self.last_front_signature = sig
        if self.stagnation >= STAGNATION_RESTART:
            self.stagnation = 0
            self.last_front_signature = None
            return self.initial_population()
        return pop

    def run(self, budget_episodes=10000):
        # 初始种群
        population = self.initial_population()
        episodes = 0

        while episodes < budget_episodes and self.generation <= MAX_GENERATIONS:
            evaluated = []
            # —— 连续仿真：依次执行本代所有个体 —— #
            for ind in population:
                violated, rec, stat = self.executor.run(ind)
                self._evaluate(ind, rec, violated)
                # 记录违规样本
                if violated:
                    append_jsonl(RECORD_JSONL, {
                        "gen": self.generation,
                        "fitness": ind.fitness,
                        "violated": True,
                        "reason": stat.get("reason",""),
                        "ts": time.time()
                    })
                evaluated.append(ind)
                episodes += 1
                if episodes >= budget_episodes: break

            # —— 选择（NSGA-II） —— #
            population = nsga2_select(evaluated, POP_SIZE)
            # —— 变异/交叉 —— #
            children = self.variation(population)
            population = nsga2_select(population + children, POP_SIZE)
            # —— 停滞重启 —— #
            population = self.restart_if_needed(population)

            print(f"[MOSAT] Generation {self.generation} done, episodes={episodes}")
            self.generation += 1

# ====================== I/O 工具 ======================
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
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ====================== 主入口 ======================
def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # 交通灯全绿（可保留）
    for actor in world.get_actors():
        if isinstance(actor, carla.TrafficLight):
            actor.set_state(carla.TrafficLightState.Green)
            actor.freeze(False)

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(42)

    demo = MultiVehicleDemo(world, external_ads=True)

    # MOSAT 搜索
    engine = MOSATSearchEngine(world, demo)
    engine.run(budget_episodes=400)  # 与原 Test_buget 对齐

    demo.destroy_all(); demo.close_connection()
    print("[MOSAT] Cleanup done.")

if __name__ == "__main__":
    main()
