# svgd_optimizer.py
# 通用 SVGD 优化器 + 初始位姿问题适配器 + MLP 训练封装（main 只管调用）

from __future__ import annotations
import math
import numpy as np
from typing import Optional, Tuple, List, Callable

try:
    import torch
    import carla
except Exception:
    torch = None
    carla = None


# =============== 小工具 ===============
def _vec_norm3(x, y, z):
    return math.sqrt(x * x + y * y + z * z)

def _ego_local_sd(ego_tf: "carla.Transform", loc: "carla.Location") -> Tuple[float, float]:
    yaw = math.radians(ego_tf.rotation.yaw)
    cy, sy = math.cos(yaw), math.sin(yaw)
    dx = loc.x - ego_tf.location.x
    dy = loc.y - ego_tf.location.y
    s = dx * cy + dy * sy
    d = -dx * sy + dy * cy
    return float(s), float(d)

def _sd_to_transform(ego_tf: "carla.Transform", ds: float, dd: float, dyaw: float, z: Optional[float] = None) -> "carla.Transform":
    yaw_rad = math.radians(ego_tf.rotation.yaw)
    fx, fy = math.cos(yaw_rad), math.sin(yaw_rad)      # 前向
    rx, ry = -math.sin(yaw_rad), math.cos(yaw_rad)     # 右向
    loc = carla.Location(
        x=ego_tf.location.x + ds * fx + dd * rx,
        y=ego_tf.location.y + ds * fy + dd * ry,
        z=ego_tf.location.z if z is None else z
    )
    yaw = ego_tf.rotation.yaw + dyaw
    rot = carla.Rotation(pitch=ego_tf.rotation.pitch, yaw=yaw, roll=ego_tf.rotation.roll)
    return carla.Transform(loc, rot)


# =============== SVGD Optimizer（通用） ===============
class SVGDOptimizer:
    def __init__(self, epsilon: float = 0.08, beta: float = 3.0, grad_eps: float = 0.35, bandwidth: str = "median"):
        self.epsilon = float(epsilon)
        self.beta = float(beta)
        self.grad_eps = float(grad_eps)
        self.bandwidth = bandwidth

    @staticmethod
    def _pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
        X2 = np.sum(X * X, axis=1, keepdims=True)
        return X2 + X2.T - 2.0 * (X @ X.T)

    def _rbf_kernel(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        pd2 = self._pairwise_sq_dists(X)
        N = max(1, X.shape[0])
        med = np.median(pd2[pd2 > 0]) if np.any(pd2 > 0) else 1.0
        h2 = med / (2.0 * math.log(N + 1.0)) if self.bandwidth == "median" else float(self.bandwidth)
        if not np.isfinite(h2) or h2 <= 1e-12:
            h2 = 1.0
        K = np.exp(-pd2 / (2.0 * h2))
        return K, h2

    def _finite_diff_grad(self, X: np.ndarray, value_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        N, D = X.shape
        g = np.zeros_like(X, dtype=np.float64)
        for d in range(D):
            E = np.zeros_like(X); E[:, d] = self.grad_eps
            f1 = value_fn(X + E).reshape(-1)
            f2 = value_fn(X - E).reshape(-1)
            g[:, d] = (f1 - f2) / (2.0 * self.grad_eps)
        if not np.any(np.isfinite(g)):
            g = np.zeros_like(X)
        return g

    def step(self,
             X: np.ndarray,
             value_fn: Callable[[np.ndarray], np.ndarray],
             grad_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
             constraint_project: Optional[Callable[[np.ndarray], np.ndarray]] = None,
             mask: Optional[np.ndarray] = None) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.size == 0:
            return X
        N, D = X.shape

        g = self._finite_diff_grad(X, value_fn) if grad_fn is None else np.asarray(grad_fn(X), dtype=np.float64)
        K, h2 = self._rbf_kernel(X)

        attract = (K @ g) / float(N)
        row_sum = np.sum(K, axis=1, keepdims=True)
        rep = (row_sum * X - K @ X) / (h2 * float(N))
        rep = -rep

        phi = attract + self.beta * rep

        if mask is not None:
            M = mask
            if M.ndim == 1: M = M.reshape(-1, 1)
            if M.shape[1] == 1 and D > 1: M = np.repeat(M, D, axis=1)
            phi = phi * M

        X_new = X + self.epsilon * phi
        if constraint_project is not None:
            X_new = constraint_project(X_new)
        return X_new


# =============== 初始位姿：问题适配器 ===============
class InitialPoseSVGDProblem:
    """
    把 scenario_conf['surrounding_info']（非行人）映射为粒子 X=(ds,dd,dyaw)，
    使用 surrogate MLP 给分（value_fn），并提供约束投影与写回。
    """
    def __init__(self,
                 world_map: "carla.Map",
                 surrogate,
                 scenario_conf: dict,
                 ds_lim: float = 25.0,
                 dd_lim: float = 4.5,
                 dyaw_lim: float = 20.0,
                 min_sep: float = 3.5,
                 top_k: Optional[int] = None):
        assert carla is not None, "carla 未导入"
        self.world_map = world_map
        self.surrogate = surrogate
        self.sc = scenario_conf
        self.ds_lim, self.dd_lim, self.dyaw_lim = ds_lim, dd_lim, dyaw_lim
        self.min_sep = min_sep
        self.top_k = top_k

        self.ego_tf = self.sc["ego_transform"]
        self._items = self._collect_items(self.sc["surrounding_info"])
        self._opt_idx = [i for (i, t, _) in self._items if t != "pedestrian"]
        self._X = self._extract_particles()
        self._last_scores = None

    # --- 解析 surrounding_info ---
    @staticmethod
    def _norm_type(t) -> str:
        t = str(t).lower()
        if "pedestrian" in t or t == "walker": return "pedestrian"
        if "bicycle" in t or "bike" in t: return "bicycle"
        if "car" in t or "vehicle" in t: return "car"
        return t

    @staticmethod
    def _collect_items(surrounding) -> List[Tuple[int, str, "carla.Transform"]]:
        out = []
        if isinstance(surrounding, list):
            for i, item in enumerate(surrounding):
                out.append((i, InitialPoseSVGDProblem._norm_type(item["type"]), item["transform"]))
        else:
            for i, tf in enumerate(surrounding["transform"]):
                t = InitialPoseSVGDProblem._norm_type(surrounding["type"][i])
                out.append((i, t, tf))
        return out

    def _get_item_tf(self, idx: int) -> "carla.Transform":
        surr = self.sc["surrounding_info"]
        return surr[idx]["transform"] if isinstance(surr, list) else surr["transform"][idx]

    def _set_item_tf(self, idx: int, tf: "carla.Transform") -> None:
        surr = self.sc["surrounding_info"]
        if isinstance(surr, list): surr[idx]["transform"] = tf
        else: surr["transform"][idx] = tf

    # --- 粒子 X <-> 场景 ---
    def _extract_particles(self) -> np.ndarray:
        ego_tf = self.ego_tf
        X = []
        for idx in self._opt_idx:
            tf = self._get_item_tf(idx)
            ds, dd = _ego_local_sd(ego_tf, tf.location)
            dyaw = tf.rotation.yaw - ego_tf.rotation.yaw
            while dyaw >= 180: dyaw -= 360
            while dyaw < -180: dyaw += 360
            X.append([ds, dd, dyaw])
        return np.asarray(X, dtype=np.float64) if X else np.zeros((0, 3), dtype=np.float64)

    def get_particles(self) -> np.ndarray:
        return self._X.copy()

    def set_particles(self, X: np.ndarray) -> None:
        self._X = np.asarray(X, dtype=np.float64)

    # --- surrogate 打分 ---
    def _build_feats_batch(self, X: np.ndarray) -> np.ndarray:
        feats = []
        for (row, idx) in zip(X, self._opt_idx):
            ds, dd, dyaw = float(row[0]), float(row[1]), float(row[2])
            tf = _sd_to_transform(self.ego_tf, ds, dd, dyaw, z=self.ego_tf.location.z)
            feats.append(self.surrogate._build_feats(self.world_map, self.ego_tf, tf))
        return np.asarray(feats, dtype=np.float32) if feats else np.zeros((0, 8), np.float32)

    def value_fn(self, X: np.ndarray) -> np.ndarray:
        if X.shape[0] == 0:
            self._last_scores = np.zeros((0,), dtype=np.float64)
            return self._last_scores
        feats = self._build_feats_batch(X)
        if torch is not None:
            with torch.no_grad():
                device = self.surrogate.device if hasattr(self.surrogate, "device") else "cpu"
                t = torch.tensor(feats, device=device)
                pred = self.surrogate.model(t).squeeze(-1)
                y = pred.detach().cpu().float().numpy().reshape(-1)
        else:
            y = np.zeros((X.shape[0],), dtype=np.float64)
        self._last_scores = y.astype(np.float64)
        return self._last_scores

    def grad_fn(self, X: np.ndarray) -> Optional[np.ndarray]:
        # 若你的 surrogate 提供显式梯度，可在此调用；默认 None => 有限差分
        return None

    def project(self, X: np.ndarray) -> np.ndarray:
        if X.size == 0: return X
        Xp = X.copy()
        Xp[:, 0] = np.clip(Xp[:, 0], -self.ds_lim, self.ds_lim)
        Xp[:, 1] = np.clip(Xp[:, 1], -self.dd_lim, self.dd_lim)
        Xp[:, 2] = np.clip(Xp[:, 2], -self.dyaw_lim, self.dyaw_lim)

        # 最小间距（s,d 上）
        if self.min_sep and Xp.shape[0] >= 2:
            SD = Xp[:, :2]
            pd2 = SVGDOptimizer._pairwise_sq_dists(SD)
            N = SD.shape[0]
            for i in range(N):
                for j in range(i + 1, N):
                    dist = math.sqrt(max(1e-12, pd2[i, j]))
                    if dist < self.min_sep:
                        dvec = (SD[i] - SD[j]) / (dist + 1e-9)
                        delta = 0.5 * (self.min_sep - dist) * dvec
                        SD[i] += delta
                        SD[j] -= delta
            SD[:, 0] = np.clip(SD[:, 0], -self.ds_lim, self.ds_lim)
            SD[:, 1] = np.clip(SD[:, 1], -self.dd_lim, self.dd_lim)
            Xp[:, :2] = SD
        return Xp

    def write_back(self) -> None:
        if self._X.size == 0: return
        for (row, idx) in zip(self._X, self._opt_idx):
            ds, dd, dyaw = float(row[0]), float(row[1]), float(row[2])
            old_tf = self._get_item_tf(idx)
            tf_new = _sd_to_transform(self.ego_tf, ds, dd, dyaw, z=old_tf.location.z)
            self._set_item_tf(idx, tf_new)

    def mask_from_scores(self, scores: np.ndarray, top_k: Optional[int] = None) -> np.ndarray:
        if scores.size == 0:
            return scores
        k = self.top_k if top_k is None else top_k
        if (k is None) or (k <= 0) or (k >= scores.size):
            return np.ones_like(scores, dtype=np.float64)
        idx = np.argsort(-scores)[:k]
        m = np.zeros_like(scores, dtype=np.float64)
        m[idx] = 1.0
        return m


# =============== 训练标签（min-distance hazard） ===============
def _closing_speed_at(ego_tf, ego_vel, npc_tf, npc_vel):
    rx = npc_tf.location.x - ego_tf.location.x
    ry = npc_tf.location.y - ego_tf.location.y
    r = math.hypot(rx, ry) + 1e-6
    ux, uy = rx / r, ry / r
    vrelx = npc_vel[0] - ego_vel[0]
    vrely = npc_vel[1] - ego_vel[1]
    v_close = -(vrelx * ux + vrely * uy)
    return max(0.0, v_close)

def _find_min_distance_window(frames, npc_id, win=2):
    d_min = float('inf'); t_star = -1
    for t in range(len(frames)):
        npcs = frames[t]["npcs"]
        if npc_id not in npcs: continue
        ego_tf = frames[t]["ego"]["tf"]; npc_tf = npcs[npc_id]["tf"]
        d = math.hypot(npc_tf.location.x - ego_tf.location.x, npc_tf.location.y - ego_tf.location.y)
        if d < d_min:
            d_min = d; t_star = t
    if t_star < 0:
        return -1, [], float('inf')
    i0 = max(0, t_star - win); i1 = min(len(frames) - 1, t_star + win)
    return t_star, list(range(i0, i1 + 1)), d_min

def hazard_from_min_distance(frames, world_map, npc_id,
                             D0=6.0, v0=0.5, sigma_v=1.0,
                             w_dist=0.9, w_close=0.7, use_heading=False, w_head=0.3, win=2):
    if not frames: return 0.0
    t_star, idxs, d_min = _find_min_distance_window(frames, npc_id, win=win)
    if t_star < 0 or not idxs: return 0.0
    collided = False; v_closes = []; heads = []
    for t in idxs:
        ego_k = frames[t]["ego"]; npcs_k = frames[t]["npcs"]
        if npc_id not in npcs_k: continue
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
    if collided: return 1.0
    s_dist = math.exp(-min(d_min, 40.0) / D0)
    v_bar = float(np.mean(v_closes)) if v_closes else 0.0
    s_close = 1.0 / (1.0 + math.exp(-(v_bar - v0) / max(sigma_v, 1e-6)))
    s_head = (float(np.mean(heads)) if (use_heading and heads) else 0.0)
    prod = 1.0
    for w, s in ((w_dist, s_dist), (w_close, s_close)): prod *= (1.0 - w * max(0.0, min(1.0, s)))
    if use_heading: prod *= (1.0 - w_head * max(0.0, min(1.0, s_head)))
    y_i = 1.0 - prod
    return max(0.0, min(1.0, y_i))

def build_initial_pose_dataset_minDist(rec, world_map, surrogate,
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
        feats = surrogate._build_feats(rec.map, ego_tf0, npc_tf0)
        y_i = hazard_from_min_distance(F, rec.map, nid,
                                       D0=D0, v0=v0, sigma_v=sigma_v,
                                       w_dist=w_dist, w_close=w_close,
                                       use_heading=use_heading, w_head=w_head, win=win)
        Xs.append(feats); Ys.append(y_i)
        ds0, dd0 = _ego_local_sd(ego_tf0, npc_tf0.location)
        dyaw0 = npc_tf0.rotation.yaw - ego_tf0.rotation.yaw
        while dyaw0 >= 180: dyaw0 -= 360
        while dyaw0 < -180: dyaw0 += 360
        META.append({"npc_id": nid, "ds0": float(ds0), "dd0": float(dd0), "dyaw0": float(dyaw0)})
    if not Xs:
        return np.zeros((0, 8), np.float32), np.zeros((0,), np.float32), []
    return np.asarray(Xs, np.float32), np.asarray(Ys, np.float32), META

def train_mlp_initial_pose_minDist(surrogate,
                                   world_map, rec,
                                   epochs=1, batch=256, lr=1e-3,
                                   D0=6.0, v0=0.5, sigma_v=1.0,
                                   w_dist=0.9, w_close=0.7, use_heading=False, w_head=0.3, win=2):
    X, Y, META = build_initial_pose_dataset_minDist(rec, world_map, surrogate,
                                                    D0=D0, v0=v0, sigma_v=sigma_v,
                                                    w_dist=w_dist, w_close=w_close,
                                                    use_heading=use_heading, w_head=w_head, win=win)
    if X.shape[0] == 0:
        return 0, 0.0, 0, (X, Y, META)
    device = surrogate.device
    model = surrogate.model
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
    if hasattr(surrogate, 'ema_update'):
        try: surrogate.ema_update(tau=0.05)
        except Exception: pass
    return total_steps, avg_loss, len(Y), (X, Y, META)


# =============== Orchestrator：把“训练 + SVGD离线搜索”打包 ===============
class InitialPoseRefiner:
    """
    main 只用三个接口：
      - refiner.train_on_episode(rec)         # 每回合末：用录像数据微调 MLP
      - refiner.refine_position_info(pos_info) # 代际末：对某个场景做 SVGD 微调
      - 可选 refiner.mask_topk(scores)         # 获取 top-k mask
    """
    def __init__(self,
                 world_map: "carla.Map",
                 surrogate,
                 svgd_eps=0.08, svgd_beta=3.0, svgd_grad_eps=0.35,
                 steps_per_case=8, top_k_particles=5,
                 ds_lim=25.0, dd_lim=4.5, dyaw_lim=20.0, min_sep=3.5,
                 train_cfg: Optional[dict] = None):
        self.world_map = world_map
        self.surrogate = surrogate
        self.optimizer = SVGDOptimizer(epsilon=svgd_eps, beta=svgd_beta, grad_eps=svgd_grad_eps)
        self.steps_per_case = steps_per_case
        self.top_k_particles = top_k_particles
        self.ds_lim, self.dd_lim, self.dyaw_lim, self.min_sep = ds_lim, dd_lim, dyaw_lim, min_sep
        self.train_cfg = train_cfg or dict(epochs=2, batch=256, lr=1e-3, D0=6.0, v0=0.5, sigma_v=1.0,
                                           w_dist=0.9, w_close=0.7, use_heading=False, w_head=0.3, win=2)

    # —— 每回合：训练 MLP（标签=回合内各 NPC 的 min-dist hazard）
    def train_on_episode(self, rec):
        return train_mlp_initial_pose_minDist(self.surrogate, self.world_map, rec, **self.train_cfg)

    # —— 代际末：对 position_info 做一次 SVGD 微调（就地修改）
    def refine_position_info(self, position_info: dict):
        problem = InitialPoseSVGDProblem(
            world_map=self.world_map,
            surrogate=self.surrogate,
            scenario_conf=position_info,
            ds_lim=self.ds_lim, dd_lim=self.dd_lim, dyaw_lim=self.dyaw_lim,
            min_sep=self.min_sep, top_k=self.top_k_particles
        )
        for _ in range(self.steps_per_case):
            X = problem.get_particles()
            if X.shape[0] == 0: break
            scores = problem.value_fn(X)
            mask = problem.mask_from_scores(scores, self.top_k_particles)
            X_new = self.optimizer.step(
                X,
                value_fn=problem.value_fn,
                grad_fn=problem.grad_fn,     # 默认 None => 有限差分
                constraint_project=problem.project,
                mask=mask
            )
            problem.set_particles(X_new)
        problem.write_back()

    # —— 可选：外部也能用 mask（如果你想在 main 里自己做）
    def mask_topk(self, scores: np.ndarray, k: Optional[int] = None) -> np.ndarray:
        if k is None:
            k = self.top_k_particles
        m = np.zeros_like(scores, dtype=np.float64)
        if scores.size == 0 or k is None or k <= 0:
            return m
        idx = np.argsort(-scores)[:min(k, scores.size)]
        m[idx] = 1.0
        return m
