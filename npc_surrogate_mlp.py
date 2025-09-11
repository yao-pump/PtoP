# npc_surrogate_mlp.py — 增强版
# 变更摘要：
# 1) 引入 target_model（EMA 冻结）专供 SVGD 的 score/score_and_grad 使用；self.model 仍用于训练（与现有 train_mlp_all_pairs 兼容）。
# 2) 提供 ema_update(tau) 与 freeze_target()，训练后做 EMA 同步，保证梯度稳定。
# 3) score_and_grad 支持可选“logit 梯度”（use_logit_grad=True），增强极端概率区间的梯度信号（默认 False，避免过激）。
# 4) 对地图派生特征做安全钳位（clamp），减少异常值导致的数值不稳。
# 5) 所有推理路径均使用 target_model.eval() 且参数 requires_grad=False，确保仅对 (ds,dd,dyaw) 建图。

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import carla  # 运行期可用；若离线工具导入失败不影响
except Exception:  # pragma: no cover
    carla = None


# ============== 几何辅助，与主程序保持一致 ==============

def _yaw_to_unit(yaw_deg: float):
    r = math.radians(yaw_deg)
    return math.cos(r), math.sin(r)


def _relative_yaw_deg(ego_tf, npc_tf) -> float:
    dy = npc_tf.rotation.yaw - ego_tf.rotation.yaw
    while dy >= 180:
        dy -= 360
    while dy < -180:
        dy += 360
    return dy


def _ego_local_sd(ego_tf, loc):
    dx = loc.x - ego_tf.location.x
    dy = loc.y - ego_tf.location.y
    cy, sy = _yaw_to_unit(ego_tf.rotation.yaw)
    s = dx * cy + dy * sy
    d = -dx * sy + dy * cy
    return s, d


def _lane_center_offset(world_map, npc_tf) -> float:
    try:
        wp = world_map.get_waypoint(
            npc_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
    except Exception:
        wp = None
    if not wp:
        return 0.0
    center = wp.transform.location
    right = wp.transform.get_right_vector()
    dvx = npc_tf.location.x - center.x
    dvy = npc_tf.location.y - center.y
    dvz = npc_tf.location.z - center.z
    return dvx * right.x + dvy * right.y + dvz * right.z


def _curvature_approx(world_map, npc_tf, ds=2.0) -> float:
    try:
        wp = world_map.get_waypoint(
            npc_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
    except Exception:
        wp = None
    if not wp:
        return 0.0
    try:
        prevs = wp.previous(ds)
        nexts = wp.next(ds)
        if not prevs or not nexts:
            return 0.0
        yaw0 = prevs[0].transform.rotation.yaw
        yaw2 = nexts[0].transform.rotation.yaw
        dyaw = math.radians(((yaw2 - yaw0 + 180) % 360) - 180)
        return float(abs(dyaw) / (2.0 * ds))
    except Exception:
        return 0.0


# ============== MLP 模型 ==============

class _HazardMLP(nn.Module):
    def __init__(self, in_dim=8, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1), nn.Sigmoid(),  # 输出概率 [0,1]
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class SurrogateOptions:
    use_logit_grad: bool = False  # True: 使用 logit(p) 的梯度，增强边界区间的信号
    clamp_lane_lat: float = 3.0   # 车道中心横向偏移的钳位（米）
    clamp_curv: float = 0.5       # 曲率的钳位（1/m），过大通常是噪声
    clamp_spd_norm: Tuple[float, float] = (0.0, 2.0)  # 速度归一化范围


class NPCHazardMLPSurrogate:
    """
    输入特征（与 score_and_grad 一致）：
      [ s/12, d/1.5, sin(Δyaw), cos(Δyaw), lane_lat, curv, is_junc, spd_norm ]

    用法：
      - 训练：操作 self.model（与旧版兼容）；
      - 推理/求梯度：经由 self.target_model（EMA 冻结），接口 score / score_and_grad。
      - 训练后调用 ema_update(tau) 同步至 target_model；或首次可调用 freeze_target() 直接拷贝。
    """

    def __init__(self, device: Optional[str] = None, ckpt_path: str = "mlp_frozen.pt", opts: Optional[SurrogateOptions] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = _HazardMLP().to(self.device)           # 训练用
        self.target_model = _HazardMLP().to(self.device)    # 推理/梯度用（冻结）
        self.opts = opts or SurrogateOptions()
        # 尝试加载到训练模型
        self._load_if_exists(ckpt_path)
        # 初始化 target 为当前训练权重，并冻结
        self.freeze_target(copy_from_model=True)

    # --------- I/O ---------
    def _load_if_exists(self, path):
        if path and os.path.isfile(path):
            try:
                payload = torch.load(path, map_location=self.device)
                state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
                self.model.load_state_dict(state)
                self.model.eval()
                print(f"[SVGD-MLP] loaded weights from {path}")
            except Exception as e:
                print(f"[SVGD-MLP] failed to load {path}: {e}. Using random init.")
        else:
            print("[SVGD-MLP] no checkpoint found, using randomly-initialized MLP")

    def save(self, path="mlp_frozen.pt", save_target: bool = False):
        try:
            to_save = self.target_model if save_target else self.model
            torch.save({"state_dict": to_save.state_dict()}, path)
            print(f"[SVGD-MLP] saved {'target_model' if save_target else 'model'} to {path}")
        except Exception as e:
            print(f"[SVGD-MLP] save error: {e}")

    # --------- 目标网络（EMA 冻结）管理 ---------
    @torch.no_grad()
    def ema_update(self, tau: float = 0.05):
        """以 EMA 同步训练权重到 target_model： target = (1-tau)*target + tau*model."""
        for p_t, p_s in zip(self.target_model.parameters(), self.model.parameters()):
            p_t.copy_(p_t * (1.0 - tau) + p_s * tau)
        self._freeze_(self.target_model)

    @torch.no_grad()
    def freeze_target(self, copy_from_model: bool = False):
        if copy_from_model:
            self.target_model.load_state_dict(self.model.state_dict())
        self._freeze_(self.target_model)

    @staticmethod
    def _freeze_(m: nn.Module):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)

    # --------- 核心 API：score，score_and_grad ---------
    @torch.no_grad()
    def score(self, world_map, ego_tf, npc_tf) -> float:
        feats = self._build_feats(world_map, ego_tf, npc_tf)
        x = torch.from_numpy(feats).float().unsqueeze(0).to(self.device)  # [1,8]
        self.target_model.eval()
        p = self.target_model(x).squeeze().item()
        return float(p)

    def score_and_grad(self, world_map, ego_tf, npc_tf, x_vec, h=0.5):  # h 保留兼容，无实际使用
        """
        x_vec = [ds, dd, dyaw_deg]（米/米/度）
        返回 (f, grad): f in [0,1]; grad 对 ds, dd, dyaw_deg 的偏导。
        仅构建 w.r.t. x_vec 的计算图，target_model 参数冻结。
        可选：use_logit_grad=True 时返回 logit 的梯度（做裁剪以防发散）。
        """
        # 自变量（可导）
        ds = torch.tensor(float(x_vec[0]), dtype=torch.float32, device=self.device, requires_grad=True)
        dd = torch.tensor(float(x_vec[1]), dtype=torch.float32, device=self.device, requires_grad=True)
        dy = torch.tensor(float(x_vec[2]), dtype=torch.float32, device=self.device, requires_grad=True)  # degree

        # 基准（当前状态）
        s0, d0 = _ego_local_sd(ego_tf, npc_tf.location)
        base_rel_deg = _relative_yaw_deg(ego_tf, npc_tf)

        # 将 (s,d,Δyaw) 的“增量”套到特征上（和 _build_feats 完全一致的缩放）
        s = (torch.tensor(s0, device=self.device) + ds) / 12.0
        d = (torch.tensor(d0, device=self.device) + dd) / 1.5
        rel_rad = torch.tensor(math.radians(base_rel_deg), device=self.device) + dy * (math.pi / 180.0)
        sp = torch.sin(rel_rad)
        cp = torch.cos(rel_rad)

        # 地图常量项（不进图）
        lane_lat = float(_lane_center_offset(world_map, npc_tf))
        lane_lat = float(max(-self.opts.clamp_lane_lat, min(self.opts.clamp_lane_lat, lane_lat)))
        curv = float(_curvature_approx(world_map, npc_tf, ds=2.0))
        curv = float(max(-self.opts.clamp_curv, min(self.opts.clamp_curv, curv)))
        try:
            wp = world_map.get_waypoint(npc_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        except Exception:
            wp = None
        is_junc = 1.0 if (wp and wp.is_junction) else 0.0
        spd = 0.0
        if wp:
            try:
                spd = float(getattr(wp, "speed_limit", 0.0))
            except Exception:
                try:
                    spd = float(wp.get_speed_limit())
                except Exception:
                    spd = 0.0
        lo, hi = self.opts.clamp_spd_norm
        spd_norm = float(np.clip(spd / 30.0, lo, hi))

        feats = torch.stack([
            s, d,
            sp, cp,
            torch.tensor(lane_lat, dtype=torch.float32, device=self.device),
            torch.tensor(curv, dtype=torch.float32, device=self.device),
            torch.tensor(is_junc, dtype=torch.float32, device=self.device),
            torch.tensor(spd_norm, dtype=torch.float32, device=self.device),
        ], dim=0).unsqueeze(0)  # [1,8]

        # 模型前向（仅对 x_vec 求导，参数冻结）
        self.target_model.eval()
        for p in self.target_model.parameters():
            p.requires_grad_(False)
        y = self.target_model(feats)  # [1,1] in [0,1]
        p = y.squeeze()               # 标量 Tensor

        # 选择梯度形态
        if self.opts.use_logit_grad:
            # logit(p) = log(p/(1-p))；∂logit/∂x = ∂p/∂x / (p*(1-p))
            # 为稳健性，对分母做 ε 防护并裁剪整体梯度。
            logit = torch.log(torch.clamp(p, 1e-6, 1-1e-6)) - torch.log(torch.clamp(1-p, 1e-6, 1.0))
            obj = logit
        else:
            obj = p

        # 反向到 (ds,dd,dy)
        self.target_model.zero_grad(set_to_none=True)
        for t in (ds, dd, dy):
            if t.grad is not None:
                t.grad = None
        obj.backward()

        g_ds = ds.grad.item() if ds.grad is not None else 0.0
        g_dd = dd.grad.item() if dd.grad is not None else 0.0
        g_dy = dy.grad.item() if dy.grad is not None else 0.0

        # 若用 logit，按 1/(p*(1-p)) 的增长趋势，做温和裁剪
        g = np.array([g_ds, g_dd, g_dy], dtype=float)
        g = np.clip(g, -10.0, 10.0)
        return float(p.item()), g

    # --------- 内部：构造与前向一致的特征 ---------
    def _build_feats(self, world_map, ego_tf, npc_tf):
        s0, d0 = _ego_local_sd(ego_tf, npc_tf.location)
        rel_deg = _relative_yaw_deg(ego_tf, npc_tf)
        rel_rad = math.radians(rel_deg)
        lane_lat = _lane_center_offset(world_map, npc_tf)
        lane_lat = float(max(-self.opts.clamp_lane_lat, min(self.opts.clamp_lane_lat, lane_lat)))
        curv = _curvature_approx(world_map, npc_tf, ds=2.0)
        curv = float(max(-self.opts.clamp_curv, min(self.opts.clamp_curv, curv)))
        try:
            wp = world_map.get_waypoint(npc_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        except Exception:
            wp = None
        is_junc = 1.0 if (wp and wp.is_junction) else 0.0
        spd = 0.0
        if wp:
            try:
                spd = float(getattr(wp, "speed_limit", 0.0))
            except Exception:
                try:
                    spd = float(wp.get_speed_limit())
                except Exception:
                    spd = 0.0
        lo, hi = self.opts.clamp_spd_norm
        spd_norm = float(np.clip(spd / 30.0, lo, hi))
        feats = np.array([
            s0 / 12.0,
            d0 / 1.5,
            math.sin(rel_rad),
            math.cos(rel_rad),
            lane_lat,
            curv,
            is_junc,
            spd_norm,
        ], dtype=np.float32)
        return feats
