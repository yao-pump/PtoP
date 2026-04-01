# npc_surrogate_mlp.py — Enhanced version
# Change summary:
# 1) Introduced target_model (EMA-frozen) exclusively for SVGD's score/score_and_grad; self.model is still used for training (compatible with existing train_mlp_all_pairs).
# 2) Provided ema_update(tau) and freeze_target() for EMA synchronization after training to ensure gradient stability.
# 3) score_and_grad supports optional “logit gradient” (use_logit_grad=True) to enhance gradient signal in extreme probability regions (default False to avoid aggressive behavior).
# 4) Applied safe clamping on map-derived features to reduce numerical instability caused by outliers.
# 5) All inference paths use target_model.eval() with requires_grad=False for parameters, ensuring computation graph is built only w.r.t. (ds,dd,dyaw).

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import carla  # Available at runtime; import failure in offline tools is acceptable
except Exception:  # pragma: no cover
    carla = None

from ptop.utils.geometry import yaw_to_unit, ego_local_sd, relative_yaw_deg


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


# ============== MLP Model ==============

class _HazardMLP(nn.Module):
    def __init__(self, in_dim=8, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1), nn.Sigmoid(),  # Output probability [0,1]
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class SurrogateOptions:
    use_logit_grad: bool = False  # True: use logit(p) gradient to enhance signal in boundary regions
    clamp_lane_lat: float = 3.0   # Clamp for lane center lateral offset (meters)
    clamp_curv: float = 0.5       # Clamp for curvature (1/m); excessively large values are usually noise
    clamp_spd_norm: Tuple[float, float] = (0.0, 2.0)  # Speed normalization range


class NPCHazardMLPSurrogate:
    """
    Input features (consistent with score_and_grad):
      [ s/12, d/1.5, sin(delta_yaw), cos(delta_yaw), lane_lat, curv, is_junc, spd_norm ]

    Usage:
      - Training: operate on self.model (backward compatible with older versions);
      - Inference/gradient computation: via self.target_model (EMA-frozen), through the score / score_and_grad interface.
      - After training, call ema_update(tau) to synchronize to target_model; or call freeze_target() for a direct copy on first use.
    """

    def __init__(self, device: Optional[str] = None, ckpt_path: str = "mlp_frozen.pt", opts: Optional[SurrogateOptions] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = _HazardMLP().to(self.device)           # For training
        self.target_model = _HazardMLP().to(self.device)    # For inference/gradient (frozen)
        self.opts = opts or SurrogateOptions()
        # Attempt to load weights into the training model
        self._load_if_exists(ckpt_path)
        # Initialize target with current training weights and freeze
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

    # --------- Target network (EMA-frozen) management ---------
    @torch.no_grad()
    def ema_update(self, tau: float = 0.05):
        """EMA-sync training weights to target_model: target = (1-tau)*target + tau*model."""
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

    # --------- Core API: score, score_and_grad ---------
    @torch.no_grad()
    def score(self, world_map, ego_tf, npc_tf) -> float:
        feats = self._build_feats(world_map, ego_tf, npc_tf)
        x = torch.from_numpy(feats).float().unsqueeze(0).to(self.device)  # [1,8]
        self.target_model.eval()
        p = self.target_model(x).squeeze().item()
        return float(p)

    def score_and_grad(self, world_map, ego_tf, npc_tf, x_vec, h=0.5):  # h kept for compatibility, not actually used
        """
        x_vec = [ds, dd, dyaw_deg] (meters/meters/degrees)
        Returns (f, grad): f in [0,1]; grad contains partial derivatives w.r.t. ds, dd, dyaw_deg.
        Computation graph is built only w.r.t. x_vec; target_model parameters are frozen.
        Optional: when use_logit_grad=True, returns logit gradient (clipped to prevent divergence).
        """
        # Independent variables (differentiable)
        ds = torch.tensor(float(x_vec[0]), dtype=torch.float32, device=self.device, requires_grad=True)
        dd = torch.tensor(float(x_vec[1]), dtype=torch.float32, device=self.device, requires_grad=True)
        dy = torch.tensor(float(x_vec[2]), dtype=torch.float32, device=self.device, requires_grad=True)  # degree

        # Baseline (current state)
        s0, d0 = ego_local_sd(ego_tf, npc_tf.location)
        base_rel_deg = relative_yaw_deg(ego_tf, npc_tf)

        # Apply (s,d,delta_yaw) increments to features (same scaling as _build_feats)
        s = (torch.tensor(s0, device=self.device) + ds) / 12.0
        d = (torch.tensor(d0, device=self.device) + dd) / 1.5
        rel_rad = torch.tensor(math.radians(base_rel_deg), device=self.device) + dy * (math.pi / 180.0)
        sp = torch.sin(rel_rad)
        cp = torch.cos(rel_rad)

        # Map-derived constant terms (not included in computation graph)
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

        # Model forward pass (gradient only w.r.t. x_vec; parameters frozen)
        self.target_model.eval()
        for p in self.target_model.parameters():
            p.requires_grad_(False)
        y = self.target_model(feats)  # [1,1] in [0,1]
        p = y.squeeze()               # Scalar Tensor

        # Choose gradient form
        if self.opts.use_logit_grad:
            # logit(p) = log(p/(1-p)); d_logit/d_x = d_p/d_x / (p*(1-p))
            # For robustness, apply epsilon guard on the denominator and clip overall gradient.
            logit = torch.log(torch.clamp(p, 1e-6, 1-1e-6)) - torch.log(torch.clamp(1-p, 1e-6, 1.0))
            obj = logit
        else:
            obj = p

        # Backpropagate to (ds, dd, dy)
        self.target_model.zero_grad(set_to_none=True)
        for t in (ds, dd, dy):
            if t.grad is not None:
                t.grad = None
        obj.backward()

        g_ds = ds.grad.item() if ds.grad is not None else 0.0
        g_dd = dd.grad.item() if dd.grad is not None else 0.0
        g_dy = dy.grad.item() if dy.grad is not None else 0.0

        # If using logit, apply gentle clipping to account for the 1/(p*(1-p)) growth trend
        g = np.array([g_ds, g_dd, g_dy], dtype=float)
        g = np.clip(g, -10.0, 10.0)
        return float(p.item()), g

    # --------- Internal: build features consistent with forward pass ---------
    def _build_feats(self, world_map, ego_tf, npc_tf):
        s0, d0 = ego_local_sd(ego_tf, npc_tf.location)
        rel_deg = relative_yaw_deg(ego_tf, npc_tf)
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
