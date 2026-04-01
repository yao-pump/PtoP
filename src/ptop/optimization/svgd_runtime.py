#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Runtime NPC SVGD (with rule-based minimum separation after each step).

- Works on a single scenario's `position_info` (in-place refine).
- Supports two encodings of surrounding_info:
  1) list of dict: [{"transform": carla.Transform, "type": "car"/"bicycle"/"pedestrian"}, ...]
  2) dict of parallel lists: {"transform": [carla.Transform, ...], "type": ["car", ...]}

- Surrogate interface expected (any of the following signatures are compatible):
    A) score_and_grad(map, ego_tf, npc_tf, x_vec, eps=None) -> (float, (dF/ds, dF/dd, dF/dyaw))
    B) score_and_grad(map, ego_tf, npc_tf, x_vec)          -> (float, grad)
    C) score_and_grad(map, ego_tf, npc_tf)                 -> (float, grad)
  where (ds,dd,dyaw) are local variables relative to EGO. If none of the above match, gradients are estimated via numerical finite differences.

- Adds a rule-based guard `_enforce_min_separation` that keeps particles
  at least `min_sep` apart in (ds, dd) after EVERY SVGD step (and at the end).
"""

import math
import random
from typing import Optional, Tuple

import numpy as np
import carla


# ---------- small geom helpers ----------

def _yaw_to_unit(yaw_deg: float) -> Tuple[float, float]:
    r = math.radians(yaw_deg)
    return math.cos(r), math.sin(r)


def _ego_local_sd(ego_tf: carla.Transform, pt: carla.Location) -> Tuple[float, float]:
    dx = pt.x - ego_tf.location.x
    dy = pt.y - ego_tf.location.y
    cy, sy = _yaw_to_unit(ego_tf.rotation.yaw)
    s = dx * cy + dy * sy
    d = -dx * sy + dy * cy
    return s, d


def _wrap_yaw_deg(a: float) -> float:
    # map to (-180, 180]
    while a <= -180.0:
        a += 360.0
    while a > 180.0:
        a -= 360.0
    return a


def _apply_local_offset(ego_tf: carla.Transform, ds: float, dd: float, dyaw_deg: float) -> carla.Transform:
    """Given ego_tf and local offsets (ds, dd, dyaw), build world transform."""
    cy, sy = _yaw_to_unit(ego_tf.rotation.yaw)
    # forward (s) = (cy, sy), right (d) = (-sy, cy)
    fx, fy = cy, sy
    rx, ry = -sy, cy
    loc = carla.Location(
        x=ego_tf.location.x + ds * fx + dd * rx,
        y=ego_tf.location.y + ds * fy + dd * ry,
        z=ego_tf.location.z  # keep same Z
    )
    yaw = _wrap_yaw_deg(ego_tf.rotation.yaw + dyaw_deg)
    rot = carla.Rotation(pitch=ego_tf.rotation.pitch, yaw=yaw, roll=ego_tf.rotation.roll)
    return carla.Transform(loc, rot)


def _decompose_to_local(ego_tf: carla.Transform, npc_tf: carla.Transform) -> Tuple[float, float, float]:
    ds, dd = _ego_local_sd(ego_tf, npc_tf.location)
    dyaw = _wrap_yaw_deg(npc_tf.rotation.yaw - ego_tf.rotation.yaw)
    return ds, dd, dyaw


class RuntimeNPCSVGD:
    def __init__(
        self,
        world_map: carla.Map,
        surrogate,
        top_k: int = 20,
        steps: int = 8,
        epsilon: float = 0.25,
        beta: float = 1.0,
        grad_eps: Optional[float] = 0.5,
        ds_lim: float = 15.0,
        dd_lim: float = 3.0,
        dyaw_lim: float = 15.0,
        min_sep: float = 1.5,
        yaw_scale: float = 0.25,  # scaling factor for yaw dim in kernel distance
        rng_seed: Optional[int] = None,
    ):
        """
        Args:
            top_k: number of NPCs (particles) to refine (picked by highest hazard).
            steps: SVGD steps.
            epsilon: step size (learning rate) for SVGD.
            beta: kernel repulsive term weight.
            grad_eps: finite-diff epsilon forwarded to surrogate (can be None if surrogate ignores).
            ds_lim/dd_lim/dyaw_lim: search box limits on (ds, dd, dyaw).
            min_sep: rule-based minimum separation in (ds,dd) between any two particles.
            yaw_scale: distance scaling used in the RBF kernel for yaw dimension.
        """
        self.map = world_map
        self.surrogate = surrogate
        self.top_k = int(top_k)
        self.steps = int(steps)
        self.epsilon = float(epsilon)
        self.beta = float(beta)
        self.grad_eps = grad_eps
        self.ds_lim = float(ds_lim)
        self.dd_lim = float(dd_lim)
        self.dyaw_lim = float(dyaw_lim)
        self.min_sep = float(min_sep)
        self.yaw_scale = float(yaw_scale)

        if rng_seed is not None:
            random.seed(rng_seed)
            np.random.seed(rng_seed)

    # ---------- utils for surrounding_info encoding ----------

    @staticmethod
    def _get_len(surrounding) -> int:
        if isinstance(surrounding, list):
            return len(surrounding)
        return len(surrounding.get("transform", []))

    @staticmethod
    def _get_item(surrounding, i: int):
        """Return (transform, type_str)"""
        if isinstance(surrounding, list):
            return surrounding[i]["transform"], str(surrounding[i]["type"]).lower()
        return surrounding["transform"][i], str(surrounding["type"][i]).lower()

    @staticmethod
    def _set_transform(surrounding, i: int, new_tf: carla.Transform) -> None:
        if isinstance(surrounding, list):
            surrounding[i]["transform"] = new_tf
        else:
            surrounding["transform"][i] = new_tf

    # ---------- kernel helpers ----------

    def _pairwise_sqdist(self, X: np.ndarray) -> np.ndarray:
        """
        RBF kernel distance with anisotropic scaling on yaw:
        d^2 = (ds_i - ds_j)^2 + (dd_i - dd_j)^2 + (yaw_scale*(dyaw_i - dyaw_j))^2
        """
        ds = X[:, 0][:, None]
        dd = X[:, 1][:, None]
        dy = X[:, 2][:, None] * self.yaw_scale

        dsd = ds - ds.T
        ddd = dd - dd.T
        dyd = dy - dy.T
        return dsd * dsd + ddd * ddd + dyd * dyd

    def _rbf_kernel(self, X: np.ndarray):
        """Return K (NxN) and bandwidth h."""
        D2 = self._pairwise_sqdist(X)
        triu = D2[np.triu_indices_from(D2, k=1)]
        med = np.median(triu) if triu.size > 0 else 1.0
        h = med / (np.log(X.shape[0] + 1.0) + 1e-9)
        h = max(h, 1e-6)
        K = np.exp(-D2 / h)
        return K, h

    def _phi(self, X: np.ndarray, G: np.ndarray):
        """
        Compute SVGD velocity field φ(X):
            φ_i = 1/N * sum_j [ K_ij * G_j + beta * ∇_{x_j} K_ij ]
        where ∇_{x_j} K_ij = K_ij * (x_j - x_i) / h
        Returns: φ (Nx3), and (K@G) and gradK (for logging norms).
        """
        N, _ = X.shape
        if N == 0:
            Z = np.zeros_like(X)
            return Z, Z, Z

        K, h = self._rbf_kernel(X)        # K NxN
        KG = K @ G                         # Nx3

        # repulsive term: sum_j K_ij * (X_j - X_i) / h
        Xj_minus_Xi = X[None, :, :] - X[:, None, :]  # [i,j,d] = X_j - X_i
        weighted = K[:, :, None] * Xj_minus_Xi       # [i,j,d]
        sum_term = weighted.sum(axis=1)              # [i,d]
        gradK = sum_term / max(h, 1e-12)

        phi = (KG + self.beta * gradK) / float(N)
        return phi, KG, gradK

    # ---------- rule-based minimum separation ----------

    def _enforce_min_separation(self, X: np.ndarray) -> np.ndarray:
        """
        Ensure any pair (i,j) has sqrt((ds_i-ds_j)^2 + (dd_i-dd_j)^2) >= min_sep.
        Iteratively push pairs apart along their (ds,dd) direction; yaw is untouched.
        """
        N = X.shape[0]
        if N <= 1:
            return X

        max_iter = 6
        for _ in range(max_iter):
            moved = False
            for i in range(N):
                for j in range(i + 1, N):
                    dx = X[i, 0] - X[j, 0]
                    dy = X[i, 1] - X[j, 1]
                    dist = math.hypot(dx, dy)
                    if dist < self.min_sep:
                        overlap = (self.min_sep - dist) + 1e-9
                        ux = dx / (dist + 1e-9)
                        uy = dy / (dist + 1e-9)
                        push = 0.5 * overlap
                        X[i, 0] += ux * push
                        X[i, 1] += uy * push
                        X[j, 0] -= ux * push
                        X[j, 1] -= uy * push
                        moved = True
            if not moved:
                break
            # box-clip s/d/yaw
            X[:, 0] = np.clip(X[:, 0], -self.ds_lim, self.ds_lim)
            X[:, 1] = np.clip(X[:, 1], -self.dd_lim, self.dd_lim)
            X[:, 2] = np.clip(X[:, 2], -self.dyaw_lim, self.dyaw_lim)

        return X

    # ---------- main entry ----------

    def refine_position_info(self, position_info: dict) -> None:
        """
        In-place refine:
            - Pick top_k NPCs by hazard at their current transforms
            - Convert to particles (ds,dd,dyaw) in ego frame
            - SVGD updates with rule-based min-separation after each step
            - Write back refined transforms
        """
        ego_tf: carla.Transform = position_info["ego_transform"]
        surrounding = position_info["surrounding_info"]
        n_total = self._get_len(surrounding)
        if n_total == 0:
            return

        # 1) collect candidates (prefer vehicles; keep bicycles; skip pedestrians here)
        idxs, tfs, scores = [], [], []
        for i in range(n_total):
            tf_i, typ_i = self._get_item(surrounding, i)
            typ_i = str(typ_i).lower()
            if typ_i not in ("car", "bicycle", "vehicle", "auto", "vehicle.tesla.model3","pedestrian"):
                continue
            try:
                Fi = float(self.surrogate.score(self.map, ego_tf, tf_i))
            except Exception:
                Fi = 0.0
            idxs.append(i); tfs.append(tf_i); scores.append(Fi)

        if not idxs:
            return

        # 2) pick top_k by score
        order = np.argsort(-np.asarray(scores))
        sel = order[:min(self.top_k, len(order))]
        sel_idxs = [idxs[k] for k in sel]
        sel_tfs = [tfs[k] for k in sel]

        # 3) build particle matrix X [M,3] = (ds, dd, dyaw)
        X_list = []
        for tf in sel_tfs:
            ds, dd, dy = _decompose_to_local(ego_tf, tf)
            ds = float(np.clip(ds, -self.ds_lim, self.ds_lim))
            dd = float(np.clip(dd, -self.dd_lim, self.dd_lim))
            dy = float(np.clip(dy, -self.dyaw_lim, self.dyaw_lim))
            X_list.append([ds, dd, dy])
        X = np.asarray(X_list, dtype=np.float32)
        M = X.shape[0]
        if M == 0:
            return

        # 4) SVGD iterations
        for step in range(1, self.steps + 1):
            # Evaluate grad at current particle locations
            F = np.zeros((M,), dtype=np.float32)
            G = np.zeros((M, 3), dtype=np.float32)

            for k in range(M):
                ds_k, dd_k, dy_k = float(X[k, 0]), float(X[k, 1]), float(X[k, 2])
                tf_k = _apply_local_offset(ego_tf, ds_k, dd_k, dy_k)
                x_vec = (ds_k, dd_k, dy_k)

                # ---- Compatible with multiple score_and_grad signatures; falls back to numerical finite differences on failure ----
                try:
                    # A) with x_vec and eps
                    Fk, gradk = self.surrogate.score_and_grad(self.map, ego_tf, tf_k, x_vec, eps=self.grad_eps)
                except TypeError:
                    try:
                        # B) with x_vec but without eps
                        Fk, gradk = self.surrogate.score_and_grad(self.map, ego_tf, tf_k, x_vec)
                    except TypeError:
                        try:
                            # C) only (map, ego_tf, npc_tf)
                            Fk, gradk = self.surrogate.score_and_grad(self.map, ego_tf, tf_k)
                        except Exception:
                            # D) Ultimate fallback: numerical central differences
                            Fk = float(self.surrogate.score(self.map, ego_tf, tf_k))
                            fd = self.grad_eps or 0.25

                            def sc(s, d, y):
                                t = _apply_local_offset(ego_tf, float(s), float(d), float(y))
                                return float(self.surrogate.score(self.map, ego_tf, t))

                            g_s = (sc(ds_k + fd, dd_k, dy_k) - sc(ds_k - fd, dd_k, dy_k)) / (2 * fd)
                            g_d = (sc(ds_k, dd_k + fd, dy_k) - sc(ds_k, dd_k - fd, dy_k)) / (2 * fd)
                            g_y = (sc(ds_k, dd_k, dy_k + fd) - sc(ds_k, dd_k, dy_k - fd)) / (2 * fd)
                            gradk = (g_s, g_d, g_y)
                except Exception:
                    # Fall back to numerical finite differences for other exceptions to prevent G=0 degeneracy
                    Fk = float(self.surrogate.score(self.map, ego_tf, tf_k))
                    fd = self.grad_eps or 0.25

                    def sc(s, d, y):
                        t = _apply_local_offset(ego_tf, float(s), float(d), float(y))
                        return float(self.surrogate.score(self.map, ego_tf, t))

                    g_s = (sc(ds_k + fd, dd_k, dy_k) - sc(ds_k - fd, dd_k, dy_k)) / (2 * fd)
                    g_d = (sc(ds_k, dd_k + fd, dy_k) - sc(ds_k, dd_k - fd, dy_k)) / (2 * fd)
                    g_y = (sc(ds_k, dd_k, dy_k + fd) - sc(ds_k, dd_k, dy_k - fd)) / (2 * fd)
                    gradk = (g_s, g_d, g_y)

                F[k] = float(Fk)
                G[k, 0] = float(gradk[0])
                G[k, 1] = float(gradk[1])
                G[k, 2] = float(gradk[2])

            # Compute φ, and log norms
            phi, KG, gradK = self._phi(X, G)
            norm_KG = float(np.linalg.norm(KG))
            norm_gK = float(np.linalg.norm(gradK))
            ratio = norm_KG / (norm_gK + 1e-12)
            meanF = float(F.mean()) if M > 0 else 0.0

            # Update
            X = X + self.epsilon * phi

            # Clip to box
            X[:, 0] = np.clip(X[:, 0], -self.ds_lim, self.ds_lim)
            X[:, 1] = np.clip(X[:, 1], -self.dd_lim, self.dd_lim)
            X[:, 2] = np.clip(X[:, 2], -self.dyaw_lim, self.dyaw_lim)

            # >>> Rule-based minimum separation AFTER the gradient step <<<
            X = self._enforce_min_separation(X)

            # Log (match your format)
            print(f"[SVGD] step {step}/{self.steps} | ||K·G||={norm_KG:.4f}  ||∇K||={norm_gK:.4f}  ratio={ratio:.3e}  meanF={meanF:.3f}")

        # 5) Final separation one more time (safety)
        X = self._enforce_min_separation(X)

        # 6) Write back refined transforms
        for k, idx in enumerate(sel_idxs):
            tf_new = _apply_local_offset(ego_tf, float(X[k, 0]), float(X[k, 1]), float(X[k, 2]))
            self._set_transform(surrounding, idx, tf_new)
