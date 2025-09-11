# svgd_step.py
import math
import torch
from feature import featurize_particles

def _rbf_kernel(x: torch.Tensor):
    # x: [K, D]
    with torch.no_grad():
        pd = torch.cdist(x, x, p=2)
        h = (torch.median(pd) ** 2).clamp(min=1e-6) * 3.0  # 中位数启发式 × 3
    k = torch.exp(-torch.cdist(x, x, p=2) ** 2 / (h + 1e-12))
    return k, h

@torch.no_grad()
def _clip_delta(base: torch.Tensor, new: torch.Tensor,
                clip_s=12.0, clip_d=1.5, clip_psi=math.radians(12)):
    ds = (new[:, 0] - base[:, 0]).clamp(-clip_s, clip_s)
    dd = (new[:, 1] - base[:, 1]).clamp(-clip_d, clip_d)
    dp = (new[:, 2] - base[:, 2]).clamp(-clip_psi, clip_psi)
    out = torch.stack([base[:, 0] + ds, base[:, 1] + dd, base[:, 2] + dp], dim=1)
    return out

def svgd_update_once(particles: torch.Tensor,
                     ctx: dict,
                     model,                      # HazardMLP (eval 模式)
                     epsilon=0.4):
    """
    对当前粒子集合做一次 SVGD 更新（不含地图投影，纯 (s,d,psi) 空间）
    particles: [K,3] (requires_grad=True)
    ctx: dict of tensors [K]
    return: new_particles, delta_Fsum
    """
    feats = featurize_particles(particles, ctx)           # [K,D]
    F_vals = model(feats)                                 # [K]
    F_sum = F_vals.sum()
    grads = torch.autograd.grad(F_sum, particles, create_graph=False, retain_graph=False)[0]  # [K,3]

    with torch.no_grad():
        x = particles.detach()
        k, h = _rbf_kernel(x)                             # [K,K], scalar
        # grad term
        grad_term = (k @ grads) / x.shape[0]              # [K,3]
        # repulsive term: ∑_j ∇_{x_j} k(x_i, x_j) = ∑_j (-2/h) k_ij (x_j - x_i)
        diff = x.unsqueeze(1) - x.unsqueeze(0)            # [K,K,3]
        repulse = ((-2.0 / (h + 1e-12)) * k.unsqueeze(-1) * diff).mean(dim=1)  # [K,3]
        phi = grad_term + repulse
        new = x + epsilon * phi

    with torch.no_grad():
        F_before = F_vals.sum().item()
        F_after = model(featurize_particles(new, ctx)).sum().item()

    new = _clip_delta(particles, new)
    return new, (F_after - F_before)

def svgd_update_steps(particles: torch.Tensor,
                      ctx: dict,
                      model,
                      steps=(0.4, 0.25),
                      accept_tau: float = 0.02):
    """
    多步（通常 1–2 步）SVGD；边际增益不足则提前停，最终若总增益 < 阈值则回滚。
    """
    base = particles.clone()
    total_gain = 0.0
    x = particles
    for eps in steps:
        x.requires_grad_(True)
        x_new, dF = svgd_update_once(x, ctx, model, epsilon=eps)
        total_gain += dF
        x = x_new.detach()
        if dF < accept_tau * 0.5:  # 边际收益太小，早停
            break
    if total_gain < accept_tau:
        return base, 0.0
    return x, total_gain
