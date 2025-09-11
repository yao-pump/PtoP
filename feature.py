# features.py
import math
import numpy as np
import torch
import carla

def _yaw_rad(tf: carla.Transform) -> float:
    return math.radians(tf.rotation.yaw)

def to_local_SE2(ego_tf: carla.Transform, npc_tf: carla.Transform):
    """
    把 NPC 世界位姿投到 EGO 局部坐标:
    返回 (s, d, psi) 其中 psi = npc_yaw - ego_yaw (弧度)
    """
    ex, ey = ego_tf.location.x, ego_tf.location.y
    nx, ny = npc_tf.location.x, npc_tf.location.y
    dyaw = math.radians(npc_tf.rotation.yaw - ego_tf.rotation.yaw)
    # EGO 前/右向量
    yaw = _yaw_rad(ego_tf)
    f = np.array([math.cos(yaw), math.sin(yaw)])  # forward
    r = np.array([-math.sin(yaw), math.cos(yaw)]) # right
    rel = np.array([nx - ex, ny - ey])
    s = float(np.dot(rel, f))
    d = float(np.dot(rel, r))
    # wrap psi to [-pi, pi]
    psi = math.atan2(math.sin(dyaw), math.cos(dyaw))
    return s, d, psi

def local_pose_batch(ego_tf: carla.Transform, npc_tfs):
    arr = np.zeros((len(npc_tfs), 3), dtype=np.float32)
    for i, tf in enumerate(npc_tfs):
        s, d, psi = to_local_SE2(ego_tf, tf)
        arr[i] = (s, d, psi)
    return arr  # (K,3)

def waypoint_signed_lat(world_map: carla.Map, tf: carla.Transform) -> float:
    """
    返回 NPC 到其所在线中心的有符号横向距离（右为正）
    """
    w = world_map.get_waypoint(tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    if w is None:
        return 0.0
    center = w.transform.location
    right = w.transform.get_right_vector()
    d = carla.Vector3D(tf.location.x - center.x, tf.location.y - center.y, tf.location.z - center.z)
    signed_lat = d.x * right.x + d.y * right.y + d.z * right.z
    return float(signed_lat)

def waypoint_curvature(world_map: carla.Map, tf: carla.Transform, ds: float = 2.0) -> float:
    """
    近似曲率：取前后 ds 的两个路点，三点拟合圆弧半径。返回标量曲率（可为 0）。
    """
    w0 = world_map.get_waypoint(tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    if w0 is None: return 0.0
    w1 = w0.next(ds)[0] if w0.next(ds) else w0
    w_1 = w0.previous(ds)[0] if w0.previous(ds) else w0
    p = np.array([w0.transform.location.x, w0.transform.location.y])
    p1 = np.array([w1.transform.location.x, w1.transform.location.y])
    p_1 = np.array([w_1.transform.location.x, w_1.transform.location.y])
    # 曲率 = |(p1-p0) x (p0-p_1)| / (|p1-p0| * |p0-p_1| * |p1-p_1|)
    a = p1 - p; b = p - p_1; c = p1 - p_1
    den = (np.linalg.norm(a) * np.linalg.norm(b) * np.linalg.norm(c) + 1e-9)
    num = abs(a[0]*b[1] - a[1]*b[0])
    return float(num / den)

def map_context_batch(world: carla.World, tfs):
    wm = world.get_map()
    K = len(tfs)
    lane_lat = np.zeros((K,), dtype=np.float32)
    curvature = np.zeros((K,), dtype=np.float32)
    is_intersection = np.zeros((K,), dtype=np.float32)
    speed_limit = np.zeros((K,), dtype=np.float32)
    for i, tf in enumerate(tfs):
        lane_lat[i] = waypoint_signed_lat(wm, tf)
        curvature[i] = waypoint_curvature(wm, tf)
        w = wm.get_waypoint(tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if w is not None:
            is_intersection[i] = float(w.is_junction)
            try:
                speed_limit[i] = float(w.get_speed_limit())
            except Exception:
                speed_limit[i] = 0.0
        else:
            is_intersection[i] = 0.0
            speed_limit[i] = 0.0
    return {
        "lane_center_dist": lane_lat,           # m
        "curvature": curvature,                 # ~1/m
        "is_intersection": is_intersection,     # {0,1}
        "speed_limit": speed_limit              # km/h or m/s (CARLA单位为km/h? 具体按API返回)
    }

def featurize_particles(particles: torch.Tensor, ctx: dict) -> torch.Tensor:
    """
    particles: [K,3] tensor -> (s, d, psi)
    ctx: dict of torch tensors with shape [K], optional
    return: [K, D]
    """
    s = particles[:, 0:1] / 12.0
    d = particles[:, 1:2] / 1.5
    psi = particles[:, 2:3]
    sp, cp = torch.sin(psi), torch.cos(psi)
    feats = [s, d, sp, cp]
    if ctx is not None:
        for k, v in ctx.items():
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v).float().to(particles.device)
            feats.append(v.view(-1, 1))
    return torch.cat(feats, dim=1)

def scene_fingerprint(world: carla.World, ego_tf: carla.Transform) -> str:
    """
    用于把样本按“场景簇”归档（简单可用：road_id + 是否路口）。
    """
    wm = world.get_map()
    w = wm.get_waypoint(ego_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    if w is None:
        return "unknown"
    key = f"road{w.road_id}_sec{w.section_id}_junction{int(w.is_junction)}"
    return key
