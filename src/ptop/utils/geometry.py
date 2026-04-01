"""Shared geometry helpers for ego-local coordinate transforms and CARLA vector math."""

import math
from typing import Tuple

try:
    import carla
except Exception:
    carla = None


# ---------- Yaw / angle utilities ----------

def yaw_to_unit(yaw_deg: float) -> Tuple[float, float]:
    """Convert yaw in degrees to a (cos, sin) unit vector."""
    r = math.radians(yaw_deg)
    return math.cos(r), math.sin(r)


def wrap_yaw_deg(a: float) -> float:
    """Normalize angle to (-180, 180] degrees."""
    while a <= -180.0:
        a += 360.0
    while a > 180.0:
        a -= 360.0
    return a


def relative_yaw_deg(ego_tf, npc_tf) -> float:
    """Relative yaw of NPC w.r.t. EGO in degrees, wrapped to [-180, 180)."""
    dy = npc_tf.rotation.yaw - ego_tf.rotation.yaw
    while dy >= 180:
        dy -= 360
    while dy < -180:
        dy += 360
    return dy


# ---------- Ego-local coordinate transforms ----------

def ego_local_sd(ego_tf, pt) -> Tuple[float, float]:
    """
    Project a point into the ego-local frame.

    Returns (s, d) where:
      s = longitudinal (forward positive)
      d = lateral (right positive)
    """
    dx = pt.x - ego_tf.location.x
    dy = pt.y - ego_tf.location.y
    cy, sy = yaw_to_unit(ego_tf.rotation.yaw)
    s = dx * cy + dy * sy
    d = -dx * sy + dy * cy
    return s, d


def to_local(ego_tf, loc) -> Tuple[float, float]:
    """
    Transform a location into the ego-local frame using negative yaw rotation.

    Returns (lx, ly) — forward/left convention used by the ART risk scorer.
    Note: this uses cos(-yaw)/sin(-yaw), which differs from ego_local_sd's convention.
    """
    dx = loc.x - ego_tf.location.x
    dy = loc.y - ego_tf.location.y
    yaw = math.radians(ego_tf.rotation.yaw)
    c, s = math.cos(-yaw), math.sin(-yaw)
    return c * dx - s * dy, s * dx + c * dy


def vel_local(ego_tf, vel) -> Tuple[float, float]:
    """Transform a velocity vector into the ego-local frame."""
    yaw = math.radians(ego_tf.rotation.yaw)
    c, s = math.cos(-yaw), math.sin(-yaw)
    return c * vel.x - s * vel.y, s * vel.x + c * vel.y


def decompose_to_local(ego_tf, npc_tf) -> Tuple[float, float, float]:
    """Decompose NPC pose into ego-local (ds, dd, dyaw_deg)."""
    ds, dd = ego_local_sd(ego_tf, npc_tf.location)
    dyaw = wrap_yaw_deg(npc_tf.rotation.yaw - ego_tf.rotation.yaw)
    return ds, dd, dyaw


def apply_local_offset(ego_tf, ds: float, dd: float, dyaw_deg: float):
    """Given ego_tf and local offsets (ds, dd, dyaw), build a world Transform."""
    cy, sy = yaw_to_unit(ego_tf.rotation.yaw)
    fx, fy = cy, sy
    rx, ry = -sy, cy
    loc = carla.Location(
        x=ego_tf.location.x + ds * fx + dd * rx,
        y=ego_tf.location.y + ds * fy + dd * ry,
        z=ego_tf.location.z,
    )
    yaw = wrap_yaw_deg(ego_tf.rotation.yaw + dyaw_deg)
    rot = carla.Rotation(pitch=ego_tf.rotation.pitch, yaw=yaw, roll=ego_tf.rotation.roll)
    return carla.Transform(loc, rot)


# ---------- CARLA vector math ----------

def vec_norm(v) -> float:
    """Euclidean norm of a carla.Vector3D."""
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def unit_vec(a, b):
    """Unit vector from location a to location b."""
    dx, dy, dz = (b.x - a.x), (b.y - a.y), (b.z - a.z)
    n = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-9
    return carla.Vector3D(dx / n, dy / n, dz / n)


def dot(a, b) -> float:
    """Dot product of two carla.Vector3D."""
    return a.x * b.x + a.y * b.y + a.z * b.z


def spd_and_vec(actor) -> Tuple[float, object]:
    """Return (speed_scalar, velocity_vector) for a CARLA actor."""
    v = actor.get_velocity()
    return vec_norm(v), v
