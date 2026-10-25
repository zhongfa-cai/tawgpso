"""Shared objective, feasibility, collision, clearance, altitude, AGL, smoothness, and waypoint-complexity evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from algorithm.tawg_pso import AdaptiveWaypointConfig, adaptive_waypoint_penalty, build_path_from_xg, summarize_gate_info

Array = np.ndarray
Obstacle = Tuple[float, float, float, float, str]
_OBSTACLE_ARRAY_CACHE = {}
CYL_PLAT_FRAC: float = 0.74
CYL_SHOULDER_FR: float = 0.3
CYL_SOFT_TOP: float = 0.06
DOME_BLEND: float = 0.5
DOME_P: float = 1.75
DOME_Q: float = 1.28

def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)

def _smoothstep_array(x: Array) -> Array:
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

def _unpack_obstacle(ob) -> Obstacle:
    if hasattr(ob, 'center') and hasattr(ob, 'radius') and hasattr(ob, 'height'):
        return (float(ob.center[0]), float(ob.center[1]), float(ob.radius), float(ob.height), str(getattr(ob, 'kind', 'cone')))
    cx, cy, r, h, kind = ob
    return (float(cx), float(cy), float(r), float(h), str(kind))

def _obstacle_arrays(obstacles: Iterable[Obstacle]):
    cache_key = id(obstacles) if isinstance(obstacles, (list, tuple)) else None
    if cache_key is not None:
        cached = _OBSTACLE_ARRAY_CACHE.get(cache_key)
        if cached is not None and cached[0] == len(obstacles):
            return cached[1]
    obs = [_unpack_obstacle(ob) for ob in obstacles]
    if not obs:
        if cache_key is not None:
            _OBSTACLE_ARRAY_CACHE[cache_key] = (0, None)
        return None
    cx = np.asarray([ob[0] for ob in obs], dtype=float)
    cy = np.asarray([ob[1] for ob in obs], dtype=float)
    r = np.maximum(np.asarray([ob[2] for ob in obs], dtype=float), 0.0)
    h = np.maximum(np.asarray([ob[3] for ob in obs], dtype=float), 0.0)
    kinds = np.asarray([ob[4].lower() for ob in obs], dtype=object)
    r_core = r * CYL_PLAT_FRAC
    r_sh = np.minimum(r - r_core, r * CYL_SHOULDER_FR)
    rmax = np.where(kinds == 'cylinder', np.maximum(0.0, r_core + r_sh), r)
    is_cylinder = kinds == 'cylinder'
    data = (cx, cy, r, h, rmax, is_cylinder)
    if cache_key is not None:
        if len(_OBSTACLE_ARRAY_CACHE) > 32:
            _OBSTACLE_ARRAY_CACHE.clear()
        _OBSTACLE_ARRAY_CACHE[cache_key] = (len(obs), data)
    return data

def _cylinder_top_height_array(x: Array, y: Array, cx: Array, cy: Array, r: Array, h: Array) -> Array:
    rho = np.hypot(np.asarray(x, dtype=float) - cx, np.asarray(y, dtype=float) - cy)
    out = np.zeros_like(rho, dtype=float)
    r_core = np.maximum(0.0, np.minimum(r, r * CYL_PLAT_FRAC))
    r_sh = np.maximum(1e-09, np.minimum(r - r_core, r * CYL_SHOULDER_FR))
    r_inner = r_core * (1.0 - np.clip(CYL_SOFT_TOP, 0.0, 0.2))
    h_full = np.broadcast_to(h, rho.shape)
    r_full = np.broadcast_to(r, rho.shape)
    h_valid = (r_full > 1e-12) & (h_full > 1e-12)
    r_core_full = np.broadcast_to(r_core, rho.shape)
    r_sh_full = np.broadcast_to(r_sh, rho.shape)
    r_inner_full = np.broadcast_to(r_inner, rho.shape)
    mask_inner = h_valid & (rho <= r_inner_full)
    out[mask_inner] = h_full[mask_inner]
    mask_core = h_valid & (rho > r_inner_full) & (rho <= r_core_full)
    if np.any(mask_core):
        t = (rho - r_inner_full) / np.maximum(r_core_full - r_inner_full, 1e-09)
        out[mask_core] = h_full[mask_core] * (1.0 - 0.04 * _smoothstep_array(t[mask_core]))
    mask_shoulder = h_valid & (rho > r_core_full) & (rho < r_core_full + r_sh_full)
    if np.any(mask_shoulder):
        t = (rho - r_core_full) / r_sh_full
        out[mask_shoulder] = h_full[mask_shoulder] * (1.0 - _smoothstep_array(t[mask_shoulder]))
    return out

def _cone_top_height_array(x: Array, y: Array, cx: Array, cy: Array, r: Array, h: Array) -> Array:
    rho = np.hypot(np.asarray(x, dtype=float) - cx, np.asarray(y, dtype=float) - cy)
    out = np.zeros_like(rho, dtype=float)
    h_full = np.broadcast_to(h, rho.shape)
    r_full = np.broadcast_to(r, rho.shape)
    valid = (r_full > 1e-12) & (h_full > 1e-12)
    t = np.clip(rho / np.maximum(r_full, 1e-12), 0.0, 1.0)
    inside = valid & (t < 1.0)
    if np.any(inside):
        rc = 0.5 * (1.0 + np.cos(np.pi * t))
        poly = np.maximum(0.0, 1.0 - t ** DOME_P) ** DOME_Q
        domed = (1.0 - DOME_BLEND) * rc + DOME_BLEND * poly
        out[inside] = h_full[inside] * domed[inside]
    return out

def _obstacle_top_height_matrix(x: Array, y: Array, obstacle_data) -> Array:
    if obstacle_data is None:
        return np.zeros((np.asarray(x).size, 0), dtype=float)
    cx, cy, r, h, _rmax, is_cylinder = obstacle_data
    x_col = np.asarray(x, dtype=float).reshape(-1, 1)
    y_col = np.asarray(y, dtype=float).reshape(-1, 1)
    cx_row = cx.reshape(1, -1)
    cy_row = cy.reshape(1, -1)
    r_row = r.reshape(1, -1)
    h_row = h.reshape(1, -1)
    out = np.zeros((x_col.shape[0], cx.size), dtype=float)
    if np.any(is_cylinder):
        idx = np.where(is_cylinder)[0]
        out[:, idx] = _cylinder_top_height_array(x_col, y_col, cx_row[:, idx], cy_row[:, idx], r_row[:, idx], h_row[:, idx])
    if np.any(~is_cylinder):
        idx = np.where(~is_cylinder)[0]
        out[:, idx] = _cone_top_height_array(x_col, y_col, cx_row[:, idx], cy_row[:, idx], r_row[:, idx], h_row[:, idx])
    return out

def cylinder_top_height_xy(x: float, y: float, cx: float, cy: float, r: float, h: float, *, plat_frac: float=CYL_PLAT_FRAC, shoulder_frac: float=CYL_SHOULDER_FR, soft_top: float=CYL_SOFT_TOP) -> float:
    dx, dy = (float(x) - float(cx), float(y) - float(cy))
    rho = float(np.hypot(dx, dy))
    r = float(max(r, 0.0))
    h = float(max(h, 0.0))
    if r <= 1e-12 or h <= 1e-12:
        return 0.0
    r_core = max(0.0, min(r, r * float(plat_frac)))
    r_sh = max(1e-09, min(r - r_core, r * float(shoulder_frac)))
    r_inner = r_core * (1.0 - np.clip(float(soft_top), 0.0, 0.2))
    if rho <= r_inner:
        return h
    if rho <= r_core:
        t = (rho - r_inner) / max(r_core - r_inner, 1e-09)
        eps = 0.04
        return float(h * (1.0 - eps * smoothstep(t)))
    if rho < r_core + r_sh:
        t = (rho - r_core) / r_sh
        return float(h * (1.0 - smoothstep(t)))
    return 0.0

def cone_dome_top_height_xy(x: float, y: float, cx: float, cy: float, r: float, h: float, *, p: float=DOME_P, q: float=DOME_Q, blend: float=DOME_BLEND) -> float:
    dx, dy = (float(x) - float(cx), float(y) - float(cy))
    rho = float(np.hypot(dx, dy))
    r = float(max(r, 0.0))
    h = float(max(h, 0.0))
    if r <= 1e-12 or h <= 1e-12:
        return 0.0
    t = float(np.clip(rho / r, 0.0, 1.0))
    if t >= 1.0:
        return 0.0
    rc = 0.5 * (1.0 + np.cos(np.pi * t))
    poly = max(0.0, 1.0 - t ** float(p)) ** float(q)
    domed = (1.0 - float(blend)) * rc + float(blend) * poly
    return float(h * domed)

def obstacle_xy_max_radius(r: float, kind: str) -> float:
    r = float(max(r, 0.0))
    if str(kind).lower() == 'cylinder':
        r_core = r * CYL_PLAT_FRAC
        r_sh = min(r - r_core, r * CYL_SHOULDER_FR)
        return float(max(0.0, r_core + r_sh))
    return r

def obstacle_top_height(x: float, y: float, cx: float, cy: float, r: float, h: float, kind: str) -> float:
    kind = str(kind).lower()
    if kind == 'cylinder':
        return cylinder_top_height_xy(x, y, cx, cy, r, h)
    return cone_dome_top_height_xy(x, y, cx, cy, r, h)

def lateral_margin_xy(x: float, y: float, ob: Obstacle) -> float:
    cx, cy, r, _h, kind = ob
    rmax = obstacle_xy_max_radius(r, kind)
    return float(np.hypot(float(x) - float(cx), float(y) - float(cy)) - rmax)

def min_lateral_margin_xy(x: float, y: float, obstacles: Iterable[Obstacle]) -> float:
    return float(min_lateral_margins_xy(np.asarray([x]), np.asarray([y]), obstacles)[0])

def min_lateral_margins_xy(x: Array, y: Array, obstacles: Iterable[Obstacle]) -> Array:
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    if xs.shape != ys.shape:
        raise ValueError('x and y must have the same shape.')
    obstacle_data = _obstacle_arrays(obstacles)
    if obstacle_data is None:
        return np.full(xs.shape, np.inf, dtype=float)
    cx, cy, _r, _h, rmax, _is_cylinder = obstacle_data
    xf = xs.reshape(-1, 1)
    yf = ys.reshape(-1, 1)
    margins = np.hypot(xf - cx.reshape(1, -1), yf - cy.reshape(1, -1)) - rmax.reshape(1, -1)
    return np.min(margins, axis=1).reshape(xs.shape)

def segment_min_lateral_margin(A: Array, B: Array, obstacles: Iterable[Obstacle], *, samples: int=25) -> float:
    return float(segments_min_lateral_margins(A, B, obstacles, samples=samples)[0])

def segments_min_lateral_margins(A: Array, B: Array, obstacles: Iterable[Obstacle], *, samples: int=25) -> Array:
    A_arr = np.asarray(A, dtype=float)
    B_arr = np.asarray(B, dtype=float)
    if A_arr.ndim == 1:
        A_arr = A_arr.reshape(1, -1)
    if B_arr.ndim == 1:
        B_arr = B_arr.reshape(1, -1)
    if A_arr.shape != B_arr.shape or A_arr.shape[1] < 2:
        raise ValueError('A and B must have matching shape (n_segments, >=2).')
    n = max(5, int(samples))
    ts = np.linspace(0.0, 1.0, n, dtype=float)
    xs = A_arr[:, 0:1] + ts.reshape(1, -1) * (B_arr[:, 0:1] - A_arr[:, 0:1])
    ys = A_arr[:, 1:2] + ts.reshape(1, -1) * (B_arr[:, 1:2] - A_arr[:, 1:2])
    margins = min_lateral_margins_xy(xs, ys, obstacles)
    return np.min(margins, axis=1)

def point_in_obstacle(node: Array, ob: Obstacle) -> bool:
    node = np.asarray(node, dtype=float).reshape(-1)
    if node.size < 3 or not np.all(np.isfinite(node[:3])):
        return True
    x, y, z = (float(node[0]), float(node[1]), float(node[2]))
    cx, cy, r, h, kind = ob
    rmax = obstacle_xy_max_radius(r, kind)
    if (x - cx) ** 2 + (y - cy) ** 2 > rmax * rmax + 1e-12:
        return False
    z_top = obstacle_top_height(x, y, cx, cy, r, h, kind)
    return bool(z_top > 0.0 and z <= z_top + 1e-12)

def points_in_any_obstacle(points: Array, obstacles: Iterable[Obstacle]) -> Array:
    P = np.asarray(points, dtype=float)
    if P.ndim == 1:
        P = P.reshape(1, -1)
    if P.shape[1] < 3:
        return np.ones(P.shape[0], dtype=bool)
    finite = np.all(np.isfinite(P[:, :3]), axis=1)
    obstacle_data = _obstacle_arrays(obstacles)
    if obstacle_data is None:
        return np.zeros(P.shape[0], dtype=bool)
    cx, cy, _r, _h, rmax, _is_cylinder = obstacle_data
    x = P[:, 0]
    y = P[:, 1]
    z = P[:, 2]
    dx = x.reshape(-1, 1) - cx.reshape(1, -1)
    dy = y.reshape(-1, 1) - cy.reshape(1, -1)
    inside_xy = dx * dx + dy * dy <= rmax.reshape(1, -1) ** 2 + 1e-12
    z_top = _obstacle_top_height_matrix(x, y, obstacle_data)
    hits = inside_xy & (z_top > 0.0) & (z.reshape(-1, 1) <= z_top + 1e-12)
    return finite & np.any(hits, axis=1)

def overfly_height_gaps(x: Array, y: Array, z: Array, obstacles: Iterable[Obstacle], *, overfly_margin: float) -> Array:
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    zs = np.asarray(z, dtype=float)
    if xs.shape != ys.shape or xs.shape != zs.shape:
        raise ValueError('x, y and z must have the same shape.')
    obstacle_data = _obstacle_arrays(obstacles)
    if obstacle_data is None:
        return np.zeros(0, dtype=float)
    cx, cy, _r, _h, rmax, _is_cylinder = obstacle_data
    xf = xs.reshape(-1, 1)
    yf = ys.reshape(-1, 1)
    zf = zs.reshape(-1, 1)
    dx = xf - cx.reshape(1, -1)
    dy = yf - cy.reshape(1, -1)
    inside_xy = dx * dx + dy * dy <= rmax.reshape(1, -1) ** 2 + 1e-12
    z_top = _obstacle_top_height_matrix(xs.reshape(-1), ys.reshape(-1), obstacle_data)
    mask = inside_xy & (z_top > 0.0)
    if not np.any(mask):
        return np.zeros(0, dtype=float)
    threshold = z_top - float(max(overfly_margin, 0.0))
    return (zf - threshold)[mask].astype(float, copy=False)

def segment_hits_obstacle(A: Array, B: Array, ob: Obstacle, *, samples_far: int=21, samples_near: int=61) -> bool:
    samples = max(int(samples_far), int(samples_near))
    return bool(segments_hit_any_obstacle(A, B, [ob], samples=samples)[0])

def segments_hit_any_obstacle(A: Array, B: Array, obstacles: Iterable[Obstacle], *, samples: int=61) -> Array:
    obs = obstacles if isinstance(obstacles, (list, tuple)) else list(obstacles)
    A_arr = np.asarray(A, dtype=float)
    B_arr = np.asarray(B, dtype=float)
    if A_arr.ndim == 1:
        A_arr = A_arr.reshape(1, -1)
    if B_arr.ndim == 1:
        B_arr = B_arr.reshape(1, -1)
    if A_arr.shape != B_arr.shape or A_arr.shape[1] < 3:
        return np.ones(A_arr.shape[0], dtype=bool)
    n_segments = A_arr.shape[0]
    if n_segments <= 0:
        return np.zeros(0, dtype=bool)
    finite_segments = np.all(np.isfinite(A_arr[:, :3]), axis=1) & np.all(np.isfinite(B_arr[:, :3]), axis=1)
    if _obstacle_arrays(obs) is None:
        return ~finite_segments
    ts = np.linspace(0.0, 1.0, max(5, int(samples)), dtype=float)
    Q = A_arr[:, None, :3] + ts.reshape(1, -1, 1) * (B_arr[:, None, :3] - A_arr[:, None, :3])
    hits = points_in_any_obstacle(Q.reshape(-1, 3), obs).reshape(n_segments, -1)
    return ~finite_segments | np.any(hits, axis=1)
__all__ = ['Array', 'Obstacle', 'CYL_PLAT_FRAC', 'CYL_SHOULDER_FR', 'CYL_SOFT_TOP', 'DOME_BLEND', 'DOME_P', 'DOME_Q', 'smoothstep', 'cylinder_top_height_xy', 'cone_dome_top_height_xy', 'obstacle_xy_max_radius', 'obstacle_top_height', 'lateral_margin_xy', 'min_lateral_margin_xy', 'min_lateral_margins_xy', 'segment_min_lateral_margin', 'segments_min_lateral_margins', 'point_in_obstacle', 'points_in_any_obstacle', 'overfly_height_gaps', 'segment_hits_obstacle', 'segments_hit_any_obstacle']
try:
    from tools.adaptive_waypoint_gate import AdaptiveWaypointConfig, adaptive_waypoint_penalty, build_path_from_xg, summarize_gate_info
except Exception:
    from algorithm.tawg_pso import AdaptiveWaypointConfig, adaptive_waypoint_penalty, build_path_from_xg, summarize_gate_info
Array = np.ndarray
HeightFn = Callable[[float, float], float]
MAX_PENALTY: float = 1000000.0
_EPS = 1e-12
_COLLISION_FACTOR = 20.0
_HEIGHT_SCALE_MIN = 10.0
_DEFAULT_SEG_SAMPLES = 61
_DEFAULT_AGL_SAMPLES = 21
_DEFAULT_OVERFLY_SAMPLES = 7

@dataclass(frozen=True)
class CostWeights:
    F1_length: float = 1.0
    F2_obstacle: float = 10.0
    F3_height: float = 6.0
    F4_smooth: float = 0.8
    F5_waypoint: float = 0.6
DEFAULT_WEIGHTS = CostWeights()

def _as_point3(x: Array, name: str='point') -> Array:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size != 3:
        raise ValueError(f'{name} must have shape (3,), got shape={np.shape(x)}')
    if not np.all(np.isfinite(arr)):
        raise ValueError(f'{name} contains non-finite values.')
    return arr

def _as_path(P: Array) -> Array:
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError('path must have shape (n_points, 3).')
    if P.shape[0] < 2:
        raise ValueError('path must contain at least start and goal.')
    if not np.all(np.isfinite(P)):
        raise ValueError('path contains non-finite values.')
    return P

def _segment_lengths(P: Array) -> Array:
    P = np.asarray(P, dtype=float)
    if P.shape[0] < 2:
        return np.zeros(0, dtype=float)
    return np.linalg.norm(P[1:] - P[:-1], axis=1)

def _reference_length(start: Array, goal: Array) -> float:
    s = _as_point3(start, 'start')
    g = _as_point3(goal, 'goal')
    return float(max(np.linalg.norm(g - s), 1.0))

def _soft_hinge(gap: Union[float, Array], scale: float, power: float=2.0) -> Union[float, Array]:
    s = max(float(scale), 1e-09)
    g = np.asarray(gap, dtype=float)
    out = np.maximum(g, 0.0) / s
    out = out ** float(power)
    if np.isscalar(gap):
        return float(out)
    return out

def _eval_height_fn(height_fn: HeightFn, xs: Array, ys: Array) -> Array:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    try:
        z = height_fn(xs, ys)
        return np.asarray(z, dtype=float)
    except Exception:
        out = np.empty_like(xs, dtype=float)
        for i, (x, y) in enumerate(zip(xs, ys)):
            out[i] = float(height_fn(float(x), float(y)))
        return out

def _sample_segment(A: Array, B: Array, samples: int) -> Array:
    A = np.asarray(A, dtype=float).reshape(3)
    B = np.asarray(B, dtype=float).reshape(3)
    ts = np.linspace(0.0, 1.0, max(2, int(samples)))
    return A[None, :] + ts[:, None] * (B[None, :] - A[None, :])

def _as_obstacles(obstacles: Iterable[Obstacle]) -> Iterable[Obstacle]:
    return obstacles if isinstance(obstacles, (list, tuple)) else list(obstacles)

def _count_obstacle_hits(P: Array, obstacles: Iterable[Obstacle]) -> int:
    P = _as_path(P)
    obs = _as_obstacles(obstacles)
    if not obs:
        return 0
    node_hits = points_in_any_obstacle(P, obs)
    seg_hits = segments_hit_any_obstacle(P[:-1], P[1:], obs, samples=_DEFAULT_SEG_SAMPLES)
    return int(np.count_nonzero(node_hits) + np.count_nonzero(seg_hits))

def f1_path_length(P: Array, start: Array, goal: Array) -> float:
    P = _as_path(P)
    L = float(np.sum(_segment_lengths(P)))
    L_ref = _reference_length(start, goal)
    return float(L / L_ref)

def f2_obstacle_safety(P: Array, obstacles: Iterable[Obstacle], *, safety_margin: float=4.0) -> float:
    P = _as_path(P)
    obs = _as_obstacles(obstacles)
    if not obs:
        return 0.0
    margin = max(float(safety_margin), 1.0)
    n_seg = max(P.shape[0] - 1, 1)
    node_margins = min_lateral_margins_xy(P[:, 0], P[:, 1], obs)
    node_gaps = margin - np.asarray(node_margins, dtype=float)
    node_pens = _soft_hinge(node_gaps, scale=margin)
    seg_margins = segments_min_lateral_margins(P[:-1], P[1:], obs, samples=_DEFAULT_SEG_SAMPLES)
    seg_gaps = margin - np.asarray(seg_margins, dtype=float)
    seg_pens = _soft_hinge(seg_gaps, scale=margin)
    node_pen = float(np.mean(node_pens)) if node_pens.size else 0.0
    seg_pen = float(np.mean(seg_pens)) if seg_pens.size else 0.0
    max_node_gap = float(np.max(node_gaps)) if node_gaps.size else 0.0
    max_seg_gap = float(np.max(seg_gaps)) if seg_gaps.size else 0.0
    max_gap = max(0.0, max_node_gap, max_seg_gap)
    max_pen = float(_soft_hinge(max_gap, scale=margin))
    violation_ratio = float(max_gap / margin) if margin > 0.0 else 0.0
    clearance_barrier = 0.0
    if violation_ratio > 0.0:
        clearance_barrier = float(3.0 * violation_ratio + 10.0 * violation_ratio * violation_ratio)
    hits = 0
    node_hit_candidates = np.asarray(node_margins <= 0.0, dtype=bool)
    if np.any(node_hit_candidates):
        hits += int(np.count_nonzero(points_in_any_obstacle(P[node_hit_candidates], obs)))
    seg_hit_candidates = np.asarray(seg_margins <= 0.0, dtype=bool)
    if np.any(seg_hit_candidates):
        hits += int(np.count_nonzero(segments_hit_any_obstacle(P[:-1][seg_hit_candidates], P[1:][seg_hit_candidates], obs, samples=_DEFAULT_SEG_SAMPLES)))
    collision_rate = float(hits) / float(max(P.shape[0] + n_seg, 1))
    F2 = node_pen + seg_pen + 2.0 * max_pen + clearance_barrier + _COLLISION_FACTOR * collision_rate
    if not np.isfinite(F2):
        return float(MAX_PENALTY)
    return float(F2)

def f3_height_agl_constraints(P: Array, obstacles: Iterable[Obstacle], *, height_fn: Optional[HeightFn], hmin: float, hmax: float, agl_min: float=5.0, agl_max: Optional[float]=120.0, forbid_overfly: bool=True, overfly_margin: float=5.0) -> float:
    P = _as_path(P)
    obs = _as_obstacles(obstacles)
    hmin = float(hmin)
    hmax = float(hmax)
    if hmax <= hmin:
        raise ValueError('hmax must be larger than hmin.')
    height_scale = max(0.1 * (hmax - hmin), _HEIGHT_SCALE_MIN)
    agl_min = float(max(agl_min, 0.0))
    overfly_margin = float(max(overfly_margin, 0.0))
    seg_a = P[:-1]
    seg_b = P[1:]
    n_seg = int(seg_a.shape[0])
    ts = np.linspace(0.0, 1.0, max(2, int(_DEFAULT_AGL_SAMPLES)), dtype=float)
    Q = seg_a[:, None, :] + ts.reshape(1, -1, 1) * (seg_b[:, None, :] - seg_a[:, None, :])
    Q = Q.reshape(-1, 3)
    xs, ys, zs = (Q[:, 0], Q[:, 1], Q[:, 2])
    total_count = int(Q.shape[0])
    total_pen = 0.0
    total_pen += float(np.sum(_soft_hinge(hmin - zs, scale=height_scale)))
    total_pen += float(np.sum(_soft_hinge(zs - hmax, scale=height_scale)))
    if height_fn is not None:
        zsurf = _eval_height_fn(height_fn, xs, ys)
        agl = zs - zsurf
        total_pen += float(np.sum(_soft_hinge(agl_min - agl, scale=max(agl_min, 1.0))))
        if agl_max is not None:
            agl_max_f = float(agl_max)
            total_pen += float(np.sum(_soft_hinge(agl - agl_max_f, scale=max(agl_max_f, 1.0))))
    if forbid_overfly and overfly_margin > 0.0 and obs and (n_seg > 0):
        ts_over = np.linspace(0.0, 1.0, max(2, int(_DEFAULT_OVERFLY_SAMPLES)), dtype=float)
        Q_over = seg_a[:, None, :] + ts_over.reshape(1, -1, 1) * (seg_b[:, None, :] - seg_a[:, None, :])
        Q_over = Q_over.reshape(-1, 3)
        gaps = overfly_height_gaps(Q_over[:, 0], Q_over[:, 1], Q_over[:, 2], obs, overfly_margin=overfly_margin)
        if gaps.size > 0:
            total_pen += float(np.sum(_soft_hinge(gaps, scale=max(overfly_margin, 1.0))))
            total_count += int(gaps.size)
    mean_pen = float(total_pen / float(max(total_count, 1)))
    barrier_pen = 0.0
    if height_fn is not None:
        max_low_agl_gap = float(np.max(np.maximum(agl_min - agl, 0.0))) if agl.size else 0.0
        if max_low_agl_gap > 0.0:
            gap_ratio = max_low_agl_gap / max(agl_min, 1.0)
            barrier_pen += 8.0 * gap_ratio + 20.0 * gap_ratio * gap_ratio
        if agl_max is not None:
            agl_max_f = float(agl_max)
            max_high_agl_gap = float(np.max(np.maximum(agl - agl_max_f, 0.0))) if agl.size else 0.0
            if max_high_agl_gap > 0.0:
                gap_ratio = max_high_agl_gap / max(agl_max_f, 1.0)
                barrier_pen += 4.0 * gap_ratio + 8.0 * gap_ratio * gap_ratio
    return float(mean_pen + barrier_pen)

def f4_smoothness(P: Array) -> float:
    P = _as_path(P)
    if P.shape[0] < 3:
        return 0.0
    V = P[1:] - P[:-1]
    dx, dy, dz = (V[:, 0], V[:, 1], V[:, 2])
    horiz = np.maximum(np.hypot(dx, dy), _EPS)
    yaw = np.unwrap(np.arctan2(dy, dx))
    pitch = np.arctan2(dz, horiz)
    if yaw.size < 2:
        return 0.0
    dyaw = np.abs(np.diff(yaw)) / np.pi
    dpitch = np.abs(np.diff(pitch)) / (0.5 * np.pi)
    return float(np.mean(dyaw + dpitch))

def f5_waypoint_complexity(gate_info: Dict[str, Any], gate_cfg: AdaptiveWaypointConfig, *, iteration: Optional[int]=None, max_iter: Optional[int]=None) -> float:
    raw_penalty, _detail = adaptive_waypoint_penalty(gate_info=gate_info, cfg=gate_cfg, iteration=iteration, max_iter=max_iter)
    q = np.asarray(gate_info.get('q', []), dtype=float)
    k_cap = max(int(q.size), 1)
    return float(raw_penalty / float(k_cap))

def waypoint_structure_regularizer(gate_info: Dict[str, Any], gate_cfg: AdaptiveWaypointConfig, *, iteration: Optional[int]=None, max_iter: Optional[int]=None) -> float:
    return f5_waypoint_complexity(gate_info, gate_cfg, iteration=iteration, max_iter=max_iter)

def path_cost_from_path(P: Array, *, start: Array, goal: Array, obstacles: Iterable[Obstacle], height_fn: Optional[HeightFn], hmin: float, hmax: float, agl_min: float=5.0, agl_max: Optional[float]=120.0, safety_margin: float=4.0, forbid_overfly: bool=True, overfly_margin: float=5.0, weights: CostWeights=DEFAULT_WEIGHTS, return_detail: bool=False) -> Union[float, Tuple[float, Dict[str, float]]]:
    try:
        P = _as_path(P)
        start = _as_point3(start, 'start')
        goal = _as_point3(goal, 'goal')
        L_ref = _reference_length(start, goal)
        F1 = f1_path_length(P, start, goal)
        F2 = f2_obstacle_safety(P, obstacles, safety_margin=safety_margin)
        F3 = f3_height_agl_constraints(P, obstacles, height_fn=height_fn, hmin=hmin, hmax=hmax, agl_min=agl_min, agl_max=agl_max, forbid_overfly=forbid_overfly, overfly_margin=overfly_margin)
        F4 = f4_smoothness(P)
        F_norm = weights.F1_length * F1 + weights.F2_obstacle * F2 + weights.F3_height * F3 + weights.F4_smooth * F4
        fit = float(L_ref * F_norm)
        if not np.isfinite(fit):
            fit = MAX_PENALTY
        if not return_detail:
            return fit
        detail = {'F1_length': float(F1), 'F2_obstacle': float(F2), 'F3_height': float(F3), 'F4_smooth': float(F4), 'F5_waypoint': 0.0, 'waypoint_regularizer': 0.0, 'F_norm': float(F_norm), 'L_ref': float(L_ref), 'path_length': float(np.sum(_segment_lengths(P))), 'fitness': float(fit)}
        return (fit, detail)
    except Exception:
        if not return_detail:
            return float(MAX_PENALTY)
        return (float(MAX_PENALTY), {'fitness': float(MAX_PENALTY)})

def total_cost_sipso_adaptive(xg: Array, *, start: Array, goal: Array, obstacles: Iterable[Obstacle], height_fn: Optional[HeightFn], gate_cfg: AdaptiveWaypointConfig, hmin: float, hmax: float, agl_min: float=5.0, agl_max: Optional[float]=120.0, safety_margin: float=4.0, forbid_overfly: bool=True, overfly_margin: float=5.0, weights: CostWeights=DEFAULT_WEIGHTS, iteration: Optional[int]=None, max_iter: Optional[int]=None, return_detail: bool=False) -> Union[float, Tuple[float, Dict[str, Any]]]:
    try:
        start = _as_point3(start, 'start')
        goal = _as_point3(goal, 'goal')
        xg = np.asarray(xg, dtype=float).reshape(-1)
        expected = 4 * int(gate_cfg.k_cap)
        if xg.size != expected:
            raise ValueError(f'xg length mismatch: expected {expected}, got {xg.size}.')
        if not np.all(np.isfinite(xg)):
            raise ValueError('xg contains non-finite values.')
        P, gate_info = build_path_from_xg(xg, start, goal, gate_cfg)
        P = _as_path(P)
        L_ref = _reference_length(start, goal)
        F1 = f1_path_length(P, start, goal)
        F2 = f2_obstacle_safety(P, obstacles, safety_margin=safety_margin)
        F3 = f3_height_agl_constraints(P, obstacles, height_fn=height_fn, hmin=hmin, hmax=hmax, agl_min=agl_min, agl_max=agl_max, forbid_overfly=forbid_overfly, overfly_margin=overfly_margin)
        F4 = f4_smoothness(P)
        F5 = f5_waypoint_complexity(gate_info, gate_cfg, iteration=iteration, max_iter=max_iter)
        F_norm = weights.F1_length * F1 + weights.F2_obstacle * F2 + weights.F3_height * F3 + weights.F4_smooth * F4 + weights.F5_waypoint * F5
        fit = float(L_ref * F_norm)
        if not np.isfinite(fit):
            fit = MAX_PENALTY
        if not return_detail:
            return fit
        gate_summary = summarize_gate_info(gate_info)
        detail: Dict[str, Any] = {'F1_length': float(F1), 'F2_obstacle': float(F2), 'F3_height': float(F3), 'F4_smooth': float(F4), 'F5_waypoint': float(F5), 'waypoint_regularizer': float(F5), 'F_norm': float(F_norm), 'L_ref': float(L_ref), 'path_length': float(np.sum(_segment_lengths(P))), 'fitness': float(fit), 'k_eff': int(gate_summary.get('k_eff', 0)), 'k_soft': float(gate_summary.get('k_soft', 0.0)), 'gate_binary': float(gate_summary.get('binary_term', 0.0))}
        return (fit, detail)
    except Exception as exc:
        if not return_detail:
            return float(MAX_PENALTY)
        return (float(MAX_PENALTY), {'fitness': float(MAX_PENALTY), 'error': str(exc)})

def build_cost_sipso_adaptive(*, start: Array, goal: Array, obstacles: Iterable[Obstacle], height_fn: Optional[HeightFn], gate_cfg: AdaptiveWaypointConfig, hmin: float, hmax: float, agl_min: float=5.0, agl_max: Optional[float]=120.0, safety_margin: float=4.0, forbid_overfly: bool=True, overfly_margin: float=5.0, weights: CostWeights=DEFAULT_WEIGHTS, max_iter: Optional[int]=None) -> Callable[[Array], float]:
    state: Dict[str, Optional[int]] = {'iteration': None, 'max_iter': max_iter}
    start_arr = _as_point3(start, 'start')
    goal_arr = _as_point3(goal, 'goal')
    obs_list = list(obstacles)

    def _cost(xg: Array) -> float:
        return float(total_cost_sipso_adaptive(xg, start=start_arr, goal=goal_arr, obstacles=obs_list, height_fn=height_fn, gate_cfg=gate_cfg, hmin=hmin, hmax=hmax, agl_min=agl_min, agl_max=agl_max, safety_margin=safety_margin, forbid_overfly=forbid_overfly, overfly_margin=overfly_margin, weights=weights, iteration=state['iteration'], max_iter=state['max_iter'], return_detail=False))

    def set_iteration(iteration: Optional[int], max_iter_override: Optional[int]=None) -> None:
        state['iteration'] = None if iteration is None else int(iteration)
        if max_iter_override is not None:
            state['max_iter'] = int(max_iter_override)

    def detail(xg: Array) -> Dict[str, Any]:
        _fit, info = total_cost_sipso_adaptive(xg, start=start_arr, goal=goal_arr, obstacles=obs_list, height_fn=height_fn, gate_cfg=gate_cfg, hmin=hmin, hmax=hmax, agl_min=agl_min, agl_max=agl_max, safety_margin=safety_margin, forbid_overfly=forbid_overfly, overfly_margin=overfly_margin, weights=weights, iteration=state['iteration'], max_iter=state['max_iter'], return_detail=True)
        return dict(info)

    def decode_path(xg: Array) -> Tuple[Array, Dict[str, Any]]:
        return build_path_from_xg(xg, start_arr, goal_arr, gate_cfg)

    def evaluate_path(P: Array, return_detail: bool=False):
        return path_cost_from_path(P, start=start_arr, goal=goal_arr, obstacles=obs_list, height_fn=height_fn, hmin=hmin, hmax=hmax, agl_min=agl_min, agl_max=agl_max, safety_margin=safety_margin, forbid_overfly=forbid_overfly, overfly_margin=overfly_margin, weights=weights, return_detail=return_detail)
    _cost.set_iteration = set_iteration
    _cost.detail = detail
    _cost.decode_path = decode_path
    _cost.evaluate_path = evaluate_path
    _cost.gate_cfg = gate_cfg
    return _cost

def _sort_waypoints_along_mission(waypoints: Array, start: Array, goal: Array) -> Array:
    W = np.asarray(waypoints, dtype=float)
    if W.size == 0:
        return W.reshape(0, 3)
    W = W.reshape(-1, 3)
    s = _as_point3(start, 'start')
    g = _as_point3(goal, 'goal')
    axis = g - s
    denom = float(np.dot(axis, axis))
    if denom <= _EPS:
        return W.copy()
    t = (W - s[None, :]) @ axis / denom
    order = np.argsort(t, kind='mergesort')
    return W[order].copy()

def build_fixed_path_from_xyz(xyz: Array, start: Array, goal: Array, *, sort_waypoints: bool=True) -> Tuple[Array, Dict[str, Any]]:
    start_arr = _as_point3(start, 'start')
    goal_arr = _as_point3(goal, 'goal')
    xyz_arr = np.asarray(xyz, dtype=float).reshape(-1)
    if xyz_arr.size % 3 != 0:
        raise ValueError(f'fixed xyz length must be divisible by 3, got {xyz_arr.size}.')
    if not np.all(np.isfinite(xyz_arr)):
        raise ValueError('fixed xyz vector contains non-finite values.')
    waypoints = xyz_arr.reshape(-1, 3)
    if sort_waypoints:
        waypoints = _sort_waypoints_along_mission(waypoints, start_arr, goal_arr)
    path = np.vstack([start_arr.reshape(1, 3), waypoints, goal_arr.reshape(1, 3)])
    info = {'encoding': 'fixed_xyz', 'k_eff': int(waypoints.shape[0]), 'K_eff': int(waypoints.shape[0]), 'sort_waypoints': bool(sort_waypoints)}
    return (path, info)

def total_cost_fixed_waypoints(xyz: Array, *, start: Array, goal: Array, obstacles: Iterable[Obstacle], height_fn: Optional[HeightFn], hmin: float, hmax: float, agl_min: float=5.0, agl_max: Optional[float]=120.0, safety_margin: float=4.0, forbid_overfly: bool=True, overfly_margin: float=5.0, weights: CostWeights=DEFAULT_WEIGHTS, sort_waypoints: bool=True, return_detail: bool=False) -> Union[float, Tuple[float, Dict[str, Any]]]:
    try:
        P, decode_info = build_fixed_path_from_xyz(xyz, start=start, goal=goal, sort_waypoints=sort_waypoints)
        out = path_cost_from_path(P, start=start, goal=goal, obstacles=obstacles, height_fn=height_fn, hmin=hmin, hmax=hmax, agl_min=agl_min, agl_max=agl_max, safety_margin=safety_margin, forbid_overfly=forbid_overfly, overfly_margin=overfly_margin, weights=weights, return_detail=return_detail)
        if not return_detail:
            return float(out)
        fit, detail = out
        info = dict(detail)
        info.update(decode_info)
        return (float(fit), info)
    except Exception as exc:
        if not return_detail:
            return float(MAX_PENALTY)
        return (float(MAX_PENALTY), {'fitness': float(MAX_PENALTY), 'encoding': 'fixed_xyz', 'error': str(exc)})

def build_cost_fixed_waypoints(*, start: Array, goal: Array, obstacles: Iterable[Obstacle], height_fn: Optional[HeightFn], hmin: float, hmax: float, agl_min: float=5.0, agl_max: Optional[float]=120.0, safety_margin: float=4.0, forbid_overfly: bool=True, overfly_margin: float=5.0, weights: CostWeights=DEFAULT_WEIGHTS, sort_waypoints: bool=True) -> Callable[[Array], float]:
    start_arr = _as_point3(start, 'start')
    goal_arr = _as_point3(goal, 'goal')
    obs_list = list(obstacles)

    def _cost(xyz: Array) -> float:
        return float(total_cost_fixed_waypoints(xyz, start=start_arr, goal=goal_arr, obstacles=obs_list, height_fn=height_fn, hmin=hmin, hmax=hmax, agl_min=agl_min, agl_max=agl_max, safety_margin=safety_margin, forbid_overfly=forbid_overfly, overfly_margin=overfly_margin, weights=weights, sort_waypoints=sort_waypoints, return_detail=False))

    def set_iteration(_iteration: Optional[int], _max_iter_override: Optional[int]=None) -> None:
        return None

    def detail(xyz: Array) -> Dict[str, Any]:
        _fit, info = total_cost_fixed_waypoints(xyz, start=start_arr, goal=goal_arr, obstacles=obs_list, height_fn=height_fn, hmin=hmin, hmax=hmax, agl_min=agl_min, agl_max=agl_max, safety_margin=safety_margin, forbid_overfly=forbid_overfly, overfly_margin=overfly_margin, weights=weights, sort_waypoints=sort_waypoints, return_detail=True)
        return dict(info)

    def decode_path(xyz: Array) -> Tuple[Array, Dict[str, Any]]:
        return build_fixed_path_from_xyz(xyz, start=start_arr, goal=goal_arr, sort_waypoints=sort_waypoints)

    def evaluate_path(P: Array, return_detail: bool=False):
        return path_cost_from_path(P, start=start_arr, goal=goal_arr, obstacles=obs_list, height_fn=height_fn, hmin=hmin, hmax=hmax, agl_min=agl_min, agl_max=agl_max, safety_margin=safety_margin, forbid_overfly=forbid_overfly, overfly_margin=overfly_margin, weights=weights, return_detail=return_detail)
    _cost.set_iteration = set_iteration
    _cost.detail = detail
    _cost.decode_path = decode_path
    _cost.evaluate_path = evaluate_path
    _cost.encoding = 'fixed_xyz'
    return _cost

def path_is_feasible(P: Array, *, obstacles: Iterable[Obstacle], height_fn: Optional[HeightFn], hmin: float, hmax: float, agl_min: float=5.0, agl_max: Optional[float]=120.0, safety_margin: float=4.0, forbid_overfly: bool=True, overfly_margin: float=5.0, tol: float=0.0001) -> bool:
    try:
        P = _as_path(P)
        obs = list(obstacles)
        if _count_obstacle_hits(P, obs) > 0:
            return False
        F2 = f2_obstacle_safety(P, obs, safety_margin=safety_margin)
        F3 = f3_height_agl_constraints(P, obs, height_fn=height_fn, hmin=hmin, hmax=hmax, agl_min=agl_min, agl_max=agl_max, forbid_overfly=forbid_overfly, overfly_margin=overfly_margin)
        return bool(F2 <= tol and F3 <= tol)
    except Exception:
        return False

def _height_values(height_fn: Optional[HeightFn], xs: Array, ys: Array) -> Array:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if height_fn is None:
        return np.zeros_like(xs, dtype=float)
    try:
        return np.asarray(height_fn(xs, ys), dtype=float)
    except Exception:
        out = np.empty_like(xs, dtype=float)
        for index, (x, y) in enumerate(zip(xs.reshape(-1), ys.reshape(-1))):
            out.reshape(-1)[index] = float(height_fn(float(x), float(y)))
        return out

def _sample_path_segments(P: Array, samples: int) -> Array:
    path = _as_path(P)
    ts = np.linspace(0.0, 1.0, max(2, int(samples)), dtype=float)
    return np.vstack([a[None, :] + ts[:, None] * (b[None, :] - a[None, :]) for a, b in zip(path[:-1], path[1:])])

def compute_path_diagnostics(P: Array, ctx: Dict[str, Any]) -> Dict[str, Any]:
    path = _as_path(P)
    obstacles = list(ctx.get('obstacles', []))
    cfg = dict(ctx.get('cfg', {}))
    if obstacles:
        node_margins = min_lateral_margins_xy(path[:, 0], path[:, 1], obstacles)
        segment_margins = segments_min_lateral_margins(path[:-1], path[1:], obstacles, samples=_DEFAULT_SEG_SAMPLES)
        min_clearance = float(np.min(np.r_[node_margins.reshape(-1), segment_margins.reshape(-1)]))
        node_hits = points_in_any_obstacle(path, obstacles)
        segment_hits = segments_hit_any_obstacle(path[:-1], path[1:], obstacles, samples=_DEFAULT_SEG_SAMPLES)
        collision_count = int(np.count_nonzero(node_hits) + np.count_nonzero(segment_hits))
        collision_ratio = float(collision_count / max(path.shape[0] + path.shape[0] - 1, 1))
    else:
        min_clearance = float('inf')
        collision_count = 0
        collision_ratio = 0.0
    samples = _sample_path_segments(path, _DEFAULT_AGL_SAMPLES)
    z = samples[:, 2]
    terrain = _height_values(ctx.get('height_fn'), samples[:, 0], samples[:, 1])
    agl = z - terrain
    hmin = float(cfg.get('HMIN', ctx.get('HMIN', 0.0)))
    hmax = float(cfg.get('HMAX', ctx.get('HMAX', 200.0)))
    agl_min = float(cfg.get('AGL_MIN', ctx.get('AGL_MIN', 5.0)))
    agl_max_raw = cfg.get('AGL_MAX', ctx.get('AGL_MAX', 120.0))
    agl_max = None if agl_max_raw is None else float(agl_max_raw)
    count = max(int(samples.shape[0]), 1)
    return {
        'min_clearance': min_clearance,
        'collision_count': collision_count,
        'collision_ratio': collision_ratio,
        'height_low_violation_ratio': float(np.count_nonzero(z < hmin) / count),
        'height_high_violation_ratio': float(np.count_nonzero(z > hmax) / count),
        'agl_low_violation_ratio': float(np.count_nonzero(agl < agl_min) / count),
        'agl_high_violation_ratio': 0.0 if agl_max is None else float(np.count_nonzero(agl > agl_max) / count),
        'min_agl': float(np.min(agl)),
        'max_agl': float(np.max(agl)),
        'sample_count': count,
    }


class EvaluationBudget:
    """Transparent objective wrapper enforcing the frozen evaluation cap."""

    def __init__(self, cost: Callable[[Array], float], max_evals: int) -> None:
        self._cost = cost
        self.max_evals = max(1, int(max_evals))
        self.n_evals = 0
        self.exhausted_calls = 0
        self.best_x: Array | None = None
        self.best_f = float('inf')

    @property
    def remaining(self) -> int:
        return max(0, self.max_evals - self.n_evals)

    def _penalty_detail(self) -> dict[str, Any]:
        return {'fitness': float(MAX_PENALTY), 'F1_length': float(MAX_PENALTY), 'F2_obstacle': float(MAX_PENALTY), 'F3_height': float(MAX_PENALTY), 'F4_smooth': 0.0, 'F5_waypoint': 0.0}

    def _record(self, x: Array, value: Any) -> float:
        try:
            result = float(value)
        except Exception:
            result = float(MAX_PENALTY)
        if not np.isfinite(result):
            result = float(MAX_PENALTY)
        if result < self.best_f:
            self.best_f = result
            self.best_x = np.asarray(x, dtype=float).copy()
        return result

    def __call__(self, x: Array) -> float:
        x_arr = np.asarray(x, dtype=float).copy()
        if self.n_evals >= self.max_evals:
            self.exhausted_calls += 1
            return float(MAX_PENALTY)
        self.n_evals += 1
        try:
            value = self._cost(x_arr)
        except Exception:
            value = MAX_PENALTY
        return self._record(x_arr, value)

    def detail(self, x: Array) -> dict[str, Any]:
        x_arr = np.asarray(x, dtype=float).copy()
        if self.n_evals >= self.max_evals:
            self.exhausted_calls += 1
            return self._penalty_detail()
        self.n_evals += 1
        detail_fn = getattr(self._cost, 'detail', None)
        if callable(detail_fn):
            try:
                info = dict(detail_fn(x_arr))
            except Exception:
                info = self._penalty_detail()
        else:
            info = {'fitness': self._cost(x_arr)}
        info['fitness'] = self._record(x_arr, info.get('fitness', MAX_PENALTY))
        return info

    def decode_path(self, x: Array):
        decoder = getattr(self._cost, 'decode_path', None)
        if not callable(decoder):
            raise RuntimeError('budgeted objective has no decode_path delegate')
        return decoder(np.asarray(x, dtype=float))

    def evaluate_path(self, path: Array, return_detail: bool=False):
        evaluator = getattr(self._cost, 'evaluate_path', None)
        if callable(evaluator):
            return evaluator(np.asarray(path, dtype=float), return_detail=return_detail)
        return float(MAX_PENALTY)

    def set_iteration(self, iteration: int | None, max_iter_override: int | None=None) -> None:
        setter = getattr(self._cost, 'set_iteration', None)
        if not callable(setter):
            return
        try:
            setter(iteration, max_iter_override)
        except TypeError:
            setter(iteration)

    def metadata(self, attempted_evals: Any=None) -> dict[str, Any]:
        try:
            attempted = int(float(attempted_evals))
        except Exception:
            attempted = self.n_evals + self.exhausted_calls
        data: dict[str, Any] = {'evals_used': int(self.n_evals), 'eval_budget': int(self.max_evals), 'budget_remaining': int(self.remaining), 'budget_exhausted': bool(self.n_evals >= self.max_evals), 'budget_exhausted_calls': int(self.exhausted_calls), 'budget_strict_ok': bool(self.n_evals <= self.max_evals)}
        if attempted > self.n_evals:
            data['evals_attempted'] = attempted
        if np.isfinite(self.best_f):
            data['budget_best_cost'] = float(self.best_f)
        return data

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cost, name)


__all__ = ['Array', 'HeightFn', 'Obstacle', 'MAX_PENALTY', 'CostWeights', 'DEFAULT_WEIGHTS', 'EvaluationBudget', 'f1_path_length', 'f2_obstacle_safety', 'f3_height_agl_constraints', 'f4_smoothness', 'f5_waypoint_complexity', 'waypoint_structure_regularizer', 'path_cost_from_path', 'total_cost_sipso_adaptive', 'build_cost_sipso_adaptive', 'build_fixed_path_from_xyz', 'total_cost_fixed_waypoints', 'build_cost_fixed_waypoints', 'path_is_feasible', 'compute_path_diagnostics']
