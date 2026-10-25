"""Deterministic synthetic terrain generator shared by sparse, medium, and dense obstacle layouts."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource, LinearSegmentedColormap

from environments.synthetic.obstacles import get_obstacle_base

GRID_LIMIT = 200.0
GRID_STEP = 0.35
SUPERSAMPLE = 2
START_XY, START_Z = ((0.0, 0.0), 0.0)
GOAL_XY, GOAL_Z = ((200.0, 200.0), 100.0)
CYLINDER_H = 160.0
H_TOL = 1.0
Z_ASPECT_SCALE = 0.4
VIEW_ELEV, VIEW_AZIM = (20, -35)
LIGHT_AZ, LIGHT_ALT = (315, 34)
GAMMA = 1.05

def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {'0', 'false', 'no', 'off'}

def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)

def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return int(default)
    try:
        return int(value)
    except ValueError:
        return int(default)
TERRAIN_ENABLE = _env_bool('DEM_TERRAIN_ENABLE', True)
TERRAIN_SEED = _env_int('DEM_TERRAIN_SEED', 2026)
TERRAIN_PEAK_COUNT = _env_int('DEM_TERRAIN_PEAK_COUNT', 50)
TERRAIN_PEAK_HEIGHT = _env_float('DEM_TERRAIN_PEAK_HEIGHT', 160)
TERRAIN_VALLEY_DEPTH = _env_float('DEM_TERRAIN_VALLEY_DEPTH', 0.0)
TERRAIN_ALTITUDE_SCALE = _env_float('DEM_TERRAIN_ALTITUDE_SCALE', 14.0)
TERRAIN_RADIUS_MIN = _env_float('DEM_TERRAIN_RADIUS_MIN', 14.0)
TERRAIN_RADIUS_MAX = _env_float('DEM_TERRAIN_RADIUS_MAX', 30.0)
TERRAIN_EDGE_MARGIN = _env_float('DEM_TERRAIN_EDGE_MARGIN', 14.0)
TERRAIN_NOISE_SCALE = _env_float('DEM_TERRAIN_NOISE_SCALE', 0.08)
TERRAIN_EDGE_TAPER = _env_bool('DEM_TERRAIN_EDGE_TAPER', True)
TERRAIN_EDGE_TAPER_WIDTH = _env_float('DEM_TERRAIN_EDGE_TAPER_WIDTH', 18.0)
TERRAIN_EDGE_TAPER_MIN = _env_float('DEM_TERRAIN_EDGE_TAPER_MIN', 0.18)
CYL_PLAT_FRAC = 0.74
CYL_SHOULDER_FR = 0.3
CYL_SOFT_TOP = 0.06
DOME_BLEND = 0.5
DOME_P = 1.75
DOME_Q = 1.28

def make_soft_hill_cmap() -> LinearSegmentedColormap:
    colors = [(0.0, '#0b5d47'), (0.3, '#3a9354'), (0.55, '#1e7340'), (0.78, '#115d34'), (0.93, '#0b472a'), (1.0, '#073720')]
    return LinearSegmentedColormap.from_list('SoftHill', colors)
Kind = Literal['cone', 'cylinder']

@dataclass(frozen=True)
class Obstacle:
    center: Tuple[float, float]
    radius: float
    height: float
    kind: Kind

def load_obstacles(environment_name: str | None=None) -> List[Obstacle]:
    set_name = str(environment_name or os.environ.get('OBSTACLE_SET', 'medium_1')).strip()
    base = get_obstacle_base(set_name)
    obs: List[Obstacle] = []
    for item in base:
        c, r, h = item
        rr = float(r)
        hh = float(h)
        kind: Kind = 'cylinder' if abs(hh - CYLINDER_H) <= H_TOL else 'cone'
        obs.append(Obstacle(center=(float(c[0]), float(c[1])), radius=rr, height=hh, kind=kind))
    return obs

def load_obstacle_tuples(environment_name: str | None=None) -> List[Tuple[float, float, float, float, str]]:
    out: List[Tuple[float, float, float, float, str]] = []
    for ob in load_obstacles(environment_name):
        out.append((float(ob.center[0]), float(ob.center[1]), float(ob.radius), float(ob.height), str(ob.kind)))
    return out

def make_grids(limit: float=GRID_LIMIT, step: float=GRID_STEP, ss: int=SUPERSAMPLE):
    n_t = int(round(limit / step))
    n_t = max(1, n_t)
    axis_t = np.linspace(0.0, limit, n_t + 1, dtype=np.float64)
    Xt, Yt = np.meshgrid(axis_t, axis_t)
    if ss <= 1:
        return (Xt, Yt, Xt, Yt, n_t, 1)
    n_f = ss * n_t
    axis_f = np.linspace(0.0, limit, n_f + 1, dtype=np.float64)
    Xf, Yf = np.meshgrid(axis_f, axis_f)
    return (Xf, Yf, Xt, Yt, n_t, ss)

def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

def _normalize_to_range(Z: np.ndarray, z_min: float, z_max: float) -> np.ndarray:
    z0 = float(np.min(Z))
    z1 = float(np.max(Z))
    if z1 - z0 < 1e-12:
        return np.full_like(Z, fill_value=z_min, dtype=np.float64)
    Zn = (Z - z0) / (z1 - z0)
    return Zn * (z_max - z_min) + z_min

def _apply_edge_taper(Z: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    if not TERRAIN_EDGE_TAPER:
        return Z
    width = max(float(TERRAIN_EDGE_TAPER_WIDTH), 1e-09)
    d_left = X - float(np.min(X))
    d_right = float(np.max(X)) - X
    d_bottom = Y - float(np.min(Y))
    d_top = float(np.max(Y)) - Y
    d_edge = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    t = np.clip(d_edge / width, 0.0, 1.0)
    scale = TERRAIN_EDGE_TAPER_MIN + (1.0 - TERRAIN_EDGE_TAPER_MIN) * _smoothstep(t)
    return Z * scale

def build_mountain_base_terrain(X: np.ndarray, Y: np.ndarray, seed: int=TERRAIN_SEED, num_peaks: int=TERRAIN_PEAK_COUNT, peak_height: float=TERRAIN_PEAK_HEIGHT, valley_depth: float=TERRAIN_VALLEY_DEPTH, altitude_scale: float=TERRAIN_ALTITUDE_SCALE, radius_min: float=TERRAIN_RADIUS_MIN, radius_max: float=TERRAIN_RADIUS_MAX, edge_margin: float=TERRAIN_EDGE_MARGIN, noise_scale: float=TERRAIN_NOISE_SCALE) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    base = altitude_scale * (np.sin(0.03 * X) * np.sin(0.03 * Y) + 0.5 * np.sin(0.07 * X) * np.sin(0.07 * Y) + 0.25 * np.sin(0.1 * X) * np.sin(0.1 * Y) + noise_scale * rng.random(X.shape))
    x_min, x_max = (float(np.min(X)), float(np.max(X)))
    y_min, y_max = (float(np.min(Y)), float(np.max(Y)))
    margin = float(max(edge_margin, radius_max))
    if x_max - x_min <= 2.0 * margin:
        margin = max(0.0, 0.1 * (x_max - x_min))
    if y_max - y_min <= 2.0 * margin:
        margin = max(0.0, 0.1 * (y_max - y_min))
    xs = X[0, :]
    ys = Y[:, 0]
    peak_count = max(0, int(num_peaks))
    r_low = max(1e-06, min(float(radius_min), float(radius_max)))
    r_high = max(r_low, max(float(radius_min), float(radius_max)))
    for _ in range(peak_count):
        cx = rng.uniform(x_min + margin, x_max - margin)
        cy = rng.uniform(y_min + margin, y_max - margin)
        r = rng.uniform(r_low, r_high)
        sigma = max(r / 2.0, 1e-06)
        amp = float(peak_height) * rng.uniform(0.45, 1.0)
        ix0 = max(0, int(np.searchsorted(xs, cx - r, side='left')))
        ix1 = min(len(xs), int(np.searchsorted(xs, cx + r, side='right')))
        iy0 = max(0, int(np.searchsorted(ys, cy - r, side='left')))
        iy1 = min(len(ys), int(np.searchsorted(ys, cy + r, side='right')))
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        subX = X[iy0:iy1, ix0:ix1]
        subY = Y[iy0:iy1, ix0:ix1]
        dist2 = (subX - cx) ** 2 + (subY - cy) ** 2
        mask = dist2 <= r ** 2
        if not np.any(mask):
            continue
        local = np.zeros_like(dist2, dtype=np.float64)
        local[mask] = amp * np.exp(-dist2[mask] / (2.0 * sigma ** 2))
        base[iy0:iy1, ix0:ix1] += local
    base = _normalize_to_range(base, float(valley_depth), float(peak_height))
    base = _apply_edge_taper(base, X, Y)
    return np.clip(base, 0.0, None)

def height_cylinder_rounded(X: np.ndarray, Y: np.ndarray, center: Tuple[float, float], r: float, h: float, plat_frac: float=CYL_PLAT_FRAC, shoulder_frac: float=CYL_SHOULDER_FR, soft_top: float=CYL_SOFT_TOP) -> np.ndarray:
    dx, dy = (X - center[0], Y - center[1])
    rho = np.hypot(dx, dy)
    r_core = max(0.0, min(float(r), float(r) * float(plat_frac)))
    r_sh = max(1e-06, min(float(r) - r_core, float(r) * float(shoulder_frac)))
    Z = np.zeros_like(rho, dtype=np.float64)
    r_inner = r_core * (1.0 - np.clip(float(soft_top), 0.0, 0.2))
    inside_inner = rho <= r_inner
    Z[inside_inner] = float(h)
    ring = (rho > r_inner) & (rho <= r_core)
    if np.any(ring):
        t = (rho[ring] - r_inner) / max(r_core - r_inner, 1e-09)
        eps = 0.04
        Z[ring] = float(h) * (1.0 - _smoothstep(t) * eps)
    shoulder = (rho > r_core) & (rho < r_core + r_sh)
    if np.any(shoulder):
        t = (rho[shoulder] - r_core) / r_sh
        Z[shoulder] = float(h) * (1.0 - _smoothstep(t))
    return Z

def height_cone_dome(X: np.ndarray, Y: np.ndarray, center: Tuple[float, float], r: float, h: float, p: float=DOME_P, q: float=DOME_Q, blend: float=DOME_BLEND) -> np.ndarray:
    dx, dy = (X - center[0], Y - center[1])
    rho = np.hypot(dx, dy)
    t = np.clip(rho / max(float(r), 1e-09), 0.0, 1.0)
    raised_cosine = 0.5 * (1.0 + np.cos(np.pi * t))
    polynomial = np.clip(1.0 - t ** float(p), 0.0, 1.0) ** float(q)
    domed = (1.0 - float(blend)) * raised_cosine + float(blend) * polynomial
    domed[t >= 1.0] = 0.0
    return float(h) * domed

def _build_subtle_flat_terrain(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    Z = 0.7 * np.sin(2.0 * np.pi * X / GRID_LIMIT) + 0.7 * np.cos(2.0 * np.pi * Y / GRID_LIMIT)
    return np.clip(Z, 0.0, None)

def build_dem_aligned(Xf: np.ndarray, Yf: np.ndarray, Xt: np.ndarray, Yt: np.ndarray, obstacles: List[Obstacle], ss: int):
    if TERRAIN_ENABLE:
        Zf = build_mountain_base_terrain(Xf, Yf)
    else:
        Zf = _build_subtle_flat_terrain(Xf, Yf)
    for ob in obstacles:
        if ob.kind == 'cylinder':
            Zi = height_cylinder_rounded(Xf, Yf, ob.center, ob.radius, ob.height)
        else:
            Zi = height_cone_dome(Xf, Yf, ob.center, ob.radius, ob.height)
        Zf = np.maximum(Zf, Zi)
    Zf = np.clip(Zf, 0.0, None)
    if ss <= 1:
        return (Xf, Yf, Zf)
    Z = Zf[::ss, ::ss]
    if Z.shape != Xt.shape:
        Z = Z[:Xt.shape[0], :Xt.shape[1]]
    return (Xt, Yt, Z)

def _bilinear_height(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, xq: float, yq: float) -> float:
    xs, ys = (X[0, :], Y[:, 0])
    xq = float(xq)
    yq = float(yq)
    if not (xs[0] <= xq <= xs[-1] and ys[0] <= yq <= ys[-1]):
        return 0.0
    ix = int(np.clip(np.searchsorted(xs, xq) - 1, 0, len(xs) - 2))
    iy = int(np.clip(np.searchsorted(ys, yq) - 1, 0, len(ys) - 2))
    x1, x2 = (xs[ix], xs[ix + 1])
    y1, y2 = (ys[iy], ys[iy + 1])
    q11 = Z[iy, ix]
    q12 = Z[iy, ix + 1]
    q21 = Z[iy + 1, ix]
    q22 = Z[iy + 1, ix + 1]
    tx = 0.0 if abs(x2 - x1) < 1e-12 else (xq - x1) / (x2 - x1)
    ty = 0.0 if abs(y2 - y1) < 1e-12 else (yq - y1) / (y2 - y1)
    return float((1.0 - tx) * (1.0 - ty) * q11 + tx * (1.0 - ty) * q12 + (1.0 - tx) * ty * q21 + tx * ty * q22)

def query_dem_height(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, xq: float, yq: float) -> float:
    return _bilinear_height(X, Y, Z, xq, yq)

def render_dem_3d(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, start_xy: Tuple[float, float]=START_XY, start_z: Optional[float]=START_Z, goal_xy: Tuple[float, float]=GOAL_XY, goal_z: Optional[float]=GOAL_Z, save_path: Optional[str]=None, show: bool=True):
    fig = plt.figure(figsize=(7.8, 5.6))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    z_min, z_max = (float(np.min(Z)), float(np.max(Z)))
    zn = np.clip((Z - z_min) / max(1e-09, z_max - z_min), 0.0, 1.0) ** GAMMA
    cmap = make_soft_hill_cmap()
    facecolors = LightSource(azdeg=LIGHT_AZ, altdeg=LIGHT_ALT).shade(zn, cmap=cmap, vert_exag=1.0, blend_mode='soft')
    facecolors[..., :3][zn < 0.06] = np.array([130, 190, 125], dtype=float)[None, :] / 255.0
    if facecolors.shape[-1] == 3:
        facecolors = np.concatenate([facecolors, np.ones((*facecolors.shape[:2], 1))], axis=-1)
    facecolors[..., 3] = 0.92
    ax.plot_surface(X, Y, Z, facecolors=facecolors, linewidth=0, edgecolor='none', antialiased=False, shade=False)
    sz = start_z if start_z is not None else _bilinear_height(X, Y, Z, *start_xy)
    gz = goal_z if goal_z is not None else _bilinear_height(X, Y, Z, *goal_xy)
    z_axis_max = max(200.0, float(np.max(Z)) * 1.08, float(sz) + 10.0, float(gz) + 10.0)
    ax.set(xlim=(float(np.min(X)), float(np.max(X))), ylim=(float(np.min(Y)), float(np.max(Y))), zlim=(0.0, z_axis_max), xlabel='X/m', ylabel='Y/m', zlabel='Z/m')
    dx = float(np.max(X) - np.min(X))
    dy = float(np.max(Y) - np.min(Y))
    dz = float(z_axis_max)
    try:
        ax.set_box_aspect((dx, dy, dz * Z_ASPECT_SCALE))
    except Exception:
        pass
    ax.scatter(*start_xy, sz, marker='s', s=95, facecolors='none', edgecolors='#f4cc3a', linewidths=1.8, label='Starting point')
    ax.scatter(*goal_xy, gz, marker='*', s=160, edgecolors='k', facecolors='#8d3c2e', linewidths=0.8, label='Target point')
    ax.legend(loc='upper left')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=360)
    if show:
        plt.show()
    return (fig, ax)

def draw_path_combo(ax, P, color: str='cyan', lw: float=2.2, marker_size: float=4.0, mark_clearance: bool=True, label: str='SIPSO Path'):
    P_vis = np.asarray(P, dtype=float).copy()
    line, = ax.plot3D(P_vis[:, 0], P_vis[:, 1], P_vis[:, 2], '-', linewidth=lw, color=color, zorder=10, label=label)
    try:
        line.set_depthshade(False)
    except Exception:
        pass
    ax.plot3D(P_vis[:, 0], P_vis[:, 1], P_vis[:, 2], 'o', markersize=marker_size, color=color, zorder=11)

def load_dem(clearance: float=5.0):
    Xf, Yf, Xt, Yt, _, ss = make_grids()
    X, Y, Z = build_dem_aligned(Xf, Yf, Xt, Yt, load_obstacles(), ss)
    sz_surf = _bilinear_height(X, Y, Z, *START_XY)
    gz_surf = _bilinear_height(X, Y, Z, *GOAL_XY)
    clr = float(clearance)
    if START_Z is None:
        sz = float(sz_surf + clr)
    else:
        sz = max(float(START_Z), float(sz_surf + clr))
    if GOAL_Z is None:
        gz = float(gz_surf + clr)
    else:
        gz = max(float(GOAL_Z), float(gz_surf + clr))
    START = np.array([START_XY[0], START_XY[1], sz], dtype=float)
    GOAL = np.array([GOAL_XY[0], GOAL_XY[1], gz], dtype=float)
    return (X, Y, Z, START, GOAL)

def main():
    X, Y, Z, START, GOAL = load_dem(clearance=5.0)
    render_dem_3d(X, Y, Z, start_xy=(float(START[0]), float(START[1])), start_z=float(START[2]), goal_xy=(float(GOAL[0]), float(GOAL[1])), goal_z=float(GOAL[2]), show=True)
if __name__ == '__main__':
    main()

def _seed_all(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed)
    random.seed(seed)

def make_height_fn(X, Y, Z):
    xs = X[0, :]
    ys = Y[:, 0]

    def _h(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        ix = np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2)
        iy = np.clip(np.searchsorted(ys, y) - 1, 0, len(ys) - 2)
        x1, x2 = (xs[ix], xs[ix + 1])
        y1, y2 = (ys[iy], ys[iy + 1])
        q11 = Z[iy, ix]
        q21 = Z[iy + 1, ix]
        q12 = Z[iy, ix + 1]
        q22 = Z[iy + 1, ix + 1]
        tx = np.where(x2 == x1, 0.0, (x - x1) / (x2 - x1))
        ty = np.where(y2 == y1, 0.0, (y - y1) / (y2 - y1))
        return (1 - tx) * (1 - ty) * q11 + tx * (1 - ty) * q12 + (1 - tx) * ty * q21 + tx * ty * q22
    return _h

def make_obstacles_for_cost(radius_expand: float=0.0):
    obs_dc = load_obstacle_tuples()
    out = []
    re = float(radius_expand)
    for cx, cy, r, h, kind in obs_dc:
        k = str(kind).lower()
        kind2 = 'cylinder' if 'cyl' in k else 'cone'
        r2 = max(0.0, float(r) + re)
        out.append((float(cx), float(cy), r2, float(h), kind2))
    return out

def path_length(P: np.ndarray) -> float:
    P = np.asarray(P, dtype=float)
    if P.shape[0] < 2:
        return 0.0
    diffs = P[1:] - P[:-1]
    return float(np.sum(np.linalg.norm(diffs, axis=1)))
