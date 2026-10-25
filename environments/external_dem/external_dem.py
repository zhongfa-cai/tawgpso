"""Frozen CI-EXT and MSH-EXT native-metric DEM loading, decoding, context construction, and path audit."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from evaluation.objectives import DEFAULT_WEIGHTS, build_fixed_path_from_xyz, path_cost_from_path, total_cost_fixed_waypoints, total_cost_sipso_adaptive
from configuration import protocol
from algorithm.tawg_pso import build_path_from_xg

Array = np.ndarray

def sha256_file(path: str | Path, chunk_size: int=1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()

def read_pillow_dem(path: str | Path) -> dict[str, Any]:
    image_path = Path(path)
    with Image.open(image_path) as img:
        arr = np.asarray(img, dtype=np.float64)
        tags = dict(img.tag_v2)
        width, height = img.size
    nodata = _nodata_from_tiff_tags(tags)
    return {'path': str(image_path), 'array': arr, 'width': int(width), 'height': int(height), 'tags': tags, 'nodata': nodata}

def _nodata_from_tiff_tags(tags: dict[int, Any]) -> float | None:
    raw = tags.get(42113)
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode('ascii', errors='ignore')
    if isinstance(raw, (tuple, list)) and raw:
        raw = raw[0]
    text = str(raw).strip('\x00 \t\r\n')
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None

def valid_mask(array: Array, nodata: float | None) -> Array:
    arr = np.asarray(array, dtype=np.float64)
    mask = np.isfinite(arr)
    if nodata is not None:
        nodata_value = float(nodata)
        if math.isfinite(nodata_value):
            mask &= ~np.isclose(arr, nodata_value, rtol=1e-06, atol=0.0)
        else:
            mask &= arr != nodata_value
    return mask

def dem_stats(array: Array, nodata: float | None) -> dict[str, float | int]:
    arr = np.asarray(array, dtype=np.float64)
    mask = valid_mask(arr, nodata)
    valid = arr[mask]
    stats: dict[str, float | int] = {'valid_count': int(mask.sum()), 'invalid_count': int(mask.size - mask.sum())}
    if valid.size:
        stats.update({'valid_min': float(valid.min()), 'valid_max': float(valid.max()), 'valid_mean': float(valid.mean())})
    else:
        stats.update({'valid_min': math.nan, 'valid_max': math.nan, 'valid_mean': math.nan})
    return stats

def strict_bilinear_height(array: Array, mask: Array, x_m: float, y_m: float, *, resolution_m: float) -> float:
    arr = np.asarray(array, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    row_f = float(y_m) / float(resolution_m)
    col_f = float(x_m) / float(resolution_m)
    max_row = arr.shape[0] - 1
    max_col = arr.shape[1] - 1
    eps = 1e-09
    if row_f < -eps or col_f < -eps or row_f > max_row + eps or (col_f > max_col + eps):
        return math.nan
    row_f = min(max(row_f, 0.0), float(max_row))
    col_f = min(max(col_f, 0.0), float(max_col))
    row0 = int(math.floor(row_f))
    col0 = int(math.floor(col_f))
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        return math.nan
    if row0 >= max_row:
        row0 = max_row - 1
    if col0 >= max_col:
        col0 = max_col - 1
    row1 = row0 + 1
    col1 = col0 + 1
    if not bool(valid[row0, col0] and valid[row0, col1] and valid[row1, col0] and valid[row1, col1]):
        return math.nan
    dr = row_f - row0
    dc = col_f - col0
    z00 = arr[row0, col0]
    z01 = arr[row0, col1]
    z10 = arr[row1, col0]
    z11 = arr[row1, col1]
    return float(z00 * (1.0 - dr) * (1.0 - dc) + z01 * (1.0 - dr) * dc + z10 * dr * (1.0 - dc) + z11 * dr * dc)

def point_valid_at_grid_center(array: Array, mask: Array, x_m: float, y_m: float, *, resolution_m: float) -> bool:
    row = int(round(float(y_m) / float(resolution_m)))
    col = int(round(float(x_m) / float(resolution_m)))
    return 0 <= row < array.shape[0] and 0 <= col < array.shape[1] and bool(mask[row, col])

def normalized_to_physical(xyz_norm: Array, *, hmin: float, hmax: float, x_max_m: float=4610.0, y_max_m: float=5000.0, norm_max: float=200.0) -> Array:
    pts = np.asarray(xyz_norm, dtype=np.float64)
    out = pts.copy()
    out[..., 0] = pts[..., 0] / norm_max * x_max_m
    out[..., 1] = pts[..., 1] / norm_max * y_max_m
    out[..., 2] = hmin + pts[..., 2] / norm_max * (hmax - hmin)
    return out

def normalized_agl_to_physical(xyz_norm: Array, *, height_fn: Any, agl_min: float=5.0, agl_max: float=120.0, x_max_m: float=4610.0, y_max_m: float=5000.0, norm_max: float=200.0) -> Array:
    pts = np.asarray(xyz_norm, dtype=np.float64)
    out = pts.copy()
    out[..., 0] = pts[..., 0] / norm_max * x_max_m
    out[..., 1] = pts[..., 1] / norm_max * y_max_m
    agl = float(agl_min) + pts[..., 2] / norm_max * (float(agl_max) - float(agl_min))
    terrain = np.asarray(height_fn(out[..., 0], out[..., 1]), dtype=np.float64)
    out[..., 2] = terrain + agl
    return out
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
PROTOCOL_PATH = PACKAGE_DIR / 'protocol.json'

@dataclass(frozen=True)
class TileKey:
    easting: int
    northing: int

    @property
    def label(self) -> str:
        return f'{self.easting}_{self.northing}'

def load_protocol(path: str | Path | None=None) -> dict[str, Any]:
    protocol_path = Path(path) if path is not None else PROTOCOL_PATH
    return json.loads(protocol_path.read_text(encoding='utf-8'))

def repo_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return REPO_ROOT / path_obj

def required_msh_tile_keys(protocol: dict[str, Any] | None=None) -> list[TileKey]:
    data = protocol if protocol is not None else load_protocol()
    return [TileKey(easting=int(item['easting']), northing=int(item['northing'])) for item in data['msh_ext']['tile_keys']]

def threat_rows_cols(protocol: dict[str, Any] | None=None) -> list[tuple[str, int, int, float]]:
    data = protocol if protocol is not None else load_protocol()
    res = float(data['common_grid']['resolution_m'])
    threats = []
    for item in data['mission']['threats']:
        x_m, y_m = item['center_xy_m']
        col = int(round(float(x_m) / res))
        row = int(round(float(y_m) / res))
        threats.append((str(item['id']), row, col, float(item['radius_m'])))
    return threats
Array = np.ndarray

def sample_path_horizontal(path: Array, *, max_spacing_m: float=1.0) -> Array:
    points = np.asarray(path, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3 or points.shape[0] < 2:
        raise ValueError('path must have shape (n_points, 3) with at least two points')
    out: list[Array] = []
    spacing = max(float(max_spacing_m), 1e-09)
    for i, (a, b) in enumerate(zip(points[:-1, :3], points[1:, :3])):
        horiz = float(np.hypot(*b[:2] - a[:2]))
        n = max(2, int(math.ceil(horiz / spacing)) + 1)
        ts = np.linspace(0.0, 1.0, n, dtype=np.float64)
        segment = a[None, :] + ts[:, None] * (b[None, :] - a[None, :])
        if i > 0:
            segment = segment[1:]
        out.append(segment)
    return np.vstack(out)

def _height_values(height_fn: Any, x: Array, y: Array) -> Array:
    try:
        return np.asarray(height_fn(x, y), dtype=np.float64)
    except Exception:
        flat_x = np.asarray(x, dtype=np.float64).reshape(-1)
        flat_y = np.asarray(y, dtype=np.float64).reshape(-1)
        vals = np.empty_like(flat_x, dtype=np.float64)
        for i, (xx, yy) in enumerate(zip(flat_x, flat_y)):
            vals[i] = float(height_fn(float(xx), float(yy)))
        return vals.reshape(np.asarray(x).shape)

def audit_physical_path(path: Array, ctx: Mapping[str, Any], *, max_spacing_m: float=1.0, tol: float=1e-08) -> dict[str, Any]:
    samples = sample_path_horizontal(path, max_spacing_m=max_spacing_m)
    cfg = dict(ctx.get('cfg', {}))
    height_fn = ctx.get('height_fn')
    if not callable(height_fn):
        raise ValueError('ctx must provide callable height_fn')
    terrain = _height_values(height_fn, samples[:, 0], samples[:, 1])
    z = samples[:, 2]
    invalid_terrain = ~np.isfinite(terrain)
    agl = z - terrain
    hmin = float(cfg.get('HMIN', ctx.get('HMIN', -np.inf)))
    hmax = float(cfg.get('HMAX', ctx.get('HMAX', np.inf)))
    agl_min = float(cfg.get('AGL_MIN', ctx.get('AGL_MIN', 5.0)))
    agl_max_raw = cfg.get('AGL_MAX', ctx.get('AGL_MAX', 120.0))
    agl_max = None if agl_max_raw is None else float(agl_max_raw)
    safety_margin = float(cfg.get('SAFETY_MARGIN', ctx.get('SAFETY_MARGIN', 5.0)))
    finite_z = np.isfinite(z)
    height_low = ~finite_z | (z < hmin - tol)
    height_high = ~finite_z | (z > hmax + tol)
    agl_low = invalid_terrain | (agl < agl_min - tol)
    if agl_max is None:
        agl_high = np.zeros_like(agl_low, dtype=bool)
    else:
        agl_high = invalid_terrain | (agl > agl_max + tol)
    min_clearance = float('inf')
    obstacle_violation = np.zeros(samples.shape[0], dtype=bool)
    for ob in list(ctx.get('obstacles', [])):
        cx, cy, radius, *_rest = ob
        clearance = np.hypot(samples[:, 0] - float(cx), samples[:, 1] - float(cy)) - float(radius)
        if clearance.size:
            min_clearance = min(min_clearance, float(np.nanmin(clearance)))
        obstacle_violation |= clearance < safety_margin - tol
    valid_agl = agl[~invalid_terrain & np.isfinite(agl)]
    errors: list[str] = []
    if np.any(invalid_terrain):
        errors.append('invalid_dem')
    if np.any(height_low):
        errors.append('height_low')
    if np.any(height_high):
        errors.append('height_high')
    if np.any(agl_low):
        errors.append('agl_low')
    if np.any(agl_high):
        errors.append('agl_high')
    if np.any(obstacle_violation):
        errors.append('obstacle_or_safety_margin')
    return {'passed': not errors, 'errors': errors, 'sample_count_1m': int(samples.shape[0]), 'max_spacing_m': float(max_spacing_m), 'invalid_dem_count': int(np.count_nonzero(invalid_terrain)), 'height_low_count': int(np.count_nonzero(height_low)), 'height_high_count': int(np.count_nonzero(height_high)), 'agl_low_count': int(np.count_nonzero(agl_low)), 'agl_high_count': int(np.count_nonzero(agl_high)), 'obstacle_violation_count': int(np.count_nonzero(obstacle_violation)), 'min_clearance_1m': float(min_clearance), 'min_agl_1m': float(np.min(valid_agl)) if valid_agl.size else float('nan'), 'max_agl_1m': float(np.max(valid_agl)) if valid_agl.size else float('nan')}

def path_is_feasible_1m(path: Array, ctx: Mapping[str, Any], *, tol: float=1e-08) -> bool:
    try:
        return bool(audit_physical_path(path, ctx, max_spacing_m=1.0, tol=tol)['passed'])
    except Exception:
        return False
Array = np.ndarray
NORM_MAX = 200.0
EXTERNAL_SEARCH_AUDIT_SPACING_M = 1.0
EXTERNAL_CONSTRAINT_PENALTY_SCALE = 1000.0
EXTERNAL_POST_DECODE_PATH_REPAIR_POLICY = 'disabled_for_all_algorithms; decoded paths are evaluated as returned after physical coordinate conversion'

def _as_scalar_or_array(template: Any, value: float) -> Array:
    arr = np.asarray(template, dtype=np.float64)
    return np.full(arr.shape, float(value), dtype=np.float64)

@dataclass(frozen=True)
class ExternalTerrain:
    scene_key: str
    source_path: Path
    array: Array
    mask: Array
    nodata: float | None
    resolution_m: float
    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float

    @property
    def width(self) -> int:
        return int(self.array.shape[1])

    @property
    def height(self) -> int:
        return int(self.array.shape[0])

    def valid_values(self) -> Array:
        return np.asarray(self.array, dtype=np.float64)[self.mask]

    def logical_mask(self) -> Array:
        return self.mask & self.logical_extent_mask()

    def logical_extent_mask(self) -> Array:
        rows = np.arange(self.array.shape[0], dtype=np.float64)[:, None] * float(self.resolution_m)
        cols = np.arange(self.array.shape[1], dtype=np.float64)[None, :] * float(self.resolution_m)
        eps = 1e-09
        return (cols >= float(self.x_min_m) - eps) & (cols <= float(self.x_max_m) + eps) & (rows >= float(self.y_min_m) - eps) & (rows <= float(self.y_max_m) + eps)

    def logical_valid_values(self) -> Array:
        return np.asarray(self.array, dtype=np.float64)[self.logical_mask()]

    def logical_stats(self) -> tuple[int, int, float, float]:
        row_start = max(0, int(math.ceil(float(self.y_min_m) / float(self.resolution_m) - 1e-09)))
        row_stop = min(self.height, int(math.floor(float(self.y_max_m) / float(self.resolution_m) + 1e-09)) + 1)
        col_start = max(0, int(math.ceil(float(self.x_min_m) / float(self.resolution_m) - 1e-09)))
        col_stop = min(self.width, int(math.floor(float(self.x_max_m) / float(self.resolution_m) + 1e-09)) + 1)
        window = self.array[row_start:row_stop, col_start:col_stop]
        mask = self.mask[row_start:row_stop, col_start:col_stop]
        valid_count = int(np.count_nonzero(mask))
        invalid_count = int(mask.size - valid_count)
        if valid_count <= 0:
            return (0, invalid_count, float('nan'), float('nan'))
        if valid_count == mask.size:
            return (valid_count, invalid_count, float(np.min(window)), float(np.max(window)))
        valid = window[mask]
        return (valid_count, invalid_count, float(np.min(valid)), float(np.max(valid)))

    def height_physical(self, x: Any, y: Any) -> Array:
        xs, ys = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
        row_f = ys / float(self.resolution_m)
        col_f = xs / float(self.resolution_m)
        max_row = self.array.shape[0] - 1
        max_col = self.array.shape[1] - 1
        logical_min_row = float(self.y_min_m) / float(self.resolution_m)
        logical_max_row = float(self.y_max_m) / float(self.resolution_m)
        logical_min_col = float(self.x_min_m) / float(self.resolution_m)
        logical_max_col = float(self.x_max_m) / float(self.resolution_m)
        eps = 1e-09
        in_extent = (row_f >= logical_min_row - eps) & (col_f >= logical_min_col - eps) & (row_f <= logical_max_row + eps) & (col_f <= logical_max_col + eps)
        row_f = np.clip(row_f, 0.0, float(max_row))
        col_f = np.clip(col_f, 0.0, float(max_col))
        row0 = np.floor(row_f).astype(np.int64)
        col0 = np.floor(col_f).astype(np.int64)
        out = np.full(xs.shape, np.nan, dtype=np.float64)
        if self.array.shape[0] < 2 or self.array.shape[1] < 2:
            return out
        row0 = np.where(row0 >= max_row, max_row - 1, row0)
        col0 = np.where(col0 >= max_col, max_col - 1, col0)
        row1 = row0 + 1
        col1 = col0 + 1
        inside = in_extent & (row0 >= 0) & (col0 >= 0) & (row1 < self.array.shape[0]) & (col1 < self.array.shape[1])
        if not np.any(inside):
            return out
        r0 = row0[inside]
        r1 = row1[inside]
        c0 = col0[inside]
        c1 = col1[inside]
        ok = self.mask[r0, c0] & self.mask[r0, c1] & self.mask[r1, c0] & self.mask[r1, c1]
        if not np.any(ok):
            return out
        idx = np.flatnonzero(inside)
        good_idx = idx[ok]
        rr0 = r0[ok]
        rr1 = r1[ok]
        cc0 = c0[ok]
        cc1 = c1[ok]
        dr = row_f.reshape(-1)[good_idx] - rr0
        dc = col_f.reshape(-1)[good_idx] - cc0
        arr = np.asarray(self.array, dtype=np.float64)
        z00 = arr[rr0, cc0]
        z01 = arr[rr0, cc1]
        z10 = arr[rr1, cc0]
        z11 = arr[rr1, cc1]
        out.reshape(-1)[good_idx] = z00 * (1.0 - dr) * (1.0 - dc) + z01 * (1.0 - dr) * dc + z10 * dr * (1.0 - dc) + z11 * dr * dc
        return out

@dataclass(frozen=True)
class ExternalScene:
    key: str
    scene_id: str
    terrain: ExternalTerrain
    x_max_m: float
    y_max_m: float
    hmin: float
    hmax: float
    start_physical: Array
    goal_physical: Array
    start_normalized: Array
    goal_normalized: Array
    obstacles_physical: tuple[tuple[float, float, float, float, str], ...]
    obstacles_normalized: tuple[tuple[float, float, float, float, str], ...]
    cfg_physical: dict[str, Any]
    cfg_search: dict[str, Any]
    X: Array
    Y: Array
    native_audit_terrain: ExternalTerrain | None = None

def _scene_alias(scene: str) -> str:
    value = str(scene).strip().lower().replace('-', '_')
    if value in {'ci', 'ci_ext', 'ciext'}:
        return 'ci_ext'
    if value in {'msh', 'msh_ext', 'mshext'}:
        return 'msh_ext'
    raise ValueError(f'Unknown external DEM scene: {scene!r}')

def _load_terrain(scene_key: str, spec: Mapping[str, Any], grid: Mapping[str, Any]) -> ExternalTerrain:
    if scene_key == 'ci_ext':
        path = repo_path(spec['source_path'])
        resolution_m = float(spec.get('source_resolution_m', 5.0))
    else:
        path = repo_path(spec['one_m_window_path'])
        resolution_m = float(spec['one_m_window']['resolution_m'])
    dem = read_pillow_dem(path)
    arr = np.asarray(dem['array'], dtype=np.float64)
    nodata = dem['nodata']
    mask = valid_mask(arr, nodata)
    return ExternalTerrain(scene_key=scene_key, source_path=path, array=arr, mask=mask, nodata=nodata, resolution_m=resolution_m, x_min_m=float(grid['x_center_min_m']), x_max_m=float(grid['x_center_max_m']), y_min_m=float(grid['y_center_min_m']), y_max_m=float(grid['y_center_max_m']))

def _load_native_audit_terrain(scene_key: str, data: Mapping[str, Any]) -> ExternalTerrain | None:
    del scene_key, data
    return None

def _terrain_scalar(terrain: ExternalTerrain, xy: list[float] | tuple[float, float]) -> float:
    z = float(np.asarray(terrain.height_physical(float(xy[0]), float(xy[1]))).reshape(-1)[0])
    if not np.isfinite(z):
        raise ValueError(f'Point {xy} is invalid for {terrain.scene_key} terrain.')
    return z

def _normalized_xy_agl(*, terrain: ExternalTerrain, points: Array, x_max_m: float, y_max_m: float, agl_min: float, agl_max: float) -> Array:
    pts = np.asarray(points, dtype=np.float64)
    out = pts.copy()
    out[..., 0] = pts[..., 0] / float(x_max_m) * NORM_MAX
    out[..., 1] = pts[..., 1] / float(y_max_m) * NORM_MAX
    ground = np.asarray(terrain.height_physical(pts[..., 0], pts[..., 1]), dtype=np.float64)
    agl = pts[..., 2] - ground
    out[..., 2] = (agl - float(agl_min)) / (float(agl_max) - float(agl_min)) * NORM_MAX
    return out

def _physical_xyz(scene: ExternalScene, points: Array) -> Array:
    pts = np.asarray(points, dtype=np.float64)
    out = pts.copy()
    out[..., 0] = pts[..., 0] / NORM_MAX * scene.x_max_m
    out[..., 1] = pts[..., 1] / NORM_MAX * scene.y_max_m
    agl_min = float(scene.cfg_physical['AGL_MIN'])
    agl_max = float(scene.cfg_physical['AGL_MAX'])
    agl = agl_min + pts[..., 2] / NORM_MAX * (agl_max - agl_min)
    terrain = np.asarray(scene.terrain.height_physical(out[..., 0], out[..., 1]), dtype=np.float64)
    fallback_ground = float(scene.hmin) - agl_min
    terrain_for_z = np.where(np.isfinite(terrain), terrain, fallback_ground)
    out[..., 2] = terrain_for_z + agl
    return out

def _looks_normalized_path(path: Array) -> bool:
    pts = np.asarray(path, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return False
    return bool(np.all(np.isfinite(pts[:, :3])) and np.nanmin(pts[:, :3]) >= -1e-09 and (np.nanmax(pts[:, :3]) <= NORM_MAX + 1e-09))

def _external_dense_constraint_penalty(path: Array, scene: ExternalScene) -> tuple[float, dict[str, Any]]:
    ctx = {'height_fn': scene.terrain.height_physical, 'obstacles': list(scene.obstacles_physical), 'cfg': scene.cfg_physical, 'HMIN': float(scene.hmin), 'HMAX': float(scene.hmax), 'AGL_MIN': float(scene.cfg_physical['AGL_MIN']), 'AGL_MAX': float(scene.cfg_physical['AGL_MAX']), 'SAFETY_MARGIN': float(scene.cfg_physical['SAFETY_MARGIN'])}
    try:
        audit = audit_physical_path(np.asarray(path, dtype=np.float64), ctx, max_spacing_m=EXTERNAL_SEARCH_AUDIT_SPACING_M)
    except Exception as exc:
        return (float(EXTERNAL_CONSTRAINT_PENALTY_SCALE * 10000.0), {'passed': False, 'errors': ['search_audit_exception'], 'exception': str(exc)})
    if bool(audit['passed']):
        return (0.0, audit)
    n = float(max(int(audit.get('sample_count_1m', 0)), 1))
    agl_min = float(scene.cfg_physical['AGL_MIN'])
    agl_max = float(scene.cfg_physical['AGL_MAX'])
    safety = float(scene.cfg_physical['SAFETY_MARGIN'])
    min_agl = float(audit.get('min_agl_1m', np.nan))
    max_agl = float(audit.get('max_agl_1m', np.nan))
    min_clearance = float(audit.get('min_clearance_1m', np.nan))
    count_penalty = 400.0 * float(audit.get('invalid_dem_count', 0)) / n + 120.0 * float(audit.get('height_low_count', 0)) / n + 120.0 * float(audit.get('height_high_count', 0)) / n + 160.0 * float(audit.get('agl_low_count', 0)) / n + 120.0 * float(audit.get('agl_high_count', 0)) / n + 180.0 * float(audit.get('obstacle_violation_count', 0)) / n
    gap_penalty = 0.0
    if np.isfinite(min_agl):
        gap_penalty += 20.0 * max(agl_min - min_agl, 0.0)
    else:
        gap_penalty += 1000.0
    if np.isfinite(max_agl):
        gap_penalty += 8.0 * max(max_agl - agl_max, 0.0)
    else:
        gap_penalty += 1000.0
    if np.isfinite(min_clearance):
        gap_penalty += 40.0 * max(safety - min_clearance, 0.0)
    else:
        gap_penalty += 1000.0
    return (float(EXTERNAL_CONSTRAINT_PENALTY_SCALE * (count_penalty + gap_penalty)), audit)

class ExternalDemCost:

    def __init__(self, *, scene: ExternalScene, mode: str, gate_cfg: Any | None=None, sort_waypoints: bool=True) -> None:
        self.scene = scene
        self.mode = str(mode)
        self.gate_cfg = gate_cfg
        self.sort_waypoints = bool(sort_waypoints)
        self.state = {'iteration': None, 'max_iter': None}
        self.encoding = f'external_{self.mode}_normalized'

    def set_iteration(self, iteration: int | None, max_iter_override: int | None=None) -> None:
        self.state['iteration'] = None if iteration is None else int(iteration)
        if max_iter_override is not None:
            self.state['max_iter'] = int(max_iter_override)

    def normalized_xyz_to_physical(self, xyz: Array) -> Array:
        return _physical_xyz(self.scene, xyz)

    def _external_penalty_for_path(self, physical_path: Array) -> tuple[float, dict[str, Any]]:
        return _external_dense_constraint_penalty(physical_path, self.scene)

    def _external_penalty_for_vector(self, vector: Array) -> tuple[float, dict[str, Any]]:
        try:
            path, _info = self.decode_path(vector)
        except Exception as exc:
            return (float(EXTERNAL_CONSTRAINT_PENALTY_SCALE * 10000.0), {'passed': False, 'errors': ['decode_exception'], 'exception': str(exc)})
        return self._external_penalty_for_path(path)

    def path_to_physical(self, path: Array) -> Array:
        P = np.asarray(path, dtype=np.float64)
        if _looks_normalized_path(P):
            return self.normalized_xyz_to_physical(P[:, :3])
        return P[:, :3].copy()

    def _xg_to_physical(self, xg: Array) -> Array:
        arr = np.asarray(xg, dtype=np.float64).reshape(-1, 4).copy()
        arr[:, :3] = self.normalized_xyz_to_physical(arr[:, :3])
        return arr.reshape(-1)

    def _xyz_to_physical(self, xyz: Array) -> Array:
        arr = np.asarray(xyz, dtype=np.float64).reshape(-1, 3).copy()
        return self.normalized_xyz_to_physical(arr).reshape(-1)

    def __call__(self, vector: Array) -> float:
        if self.mode == 'adaptive':
            base = float(total_cost_sipso_adaptive(self._xg_to_physical(vector), start=self.scene.start_physical, goal=self.scene.goal_physical, obstacles=self.scene.obstacles_physical, height_fn=self.scene.terrain.height_physical, gate_cfg=self.gate_cfg, hmin=self.scene.hmin, hmax=self.scene.hmax, agl_min=float(self.scene.cfg_physical['AGL_MIN']), agl_max=float(self.scene.cfg_physical['AGL_MAX']), safety_margin=float(self.scene.cfg_physical['SAFETY_MARGIN']), forbid_overfly=True, overfly_margin=float(self.scene.cfg_physical['OVERFLY_MARGIN']), weights=DEFAULT_WEIGHTS, iteration=self.state['iteration'], max_iter=self.state['max_iter'], return_detail=False))
        else:
            base = float(total_cost_fixed_waypoints(self._xyz_to_physical(vector), start=self.scene.start_physical, goal=self.scene.goal_physical, obstacles=self.scene.obstacles_physical, height_fn=self.scene.terrain.height_physical, hmin=self.scene.hmin, hmax=self.scene.hmax, agl_min=float(self.scene.cfg_physical['AGL_MIN']), agl_max=float(self.scene.cfg_physical['AGL_MAX']), safety_margin=float(self.scene.cfg_physical['SAFETY_MARGIN']), forbid_overfly=True, overfly_margin=float(self.scene.cfg_physical['OVERFLY_MARGIN']), weights=DEFAULT_WEIGHTS, sort_waypoints=self.sort_waypoints, return_detail=False))
        if not np.isfinite(base):
            base = float(EXTERNAL_CONSTRAINT_PENALTY_SCALE * 1000.0)
        penalty, _audit = self._external_penalty_for_vector(vector)
        return float(base + penalty)

    def detail(self, vector: Array) -> dict[str, Any]:
        if self.mode == 'adaptive':
            fit, detail = total_cost_sipso_adaptive(self._xg_to_physical(vector), start=self.scene.start_physical, goal=self.scene.goal_physical, obstacles=self.scene.obstacles_physical, height_fn=self.scene.terrain.height_physical, gate_cfg=self.gate_cfg, hmin=self.scene.hmin, hmax=self.scene.hmax, agl_min=float(self.scene.cfg_physical['AGL_MIN']), agl_max=float(self.scene.cfg_physical['AGL_MAX']), safety_margin=float(self.scene.cfg_physical['SAFETY_MARGIN']), forbid_overfly=True, overfly_margin=float(self.scene.cfg_physical['OVERFLY_MARGIN']), weights=DEFAULT_WEIGHTS, iteration=self.state['iteration'], max_iter=self.state['max_iter'], return_detail=True)
        else:
            fit, detail = total_cost_fixed_waypoints(self._xyz_to_physical(vector), start=self.scene.start_physical, goal=self.scene.goal_physical, obstacles=self.scene.obstacles_physical, height_fn=self.scene.terrain.height_physical, hmin=self.scene.hmin, hmax=self.scene.hmax, agl_min=float(self.scene.cfg_physical['AGL_MIN']), agl_max=float(self.scene.cfg_physical['AGL_MAX']), safety_margin=float(self.scene.cfg_physical['SAFETY_MARGIN']), forbid_overfly=True, overfly_margin=float(self.scene.cfg_physical['OVERFLY_MARGIN']), weights=DEFAULT_WEIGHTS, sort_waypoints=self.sort_waypoints, return_detail=True)
        penalty, audit = self._external_penalty_for_vector(vector)
        if not np.isfinite(fit):
            fit = float(EXTERNAL_CONSTRAINT_PENALTY_SCALE * 1000.0)
        out = dict(detail)
        out['external_search_audit_spacing_m'] = float(EXTERNAL_SEARCH_AUDIT_SPACING_M)
        out['external_dense_constraint_penalty'] = float(penalty)
        out['external_search_audit_passed'] = float(bool(audit.get('passed', False)))
        out['fitness'] = float(fit + penalty)
        out['encoding'] = self.encoding
        return out

    def decode_path(self, vector: Array) -> tuple[Array, dict[str, Any]]:
        if self.mode == 'adaptive':
            path, info = build_path_from_xg(self._xg_to_physical(vector), self.scene.start_physical, self.scene.goal_physical, self.gate_cfg)
        else:
            path, info = build_fixed_path_from_xyz(self._xyz_to_physical(vector), self.scene.start_physical, self.scene.goal_physical, sort_waypoints=self.sort_waypoints)
        out = dict(info)
        out['encoding'] = self.encoding
        out['path_space'] = 'physical_m'
        return (np.asarray(path, dtype=np.float64), out)

    def evaluate_path(self, path: Array, return_detail: bool=False):
        physical = self.path_to_physical(path)
        out = path_cost_from_path(physical, start=self.scene.start_physical, goal=self.scene.goal_physical, obstacles=self.scene.obstacles_physical, height_fn=self.scene.terrain.height_physical, hmin=self.scene.hmin, hmax=self.scene.hmax, agl_min=float(self.scene.cfg_physical['AGL_MIN']), agl_max=float(self.scene.cfg_physical['AGL_MAX']), safety_margin=float(self.scene.cfg_physical['SAFETY_MARGIN']), forbid_overfly=True, overfly_margin=float(self.scene.cfg_physical['OVERFLY_MARGIN']), weights=DEFAULT_WEIGHTS, return_detail=return_detail)
        penalty, audit = self._external_penalty_for_path(physical)
        if return_detail:
            fit, detail = out
            detail = dict(detail)
            detail['external_search_audit_spacing_m'] = float(EXTERNAL_SEARCH_AUDIT_SPACING_M)
            detail['external_dense_constraint_penalty'] = float(penalty)
            detail['external_search_audit_passed'] = float(bool(audit.get('passed', False)))
            if not np.isfinite(fit):
                fit = float(EXTERNAL_CONSTRAINT_PENALTY_SCALE * 1000.0)
            return (float(fit + penalty), detail)
        fit = float(out)
        if not np.isfinite(fit):
            fit = float(EXTERNAL_CONSTRAINT_PENALTY_SCALE * 1000.0)
        return float(fit + penalty)

def load_external_scene(scene: str, protocol_data: dict[str, Any] | None=None) -> ExternalScene:
    data = protocol_data or load_protocol()
    key = _scene_alias(scene)
    grid = data['common_grid']
    mission = data['mission']
    terrain = _load_terrain(key, data[key], grid)
    if key == 'ci_ext':
        expected_shape = (int(data[key]['source_height_rows']), int(data[key]['source_width_cols']))
    else:
        expected_shape = (int(data[key]['one_m_window']['height_rows']), int(data[key]['one_m_window']['width_cols']))
    if terrain.array.shape != expected_shape:
        raise ValueError(f'{key} terrain shape mismatch: {terrain.array.shape}, expected {expected_shape}')
    valid_count, _invalid_count, valid_min, valid_max = terrain.logical_stats()
    if valid_count <= 0:
        raise ValueError(f'{key} terrain has no valid cells inside the logical experiment bounds.')
    agl_min = float(mission['agl_min_m'])
    agl_max = float(mission['agl_max_m'])
    hmin = float(valid_min + agl_min)
    hmax = float(valid_max + agl_max)
    start_xy = [float(x) for x in mission['start_xy_m']]
    goal_xy = [float(x) for x in mission['goal_xy_m']]
    start = np.array([start_xy[0], start_xy[1], _terrain_scalar(terrain, start_xy) + float(mission['start_agl_m'])])
    goal = np.array([goal_xy[0], goal_xy[1], _terrain_scalar(terrain, goal_xy) + float(mission['goal_agl_m'])])
    cfg_physical = protocol.make_cfg(overrides={'HMIN': hmin, 'HMAX': hmax, 'AGL_MIN': agl_min, 'AGL_MAX': agl_max, 'SAFETY_MARGIN': float(mission['safety_margin_m']), 'FORBID_OVERFLY': True, 'OVERFLY_MARGIN': float(mission['safety_margin_m'])})
    x_max = float(grid['x_center_max_m'])
    y_max = float(grid['y_center_max_m'])
    x_scale = x_max / NORM_MAX
    y_scale = y_max / NORM_MAX
    xy_scale = 0.5 * (x_scale + y_scale)
    margin_norm = float(mission['safety_margin_m']) / xy_scale
    cfg_search = protocol.make_cfg(overrides={'HMIN': 0.0, 'HMAX': NORM_MAX, 'AGL_MIN': 0.0, 'AGL_MAX': NORM_MAX, 'SAFETY_MARGIN': margin_norm, 'FORBID_OVERFLY': True, 'OVERFLY_MARGIN': margin_norm})
    start_norm = _normalized_xy_agl(terrain=terrain, points=start, x_max_m=x_max, y_max_m=y_max, agl_min=agl_min, agl_max=agl_max)
    goal_norm = _normalized_xy_agl(terrain=terrain, points=goal, x_max_m=x_max, y_max_m=y_max, agl_min=agl_min, agl_max=agl_max)
    obstacle_height = hmax + 1000.0
    obstacles_physical = tuple(((float(item['center_xy_m'][0]), float(item['center_xy_m'][1]), float(item['radius_m']), obstacle_height, 'cylinder') for item in mission['threats']))
    obstacles_normalized = tuple(((ob[0] / x_max * NORM_MAX, ob[1] / y_max * NORM_MAX, ob[2] / xy_scale, NORM_MAX + 1000.0, 'cylinder') for ob in obstacles_physical))
    x_axis = np.linspace(0.0, x_max, int(grid['width_cols']), dtype=np.float64)
    y_axis = np.linspace(0.0, y_max, int(grid['height_rows']), dtype=np.float64)
    X, Y = np.meshgrid(x_axis, y_axis)
    return ExternalScene(key=key, scene_id=str(data[key]['scene_id']), terrain=terrain, x_max_m=x_max, y_max_m=y_max, hmin=hmin, hmax=hmax, start_physical=start.astype(np.float64), goal_physical=goal.astype(np.float64), start_normalized=start_norm.astype(np.float64), goal_normalized=goal_norm.astype(np.float64), obstacles_physical=obstacles_physical, obstacles_normalized=obstacles_normalized, cfg_physical=cfg_physical, cfg_search=cfg_search, X=X, Y=Y, native_audit_terrain=_load_native_audit_terrain(key, data))

def _terrain_height_search(scene: ExternalScene, x: Any, y: Any) -> Array:
    xs, ys = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    return np.zeros(xs.shape, dtype=np.float64)

def _decode_solution(cost: ExternalDemCost, vector: Array) -> tuple[Array, int]:
    path, info = cost.decode_path(vector)
    return (path, int(info.get('K_eff', info.get('k_eff', max(path.shape[0] - 2, 0)))))

def _path_is_feasible_for_cost(cost: ExternalDemCost, path: Array, tol: float=1e-08) -> bool:
    physical = cost.path_to_physical(path)
    eval_ctx = make_external_eval_context(cost.scene, cost)
    return path_is_feasible_1m(physical, eval_ctx, tol=tol)

def _path_cost_for_search(cost: ExternalDemCost, path: Array, _ctx: Any=None) -> float:
    out = cost.evaluate_path(path, return_detail=False)
    return float(out[0]) if isinstance(out, tuple) else float(out)

def _multi_objective_for_search(cost: ExternalDemCost, path: Array, _ctx: Any=None) -> Array:
    physical = cost.path_to_physical(path)
    fit = _path_cost_for_search(cost, path)
    length = float(np.sum(np.linalg.norm(physical[1:] - physical[:-1], axis=1))) if physical.shape[0] >= 2 else fit
    return np.array([fit, length], dtype=np.float64)

def _base_search_context(scene: ExternalScene, cost: ExternalDemCost, *, kind: str, budget: Any) -> dict[str, Any]:
    if kind == 'adaptive':
        n_group = int(protocol.N_GROUP)
        n_dims = int(protocol.N_GROUP * protocol.K_CAP)
        st_x = ((0.0, NORM_MAX), (0.0, NORM_MAX), (0.0, NORM_MAX), tuple(protocol.G_BOUNDS))
        st_v = protocol.ST_V
        extra = {'K_CAP': int(protocol.K_CAP), 'tau': float(scene.cfg_search['TAU']), 'gate_cfg': cost.gate_cfg, 'g_bounds': tuple(protocol.G_BOUNDS)}
    else:
        n_group = int(protocol.BASELINE_N_GROUP)
        n_waypoints = int(scene.cfg_search.get('BASELINE_WAYPOINTS', protocol.BASELINE_WAYPOINTS))
        n_dims = int(n_group * n_waypoints)
        st_x = ((0.0, NORM_MAX), (0.0, NORM_MAX), (0.0, NORM_MAX))
        st_v = protocol.BASELINE_ST_V
        extra = {'N_WAYPOINTS': n_waypoints, 'K_FIXED': n_waypoints, 'NMOPSO_N_NODES': 10, 'SPSO_N_NODES': n_waypoints, 'xyz_lo': np.array([0.0, 0.0, 0.0], dtype=np.float64), 'xyz_hi': np.array([NORM_MAX, NORM_MAX, NORM_MAX], dtype=np.float64)}
    ctx: dict[str, Any] = {'X': scene.X / np.array([[scene.x_max_m / NORM_MAX]]), 'Y': scene.Y / np.array([[scene.y_max_m / NORM_MAX]]), 'Z': np.zeros_like(scene.X, dtype=np.float64), 'START': scene.start_normalized.copy(), 'GOAL': scene.goal_normalized.copy(), 'height_fn': lambda x, y, _scene=scene: _terrain_height_search(_scene, x, y), 'obstacles': list(scene.obstacles_normalized), 'physical_obstacles': list(scene.obstacles_physical), 'environment_source': 'environment.external_dem', 'scenario': scene.key, 'external_scene_id': scene.scene_id, 'cost': cost, 'DIM': int(protocol.DIM), 'N_GROUP': n_group, 'N_DIMS': n_dims, 'st_x': st_x, 'st_v': st_v, 'encoding': f'external_{kind}_agl_normalized', 'baseline_encoding': f'external_{kind}_agl_normalized', 'n_particles': int(budget.n_particles), 'num_iter': int(budget.max_iter), 'eval_budget': int(budget.eval_budget), 'pso_params': {'n_particles': int(budget.n_particles), 'w': 0.58, 'c1': 2.0, 'c2': 2.0, 'num_iter': int(budget.max_iter)}, 'cfg': copy.deepcopy(scene.cfg_search), 'physical_cfg': copy.deepcopy(scene.cfg_physical), 'BASE_SEED': int(protocol.BASE_SEED), 'DEFAULT_SNAPSHOT_ITERS': tuple(protocol.DEFAULT_SNAPSHOT_ITERS), 'external_coordinate_system': 'normalized_search_physical_evaluation', 'external_height_encoding': 'agl_normalized', 'external_decode_policy': 'x/y normalized to metres; z normalized to AGL then added to the scene-specific DEM height', 'external_physical_hmin_hmax': (float(scene.hmin), float(scene.hmax)), 'external_terrain_path': str(scene.terrain.source_path), 'external_terrain_resolution_m': float(scene.terrain.resolution_m)}
    ctx.update(extra)
    ctx['decode_solution'] = lambda vector, _cost=cost: _decode_solution(_cost, vector)
    ctx['set_cost_iteration'] = lambda iteration, max_iter=None, _cost=cost: _cost.set_iteration(iteration, max_iter)
    ctx['path_is_feasible'] = lambda path, tol=1e-08, _cost=cost: _path_is_feasible_for_cost(_cost, path, tol)
    ctx['path_cost'] = lambda path, _ctx=None, _cost=cost: _path_cost_for_search(_cost, path, _ctx)
    ctx['nmopso_path_cost'] = ctx['path_cost']
    ctx['multi_objective_path_cost'] = lambda path, _ctx=None, _cost=cost: _multi_objective_for_search(_cost, path, _ctx)
    ctx['nmopso_objectives'] = ctx['multi_objective_path_cost']
    return ctx

def prepare_external_adapter_context(ctx: dict[str, Any], algorithm: str) -> dict[str, Any]:
    out = dict(ctx)
    out['external_post_decode_path_repair_enabled'] = False
    out['external_post_decode_path_repair_policy'] = EXTERNAL_POST_DECODE_PATH_REPAIR_POLICY
    out['external_adapter_algorithm'] = str(algorithm)
    return out

def make_external_eval_context(scene: ExternalScene, cost: ExternalDemCost | None=None, *, budget: Any | None=None) -> dict[str, Any]:
    eval_cost = cost or ExternalDemCost(scene=scene, mode='fixed', sort_waypoints=True)
    ctx: dict[str, Any] = {'X': scene.X.copy(), 'Y': scene.Y.copy(), 'Z': np.zeros_like(scene.X, dtype=np.float64), 'START': scene.start_physical.copy(), 'GOAL': scene.goal_physical.copy(), 'height_fn': scene.terrain.height_physical, 'obstacles': list(scene.obstacles_physical), 'environment_source': 'environment.external_dem', 'scenario': scene.key, 'external_scene_id': scene.scene_id, 'cost': eval_cost, 'DIM': int(protocol.DIM), 'N_GROUP': int(protocol.BASELINE_N_GROUP), 'N_DIMS': int(protocol.BASELINE_N_DIMS), 'st_x': ((0.0, NORM_MAX), (0.0, NORM_MAX), (0.0, NORM_MAX)), 'st_v': protocol.BASELINE_ST_V, 'encoding': 'external_physical_eval', 'cfg': copy.deepcopy(scene.cfg_physical), 'HMIN': float(scene.hmin), 'HMAX': float(scene.hmax), 'AGL_MIN': float(scene.cfg_physical['AGL_MIN']), 'AGL_MAX': float(scene.cfg_physical['AGL_MAX']), 'SAFETY_MARGIN': float(scene.cfg_physical['SAFETY_MARGIN']), 'external_height_encoding': 'physical_m', 'external_eval_terrain_path': str(scene.terrain.source_path), 'external_eval_terrain_resolution_m': float(scene.terrain.resolution_m), 'external_final_audit_spacing_m': 1.0}
    if scene.key == 'msh_ext' and float(scene.terrain.resolution_m) <= 1.0:
        ctx['external_robust_audit_label'] = 'msh_0p5m_supplemental_audit'
        ctx['external_robust_audit_spacing_m'] = 0.5
    if budget is not None and hasattr(budget, 'eval_budget'):
        ctx['eval_budget'] = int(budget.eval_budget)
        ctx['n_particles'] = int(getattr(budget, 'n_particles', 0) or 0)
        ctx['num_iter'] = int(getattr(budget, 'max_iter', 0) or 0)
    if scene.native_audit_terrain is not None:
        native = scene.native_audit_terrain
        offset = 0.5 * (float(scene.terrain.resolution_m) - float(native.resolution_m))
        ctx['native_audit_label'] = 'msh_1m_native_window'
        ctx['native_audit_height_fn'] = lambda x, y, _native=native, _offset=offset: _native.height_physical(np.asarray(x, dtype=np.float64) + _offset, np.asarray(y, dtype=np.float64) + _offset)
        ctx['native_audit_source_path'] = str(native.source_path)
    ctx['path_is_feasible'] = lambda path, tol=1e-08, _ctx=ctx: path_is_feasible_1m(path, _ctx, tol=tol)
    return ctx

def build_external_context_bundle(scene: str, *, budget: Any) -> dict[str, Any]:
    scene_obj = load_external_scene(scene)
    gate_cfg = protocol.make_gate_cfg(scene_obj.cfg_search)
    adaptive_cost = ExternalDemCost(scene=scene_obj, mode='adaptive', gate_cfg=gate_cfg)
    fixed_cost = ExternalDemCost(scene=scene_obj, mode='fixed', sort_waypoints=True)
    return {'scene': scene_obj, 'adaptive': _base_search_context(scene_obj, adaptive_cost, kind='adaptive', budget=budget), 'fixed': _base_search_context(scene_obj, fixed_cost, kind='fixed', budget=budget), 'eval': make_external_eval_context(scene_obj, fixed_cost, budget=budget)}

def physicalize_result_path(result: Mapping[str, Any], search_ctx: Mapping[str, Any], algorithm: str) -> dict[str, Any]:
    out = dict(result)
    if 'path' not in out:
        return out
    meta = dict(out.get('meta', {}))
    meta.setdefault('external_search_encoding', str(search_ctx.get('encoding', '')))
    meta.setdefault('encoding', str(search_ctx.get('encoding', '')))
    out['meta'] = meta
    path = np.asarray(out['path'], dtype=np.float64)
    cost = search_ctx.get('cost')
    path_to_physical = getattr(cost, 'path_to_physical', None)
    if not callable(path_to_physical):
        return out
    if str(algorithm) in {'SPSO', 'NMOPSO'} or _looks_normalized_path(path):
        physical = path_to_physical(path)
        out['path'] = physical
        meta = dict(out.get('meta', {}))
        meta['external_original_path_space'] = 'normalized_search'
        meta['external_result_path_space'] = 'physical_m'
        out['meta'] = meta
    return out

def scene_audit_payload(scene: ExternalScene) -> dict[str, Any]:
    valid_count, invalid_count, valid_min, valid_max = scene.terrain.logical_stats()
    support_valid_count = int(np.count_nonzero(scene.terrain.mask))
    return {'scene': scene.scene_id, 'terrain_path': str(scene.terrain.source_path), 'terrain_sha256': sha256_file(scene.terrain.source_path), 'terrain_shape_rows_cols': [scene.terrain.height, scene.terrain.width], 'logical_bounds_m': {'x_min': float(scene.terrain.x_min_m), 'x_max': float(scene.terrain.x_max_m), 'y_min': float(scene.terrain.y_min_m), 'y_max': float(scene.terrain.y_max_m)}, 'valid_count': valid_count, 'invalid_count': invalid_count, 'support_valid_count': support_valid_count, 'support_invalid_count': int(scene.terrain.mask.size - support_valid_count), 'valid_min': valid_min, 'valid_max': valid_max, 'hmin': float(scene.hmin), 'hmax': float(scene.hmax), 'start_physical': scene.start_physical.tolist(), 'goal_physical': scene.goal_physical.tolist(), 'start_normalized': scene.start_normalized.tolist(), 'goal_normalized': scene.goal_normalized.tolist(), 'normalized_height_encoding': 'agl_relative_to_scene_dem', 'terrain_resolution_m': float(scene.terrain.resolution_m), 'x_max_m': float(scene.x_max_m), 'y_max_m': float(scene.y_max_m), 'main_terrain_role': 'scene_specific_optimization_and_evaluation_dem', 'obstacle_count': int(len(scene.obstacles_physical))}

def array_sha256(array: Any) -> str:
    arr = np.ascontiguousarray(np.asarray(array))
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()

def final_path_audit_record(path: Array, eval_ctx: Mapping[str, Any]) -> dict[str, Any]:
    spacing = float(eval_ctx.get('external_final_audit_spacing_m', 1.0))
    audit = audit_physical_path(path, eval_ctx, max_spacing_m=spacing)
    record = {'external_final_audit_spacing_m': float(spacing), 'external_audit_passed_int': int(bool(audit['passed'])), 'external_audit_errors': ','.join(audit['errors']), 'external_sample_count_1m': int(audit['sample_count_1m']), 'external_invalid_dem_count': int(audit['invalid_dem_count']), 'external_obstacle_violation_count': int(audit['obstacle_violation_count']), 'external_min_clearance_1m': float(audit['min_clearance_1m']), 'external_min_agl_1m': float(audit['min_agl_1m']), 'external_max_agl_1m': float(audit['max_agl_1m'])}
    native_height_fn = eval_ctx.get('native_audit_height_fn', None)
    if callable(native_height_fn):
        native_ctx = dict(eval_ctx)
        native_ctx['height_fn'] = native_height_fn
        native_audit = audit_physical_path(path, native_ctx, max_spacing_m=1.0)
        record.update({'external_native_audit_label': str(eval_ctx.get('native_audit_label', '')), 'external_native_audit_passed_int': int(bool(native_audit['passed'])), 'external_native_audit_errors': ','.join(native_audit['errors']), 'external_native_invalid_dem_count': int(native_audit['invalid_dem_count']), 'external_native_obstacle_violation_count': int(native_audit['obstacle_violation_count']), 'external_native_min_clearance_1m': float(native_audit['min_clearance_1m']), 'external_native_min_agl_1m': float(native_audit['min_agl_1m']), 'external_native_max_agl_1m': float(native_audit['max_agl_1m'])})
    else:
        record.update({'external_native_audit_label': '', 'external_native_audit_passed_int': '', 'external_native_audit_errors': '', 'external_native_invalid_dem_count': '', 'external_native_obstacle_violation_count': '', 'external_native_min_clearance_1m': '', 'external_native_min_agl_1m': '', 'external_native_max_agl_1m': ''})
    robust_spacing = eval_ctx.get('external_robust_audit_spacing_m', None)
    if robust_spacing is not None:
        robust_audit = audit_physical_path(path, eval_ctx, max_spacing_m=float(robust_spacing))
        record.update({'external_robust_audit_label': str(eval_ctx.get('external_robust_audit_label', '')), 'external_robust_audit_spacing_m': float(robust_spacing), 'external_robust_audit_passed_int': int(bool(robust_audit['passed'])), 'external_robust_audit_errors': ','.join(robust_audit['errors']), 'external_robust_sample_count': int(robust_audit['sample_count_1m']), 'external_robust_invalid_dem_count': int(robust_audit['invalid_dem_count']), 'external_robust_obstacle_violation_count': int(robust_audit['obstacle_violation_count']), 'external_robust_min_clearance_m': float(robust_audit['min_clearance_1m']), 'external_robust_min_agl_m': float(robust_audit['min_agl_1m']), 'external_robust_max_agl_m': float(robust_audit['max_agl_1m'])})
    else:
        record.update({'external_robust_audit_label': '', 'external_robust_audit_spacing_m': '', 'external_robust_audit_passed_int': '', 'external_robust_audit_errors': '', 'external_robust_sample_count': '', 'external_robust_invalid_dem_count': '', 'external_robust_obstacle_violation_count': '', 'external_robust_min_clearance_m': '', 'external_robust_min_agl_m': '', 'external_robust_max_agl_m': ''})
    return record
