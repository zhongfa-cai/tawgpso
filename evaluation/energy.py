"""Path length, turn, pitch, altitude-change, flight-time proxy, and energy-index metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

@dataclass
class EnergyConfig:
    k_dist: float = 1.0
    k_turn: float = 8.0
    k_climb: float = 2.5
    k_pitch: float = 5.0
    k_descent: float = 0.0
    eps: float = 1e-12

def _to_path_array(path: Any, strict: bool=False) -> np.ndarray:
    try:
        arr = np.asarray(path, dtype=float)
    except Exception as exc:
        if strict:
            raise ValueError('path cannot be converted to a NumPy array.') from exc
        return np.empty((0, 3), dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        if strict:
            raise ValueError('path must have shape (N, 3) or (N, >=3).')
        return np.empty((0, 3), dtype=float)
    arr = arr[:, :3]
    if not np.all(np.isfinite(arr)):
        if strict:
            raise ValueError('path contains NaN or Inf.')
        return np.empty((0, 3), dtype=float)
    return arr

def _segment_vectors(path_xyz: np.ndarray) -> np.ndarray:
    if path_xyz.shape[0] < 2:
        return np.empty((0, 3), dtype=float)
    return np.diff(path_xyz, axis=0)

def _segment_lengths(seg_vecs: np.ndarray) -> np.ndarray:
    if seg_vecs.size == 0:
        return np.empty((0,), dtype=float)
    return np.linalg.norm(seg_vecs, axis=1)

def _safe_angle_between(v1: np.ndarray, v2: np.ndarray, eps: float=1e-12) -> float:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < eps or n2 < eps:
        return 0.0
    cos_val = float(np.dot(v1, v2) / (n1 * n2))
    cos_val = float(np.clip(cos_val, -1.0, 1.0))
    return float(np.arccos(cos_val))

def compute_path_length(path: Any, strict: bool=False) -> float:
    path_xyz = _to_path_array(path, strict=strict)
    if path_xyz.shape[0] < 2:
        return float('nan') if not strict else 0.0
    seg_vecs = _segment_vectors(path_xyz)
    seg_lens = _segment_lengths(seg_vecs)
    return float(np.sum(seg_lens))

def compute_turn_angles(path: Any, strict: bool=False) -> np.ndarray:
    path_xyz = _to_path_array(path, strict=strict)
    seg_vecs = _segment_vectors(path_xyz)
    if seg_vecs.shape[0] < 2:
        return np.empty((0,), dtype=float)
    angles = []
    for i in range(1, seg_vecs.shape[0]):
        angle = _safe_angle_between(seg_vecs[i - 1], seg_vecs[i])
        angles.append(angle)
    return np.asarray(angles, dtype=float)

def compute_pitch_angles(path: Any, strict: bool=False) -> np.ndarray:
    path_xyz = _to_path_array(path, strict=strict)
    if path_xyz.shape[0] < 2:
        return np.empty((0,), dtype=float)
    seg_vecs = _segment_vectors(path_xyz)
    dx = seg_vecs[:, 0]
    dy = seg_vecs[:, 1]
    dz = seg_vecs[:, 2]
    horizontal = np.sqrt(dx ** 2 + dy ** 2)
    pitch_angles = np.arctan2(dz, horizontal)
    return np.asarray(pitch_angles, dtype=float)

def compute_height_change(path: Any, strict: bool=False) -> Dict[str, float]:
    path_xyz = _to_path_array(path, strict=strict)
    if path_xyz.shape[0] < 2:
        return {'total_climb': float('nan'), 'total_descent': float('nan'), 'vertical_variation': float('nan'), 'net_height_change': float('nan')}
    dz = np.diff(path_xyz[:, 2])
    total_climb = float(np.sum(np.maximum(dz, 0.0)))
    total_descent = float(np.sum(np.maximum(-dz, 0.0)))
    vertical_variation = float(np.sum(np.abs(dz)))
    net_height_change = float(path_xyz[-1, 2] - path_xyz[0, 2])
    return {'total_climb': total_climb, 'total_descent': total_descent, 'vertical_variation': vertical_variation, 'net_height_change': net_height_change}

def compute_energy_index(path: Any, k_eff: Optional[int]=None, config: Optional[EnergyConfig]=None, strict: bool=False) -> Dict[str, float]:
    cfg = config if config is not None else EnergyConfig()
    path_xyz = _to_path_array(path, strict=strict)
    if path_xyz.shape[0] < 2:
        return {'valid': False, 'energy_total': float('nan'), 'energy_dist': float('nan'), 'energy_turn': float('nan'), 'energy_climb': float('nan'), 'energy_descent': float('nan'), 'energy_pitch': float('nan'), 'energy_waypoint': float('nan'), 'path_length': float('nan'), 'total_climb': float('nan'), 'total_descent': float('nan'), 'vertical_variation': float('nan'), 'turn_angle_sum_rad': float('nan'), 'turn_angle_sum_deg': float('nan'), 'pitch_change_sum_rad': float('nan'), 'pitch_change_sum_deg': float('nan'), 'k_eff': float('nan'), 'num_points': float(path_xyz.shape[0]), 'num_segments': float('nan')}
    n_points = int(path_xyz.shape[0])
    n_segments = max(n_points - 1, 0)
    if k_eff is None:
        k_eff_val = max(n_points - 2, 0)
    else:
        k_eff_val = max(int(k_eff), 0)
    seg_vecs = _segment_vectors(path_xyz)
    seg_lens = _segment_lengths(seg_vecs)
    path_length = float(np.sum(seg_lens))
    height_info = compute_height_change(path_xyz, strict=strict)
    total_climb = float(height_info['total_climb'])
    total_descent = float(height_info['total_descent'])
    vertical_variation = float(height_info['vertical_variation'])
    net_height_change = float(height_info['net_height_change'])
    turn_angles = compute_turn_angles(path_xyz, strict=strict)
    turn_angle_sum_rad = float(np.sum(np.abs(turn_angles)))
    turn_angle_sum_deg = float(np.degrees(turn_angle_sum_rad))
    pitch_angles = compute_pitch_angles(path_xyz, strict=strict)
    if pitch_angles.shape[0] >= 2:
        pitch_change = np.diff(pitch_angles)
        pitch_change_sum_rad = float(np.sum(np.abs(pitch_change)))
    else:
        pitch_change_sum_rad = 0.0
    pitch_change_sum_deg = float(np.degrees(pitch_change_sum_rad))
    energy_dist = float(cfg.k_dist * path_length)
    energy_turn = float(cfg.k_turn * turn_angle_sum_rad)
    energy_climb = float(cfg.k_climb * total_climb)
    energy_descent = float(cfg.k_descent * total_descent)
    energy_pitch = float(cfg.k_pitch * pitch_change_sum_rad)
    energy_waypoint = 0.0
    energy_total = float(energy_dist + energy_turn + energy_climb + energy_descent + energy_pitch)
    return {'valid': True, 'energy_total': energy_total, 'energy_dist': energy_dist, 'energy_turn': energy_turn, 'energy_climb': energy_climb, 'energy_descent': energy_descent, 'energy_pitch': energy_pitch, 'energy_waypoint': energy_waypoint, 'path_length': path_length, 'turn_angle_sum_rad': turn_angle_sum_rad, 'turn_angle_sum_deg': turn_angle_sum_deg, 'pitch_change_sum_rad': pitch_change_sum_rad, 'pitch_change_sum_deg': pitch_change_sum_deg, 'total_climb': total_climb, 'total_descent': total_descent, 'vertical_variation': vertical_variation, 'net_height_change': net_height_change, 'k_eff': float(k_eff_val), 'num_points': float(n_points), 'num_segments': float(n_segments)}

def energy_consumption_index(path: Any, k_eff: Optional[int]=None, config: Optional[EnergyConfig]=None, strict: bool=False) -> float:
    info = compute_energy_index(path=path, k_eff=k_eff, config=config, strict=strict)
    return float(info['energy_total'])

def summarize_energy_values(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {'energy_mean': float('nan'), 'energy_std': float('nan'), 'energy_min': float('nan'), 'energy_max': float('nan'), 'energy_median': float('nan'), 'energy_count': 0.0}
    return {'energy_mean': float(np.mean(arr)), 'energy_std': float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0, 'energy_min': float(np.min(arr)), 'energy_max': float(np.max(arr)), 'energy_median': float(np.median(arr)), 'energy_count': float(arr.size)}

def batch_compute_energy_index(paths: Sequence[Any], k_eff_list: Optional[Sequence[Optional[int]]]=None, config: Optional[EnergyConfig]=None, strict: bool=False) -> Dict[str, Any]:
    records = []
    energy_values = []
    if k_eff_list is None:
        k_eff_list = [None] * len(paths)
    for path, k_eff in zip(paths, k_eff_list):
        info = compute_energy_index(path=path, k_eff=k_eff, config=config, strict=strict)
        records.append(info)
        energy_values.append(info['energy_total'])
    summary = summarize_energy_values(energy_values)
    return {'records': records, 'summary': summary}

def compute_energy_metrics(path: Any, k_eff: Optional[int]=None, config: Optional[EnergyConfig]=None, strict: bool=False) -> Dict[str, float]:
    return compute_energy_index(path=path, k_eff=k_eff, config=config, strict=strict)
if __name__ == '__main__':
    demo_path = np.array([[0.0, 0.0, 30.0], [50.0, 20.0, 45.0], [100.0, 40.0, 60.0], [150.0, 70.0, 55.0], [200.0, 100.0, 50.0]])
    info = compute_energy_index(demo_path, k_eff=3)
    print('========== Energy Metrics Demo ==========')
    for key, value in info.items():
        print(f'{key}: {value}')
