"""Complete TAWG-PSO implementation: adaptive gating, initialization, feedback, elite guidance, local perturbation, reinjection, and PSO updates."""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np

Array = np.ndarray
_GATE_SHARP = 6.0
_GATE_EPS = 1e-12
_PRUNE_EPS = 1e-09
_G_ACTIVE = 1.0
_G_INACTIVE = -1.0
_FILL_VALUE = 0.0

@dataclass(frozen=True)
class AdaptiveWaypointConfig:
    k_cap: int
    active_threshold: float
    lambda_k: float
    lambda_binary: float

    def __post_init__(self) -> None:
        if int(self.k_cap) <= 0:
            raise ValueError('k_cap must be positive.')
        if not 0.0 < float(self.active_threshold) < 1.0:
            raise ValueError('active_threshold must be in (0, 1).')
        if float(self.lambda_k) < 0.0:
            raise ValueError('lambda_k must be non-negative.')
        if float(self.lambda_binary) < 0.0:
            raise ValueError('lambda_binary must be non-negative.')

def sigmoid_stable(x: Array) -> Array:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0.0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out

def split_xyzg(xg: Array, k_cap: Optional[int]=None) -> Tuple[Array, Array]:
    xg = np.asarray(xg, dtype=float).reshape(-1)
    if k_cap is None:
        if xg.size % 4 != 0:
            raise ValueError(f'xg length must be divisible by 4, got {xg.size}.')
        k_cap = xg.size // 4
    k_cap = int(k_cap)
    expected = 4 * k_cap
    if xg.size != expected:
        raise ValueError(f'xg length mismatch: expected {expected}, got {xg.size}.')
    if not np.all(np.isfinite(xg)):
        raise ValueError('xg contains non-finite values.')
    blocks = xg.reshape(k_cap, 4)
    xyz = blocks[:, :3].copy()
    g = blocks[:, 3].copy()
    return (xyz, g)

def gate_probabilities(g: Array, cfg: AdaptiveWaypointConfig) -> Array:
    g = np.asarray(g, dtype=float).reshape(-1)
    if not np.all(np.isfinite(g)):
        raise ValueError('gate variable contains non-finite values.')
    return sigmoid_stable(_GATE_SHARP * g)

def get_active_mask(q: Array, cfg: AdaptiveWaypointConfig) -> Array:
    q = np.asarray(q, dtype=float).reshape(-1)
    return q >= float(cfg.active_threshold)

def sort_waypoints_along_line(waypoints: Array, start: Array, goal: Array) -> Array:
    waypoints = np.asarray(waypoints, dtype=float)
    if waypoints.size == 0:
        return waypoints.reshape(0, 3)
    if waypoints.ndim != 2 or waypoints.shape[1] != 3:
        raise ValueError('waypoints must have shape (n, 3).')
    start = np.asarray(start, dtype=float).reshape(3)
    goal = np.asarray(goal, dtype=float).reshape(3)
    axis = goal - start
    norm = float(np.linalg.norm(axis))
    if norm < _GATE_EPS:
        return waypoints.copy()
    unit = axis / norm
    projection = (waypoints - start) @ unit
    order = np.argsort(projection)
    return waypoints[order].copy()

def decode_adaptive_waypoints(xg: Array, start: Array, goal: Array, cfg: AdaptiveWaypointConfig) -> Tuple[Array, Dict[str, Any]]:
    start = np.asarray(start, dtype=float).reshape(3)
    goal = np.asarray(goal, dtype=float).reshape(3)
    xyz, g = split_xyzg(xg, cfg.k_cap)
    q = gate_probabilities(g, cfg)
    mask = get_active_mask(q, cfg)
    active_xyz = xyz[mask]
    active_xyz = sort_waypoints_along_line(active_xyz, start, goal)
    k_eff = int(np.count_nonzero(mask))
    k_soft = float(np.sum(q))
    binary_term = float(np.sum(q * (1.0 - q)))
    gate_info: Dict[str, Any] = {'q': q, 'g': g, 'mask': mask, 'k_eff': k_eff, 'k_soft': k_soft, 'binary_term': binary_term}
    return (active_xyz, gate_info)

def build_path_from_xg(xg: Array, start: Array, goal: Array, cfg: AdaptiveWaypointConfig) -> Tuple[Array, Dict[str, Any]]:
    start_arr = np.asarray(start, dtype=float).reshape(1, 3)
    goal_arr = np.asarray(goal, dtype=float).reshape(1, 3)
    active_xyz, gate_info = decode_adaptive_waypoints(xg=xg, start=start_arr.reshape(3), goal=goal_arr.reshape(3), cfg=cfg)
    if active_xyz.size == 0:
        active_xyz = active_xyz.reshape(0, 3)
    path = np.vstack([start_arr, active_xyz, goal_arr])
    return (path, gate_info)

def lambda_k_at(iteration: Optional[int], max_iter: Optional[int], cfg: AdaptiveWaypointConfig) -> float:
    lam = float(cfg.lambda_k)
    if iteration is None or max_iter is None or int(max_iter) <= 0:
        return lam
    progress = float(iteration) / float(max_iter)
    progress = float(np.clip(progress, 0.0, 1.0))
    return float(lam * progress * progress)

def adaptive_waypoint_penalty(gate_info: Dict[str, Any], cfg: AdaptiveWaypointConfig, iteration: Optional[int]=None, max_iter: Optional[int]=None) -> Tuple[float, Dict[str, float]]:
    if 'q' not in gate_info:
        raise KeyError("gate_info must contain key 'q'.")
    q = np.asarray(gate_info['q'], dtype=float).reshape(-1)
    if q.size == 0:
        return (0.0, {'lambda_k': 0.0, 'k_eff': 0.0, 'k_soft': 0.0, 'binary_term': 0.0, 'penalty_k': 0.0, 'penalty_binary': 0.0, 'penalty_total': 0.0})
    k_soft = float(np.sum(q))
    binary_term = float(np.sum(q * (1.0 - q)))
    lam_k = lambda_k_at(iteration, max_iter, cfg)
    penalty_k = float(lam_k * k_soft)
    penalty_binary = float(cfg.lambda_binary * binary_term)
    penalty_total = float(penalty_k + penalty_binary)
    detail = {'lambda_k': float(lam_k), 'k_eff': float(gate_info.get('k_eff', 0)), 'k_soft': float(k_soft), 'binary_term': float(binary_term), 'penalty_k': float(penalty_k), 'penalty_binary': float(penalty_binary), 'penalty_total': float(penalty_total)}
    return (penalty_total, detail)

def pack_xyzg_from_waypoints(waypoints: Array, cfg: AdaptiveWaypointConfig) -> Array:
    waypoints = np.asarray(waypoints, dtype=float)
    if waypoints.size == 0:
        waypoints = waypoints.reshape(0, 3)
    if waypoints.ndim != 2 or waypoints.shape[1] != 3:
        raise ValueError('waypoints must have shape (k, 3).')
    if not np.all(np.isfinite(waypoints)):
        raise ValueError('waypoints contain non-finite values.')
    k = int(waypoints.shape[0])
    if k > int(cfg.k_cap):
        raise ValueError(f'too many waypoints: got {k}, k_cap={cfg.k_cap}.')
    blocks = np.zeros((int(cfg.k_cap), 4), dtype=float)
    blocks[:, :3] = _FILL_VALUE
    blocks[:, 3] = _G_INACTIVE
    if k > 0:
        blocks[:k, :3] = waypoints
        blocks[:k, 3] = _G_ACTIVE
    return blocks.reshape(-1)

def summarize_gate_info(gate_info: Dict[str, Any]) -> Dict[str, Any]:
    q = np.asarray(gate_info.get('q', []), dtype=float).reshape(-1)
    mask = np.asarray(gate_info.get('mask', []), dtype=bool).reshape(-1)
    if q.size == 0:
        return {'k_eff': 0, 'k_soft': 0.0, 'q_min': 0.0, 'q_max': 0.0, 'q_mean': 0.0, 'binary_term': 0.0}
    return {'k_eff': int(np.count_nonzero(mask)), 'k_soft': float(np.sum(q)), 'q_min': float(np.min(q)), 'q_max': float(np.max(q)), 'q_mean': float(np.mean(q)), 'binary_term': float(np.sum(q * (1.0 - q)))}

def prune_redundant_waypoints(path: Array, evaluate_path: Callable[[Array], float], feasible_check: Callable[[Array], bool], *, cost_tol: float=1e-09, max_passes: int=3) -> Tuple[Array, float, Dict[str, Any]]:
    P = np.asarray(path, dtype=float)
    info: Dict[str, Any] = {'enabled': True, 'applied': False, 'removed': 0, 'removed_indices': [], 'initial_points': int(P.shape[0]) if P.ndim == 2 else 0, 'final_points': int(P.shape[0]) if P.ndim == 2 else 0, 'initial_k_eff': int(max(P.shape[0] - 2, 0)) if P.ndim == 2 else 0, 'final_k_eff': int(max(P.shape[0] - 2, 0)) if P.ndim == 2 else 0, 'initial_cost': float('nan'), 'final_cost': float('nan'), 'skipped_reason': ''}
    if P.ndim != 2 or P.shape[1] != 3 or P.shape[0] <= 2:
        info['skipped_reason'] = 'invalid_or_no_midpoints'
        return (P, float('nan'), info)
    try:
        cur_cost = float(evaluate_path(P))
    except Exception as exc:
        info['skipped_reason'] = f'evaluate_failed: {exc}'
        return (P, float('nan'), info)
    info['initial_cost'] = float(cur_cost)
    try:
        if not bool(feasible_check(P)):
            info['skipped_reason'] = 'path_not_feasible'
            info['final_cost'] = float(cur_cost)
            return (P, float(cur_cost), info)
    except Exception as exc:
        info['skipped_reason'] = f'feasible_check_failed: {exc}'
        info['final_cost'] = float(cur_cost)
        return (P, float(cur_cost), info)
    removed_total = 0
    removed_indices: list[int] = []
    for _pass in range(max(int(max_passes), 1)):
        changed = False
        i = 1
        while i < P.shape[0] - 1:
            candidate = np.delete(P, i, axis=0)
            try:
                if not bool(feasible_check(candidate)):
                    i += 1
                    continue
                cand_cost = float(evaluate_path(candidate))
            except Exception:
                i += 1
                continue
            if cand_cost <= cur_cost + float(cost_tol):
                P = candidate
                cur_cost = float(cand_cost)
                removed_total += 1
                removed_indices.append(int(i))
                changed = True
                continue
            i += 1
        if not changed:
            break
    info['applied'] = bool(removed_total > 0)
    info['removed'] = int(removed_total)
    info['removed_indices'] = removed_indices
    info['final_points'] = int(P.shape[0])
    info['final_k_eff'] = int(max(P.shape[0] - 2, 0))
    info['final_cost'] = float(cur_cost)
    return (P, float(cur_cost), info)
__all__ = ['AdaptiveWaypointConfig', 'sigmoid_stable', 'split_xyzg', 'gate_probabilities', 'get_active_mask', 'sort_waypoints_along_line', 'decode_adaptive_waypoints', 'build_path_from_xg', 'lambda_k_at', 'adaptive_waypoint_penalty', 'pack_xyzg_from_waypoints', 'summarize_gate_info', 'prune_redundant_waypoints']
Array = np.ndarray
TAWG_PSO_PARAMETER_SET = 'C7-W4'
TAWG_PSO_C_MAX = 2.5
TAWG_PSO_C_MIN = 0.5
TAWG_PSO_W_MAX = 0.9
TAWG_PSO_W_MIN = 0.4
_PARAMETER_LOCK_ATOL = 1e-12
_SIGMOID_K = 8.0
_STAG_WINDOW = 25
_IMPROVE_TOL = 1e-10
_FEEDBACK_GAIN = 0.18
_LOW_DIVERSITY = 0.06
_EPS = 1e-12

@dataclass(frozen=True)
class DynamicPSOConfig:
    w_high: float = TAWG_PSO_W_MAX
    w_low: float = TAWG_PSO_W_MIN
    c_high: float = TAWG_PSO_C_MAX
    c_low: float = TAWG_PSO_C_MIN

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.w_low) < float(self.w_high):
            raise ValueError('Require 0 <= w_low < w_high.')
        if not 0.0 <= float(self.c_low) < float(self.c_high):
            raise ValueError('Require 0 <= c_low < c_high.')

def get_tawgpso_dynamic_config() -> DynamicPSOConfig:
    return DynamicPSOConfig(w_high=TAWG_PSO_W_MAX, w_low=TAWG_PSO_W_MIN, c_high=TAWG_PSO_C_MAX, c_low=TAWG_PSO_C_MIN)

def require_tawgpso_fixed_config(cfg: Optional[DynamicPSOConfig]=None) -> DynamicPSOConfig:
    resolved = get_tawgpso_dynamic_config() if cfg is None else cfg
    if not isinstance(resolved, DynamicPSOConfig):
        raise TypeError('TAWG-PSO dynamic_config must be a DynamicPSOConfig returned by get_tawgpso_dynamic_config().')
    expected = get_tawgpso_dynamic_config()
    mismatches = []
    for field in ('c_high', 'c_low', 'w_high', 'w_low'):
        actual_value = float(getattr(resolved, field))
        expected_value = float(getattr(expected, field))
        if not math.isclose(actual_value, expected_value, rel_tol=0.0, abs_tol=_PARAMETER_LOCK_ATOL):
            mismatches.append(f'{field}={actual_value:g} (expected {expected_value:g})')
    if mismatches:
        raise ValueError(f'TAWG-PSO parameters are locked to {TAWG_PSO_PARAMETER_SET}: Cmax={TAWG_PSO_C_MAX:g}, Cmin={TAWG_PSO_C_MIN:g}, wmax={TAWG_PSO_W_MAX:g}, wmin={TAWG_PSO_W_MIN:g}. Rejected override(s): ' + '; '.join(mismatches))
    return resolved

def iteration_progress(t: int, T: int) -> float:
    T = max(int(T), 1)
    t = min(max(int(t), 1), T)
    if T <= 1:
        return 1.0
    return float((t - 1) / (T - 1))

def sigmoid_progress(t: int, T: int, k: float=_SIGMOID_K) -> float:
    p = iteration_progress(t, T)
    z = float(k) * (p - 0.5)
    if z >= 0.0:
        return float(1.0 / (1.0 + math.exp(-z)))
    ez = math.exp(z)
    return float(ez / (1.0 + ez))

def _cosine_decay(t: int, T: int) -> float:
    p = iteration_progress(t, T)
    return float(0.5 * (1.0 + math.cos(math.pi * p)))

def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), float(lo)), float(hi)))

def population_diversity(population: Optional[Array], bounds: Optional[Array]=None) -> Optional[float]:
    if population is None:
        return None
    X = np.asarray(population, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1:
        return None
    if not np.all(np.isfinite(X)):
        return None
    std = np.std(X, axis=0)
    if bounds is not None:
        B = np.asarray(bounds, dtype=float)
        if B.shape != (X.shape[1], 2):
            return None
        scale = np.maximum(B[:, 1] - B[:, 0], _EPS)
    else:
        scale = np.maximum(np.max(X, axis=0) - np.min(X, axis=0), _EPS)
    div = float(np.mean(std / scale))
    if not np.isfinite(div):
        return None
    return max(0.0, div)

def _feedback_score(stagnation_ratio: float=0.0, diversity: Optional[float]=None) -> float:
    stag = _clip(stagnation_ratio, 0.0, 1.0)
    if stag <= 0.0:
        return 0.0
    if diversity is None:
        low_div = 0.0
    else:
        low_div = _clip((_LOW_DIVERSITY - float(diversity)) / max(_LOW_DIVERSITY, _EPS), 0.0, 1.0)
    return _clip(stag * (0.7 + 0.3 * low_div), 0.0, 1.0)

def parameters_at(t: int, T: int, cfg: Optional[DynamicPSOConfig]=None, *, stagnation_ratio: float=0.0, diversity: Optional[float]=None) -> Tuple[float, float, float, Dict[str, float]]:
    cfg = cfg or get_tawgpso_dynamic_config()
    p = iteration_progress(t, T)
    s = sigmoid_progress(t, T)
    cd = _cosine_decay(t, T)
    w_span = float(cfg.w_high - cfg.w_low)
    c_span = float(cfg.c_high - cfg.c_low)
    w = float(cfg.w_low + w_span * cd)
    c1 = float(cfg.c_high - c_span * s)
    c2 = float(cfg.c_low + c_span * s)
    fb = _feedback_score(stagnation_ratio=stagnation_ratio, diversity=diversity)
    if fb > 0.0:
        w += _FEEDBACK_GAIN * fb * w_span
        c1 += 0.5 * _FEEDBACK_GAIN * fb * c_span
        c2 -= 0.5 * _FEEDBACK_GAIN * fb * c_span
    w = _clip(w, cfg.w_low, cfg.w_high)
    c1 = _clip(c1, cfg.c_low, cfg.c_high)
    c2 = _clip(c2, cfg.c_low, cfg.c_high)
    info = {'progress': float(p), 'stage': float(s), 'cosine_decay': float(cd), 'feedback': float(fb), 'diversity': float(diversity) if diversity is not None else -1.0, 'stagnation_ratio': float(_clip(stagnation_ratio, 0.0, 1.0)), 'w': float(w), 'c1': float(c1), 'c2': float(c2)}
    return (float(w), float(c1), float(c2), info)

def w_c1_c2_at(t: int, T: int, cfg: Optional[DynamicPSOConfig]=None) -> Tuple[float, float, float]:
    w, c1, c2, _ = parameters_at(t, T, cfg)
    return (w, c1, c2)

def c1c2_at(t: int, T: int, c1_bounds: Tuple[float, float]=(TAWG_PSO_C_MIN, TAWG_PSO_C_MAX), c2_bounds: Tuple[float, float]=(TAWG_PSO_C_MIN, TAWG_PSO_C_MAX), k: float=_SIGMOID_K) -> Tuple[float, float]:
    s = sigmoid_progress(t, T, k)
    c1_low, c1_high = (min(c1_bounds), max(c1_bounds))
    c2_low, c2_high = (min(c2_bounds), max(c2_bounds))
    c1 = c1_high - (c1_high - c1_low) * s
    c2 = c2_low + (c2_high - c2_low) * s
    return (float(c1), float(c2))

def generate_schedule(T: int, cfg: Optional[DynamicPSOConfig]=None, *, include_feedback: bool=False) -> Tuple[Array, Array, Array, Array]:
    cfg = cfg or get_tawgpso_dynamic_config()
    T = max(int(T), 1)
    t_arr = np.arange(1, T + 1, dtype=int)
    w_arr = np.empty(T, dtype=float)
    c1_arr = np.empty(T, dtype=float)
    c2_arr = np.empty(T, dtype=float)
    for i, ti in enumerate(t_arr):
        if include_feedback:
            stagnation_ratio = min(1.0, i / max(_STAG_WINDOW, 1))
            w_arr[i], c1_arr[i], c2_arr[i], _ = parameters_at(int(ti), T, cfg, stagnation_ratio=stagnation_ratio, diversity=0.0)
        else:
            w_arr[i], c1_arr[i], c2_arr[i] = w_c1_c2_at(int(ti), T, cfg)
    return (t_arr, w_arr, c1_arr, c2_arr)

class DynamicParameterScheduler:

    def __init__(self, max_iter: int, cfg: Optional[DynamicPSOConfig]=None, bounds: Optional[Array]=None) -> None:
        self.max_iter = max(int(max_iter), 1)
        self.cfg = cfg or get_tawgpso_dynamic_config()
        self.bounds = None if bounds is None else np.asarray(bounds, dtype=float)
        self.best_seen = math.inf
        self.stagnant_iters = 0
        self.last_info: Dict[str, float] = {}

    def reset(self) -> None:
        self.best_seen = math.inf
        self.stagnant_iters = 0
        self.last_info = {}

    def _update_stagnation(self, best_fit: Optional[float]) -> float:
        if best_fit is None or not np.isfinite(best_fit):
            return 0.0
        best_fit = float(best_fit)
        if best_fit < self.best_seen - _IMPROVE_TOL:
            self.best_seen = best_fit
            self.stagnant_iters = 0
        else:
            self.stagnant_iters += 1
        return _clip(self.stagnant_iters / max(_STAG_WINDOW, 1), 0.0, 1.0)

    def step(self, t: int, *, population: Optional[Array]=None, best_fit: Optional[float]=None) -> Tuple[float, float, float, Dict[str, float]]:
        stag_ratio = self._update_stagnation(best_fit)
        div = population_diversity(population, self.bounds)
        w, c1, c2, info = parameters_at(t=t, T=self.max_iter, cfg=self.cfg, stagnation_ratio=stag_ratio, diversity=div)
        info['stagnant_iters'] = float(self.stagnant_iters)
        info['best_seen'] = float(self.best_seen) if np.isfinite(self.best_seen) else math.inf
        self.last_info = info
        return (w, c1, c2, info)
__all__ = ['DynamicPSOConfig', 'DynamicParameterScheduler', 'TAWG_PSO_PARAMETER_SET', 'TAWG_PSO_C_MAX', 'TAWG_PSO_C_MIN', 'TAWG_PSO_W_MAX', 'TAWG_PSO_W_MIN', 'get_tawgpso_dynamic_config', 'require_tawgpso_fixed_config', 'iteration_progress', 'sigmoid_progress', 'population_diversity', 'parameters_at', 'w_c1_c2_at', 'c1c2_at', 'generate_schedule']
Array = np.ndarray
_MIN_ELITES = 3
_RANK_DECAY = 3.0
_STAG_WINDOW = 25
_IMPROVE_TOL = 1e-10
_RANDOM_TRIGGER = 0.7
_RANDOM_PROB = 0.15
_EPS = 1e-12

@dataclass(frozen=True)
class EliteGuidanceConfig:
    elite_ratio: float = 0.1
    warmup_ratio: float = 0.15
    base_mix: float = 0.2
    max_mix: float = 0.35

    def __post_init__(self) -> None:
        if not 0.0 < float(self.elite_ratio) <= 0.5:
            raise ValueError('elite_ratio must be in (0, 0.5].')
        if not 0.0 <= float(self.warmup_ratio) < 1.0:
            raise ValueError('warmup_ratio must be in [0, 1).')
        if not 0.0 <= float(self.base_mix) <= float(self.max_mix) < 1.0:
            raise ValueError('Require 0 <= base_mix <= max_mix < 1.')

def _clip01(x: float) -> float:
    return float(min(max(float(x), 0.0), 1.0))

def _elite_iteration_progress(iteration: Optional[int], max_iter: Optional[int]) -> float:
    if iteration is None or max_iter is None:
        return 1.0
    T = max(int(max_iter), 1)
    t = min(max(int(iteration), 1), T)
    if T <= 1:
        return 1.0
    return float((t - 1) / (T - 1))

def fusion_strength(iteration: Optional[int], max_iter: Optional[int], stagnation_ratio: float, cfg: EliteGuidanceConfig) -> float:
    progress = _elite_iteration_progress(iteration, max_iter)
    if progress < float(cfg.warmup_ratio):
        return 0.0
    stag = _clip01(stagnation_ratio)
    lam = float(cfg.base_mix) + (float(cfg.max_mix) - float(cfg.base_mix)) * stag
    return float(min(max(lam, 0.0), float(cfg.max_mix)))

def _as_2d_population(X: Array) -> Array:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError('X_P must be a 2-D array with shape (n_particles, n_dims).')
    if X.shape[0] <= 0 or X.shape[1] <= 0:
        raise ValueError('X_P must not be empty.')
    return X

def _as_1d_cost(J: Array, n: int) -> Array:
    J = np.asarray(J, dtype=float).reshape(-1)
    if J.size != int(n):
        raise ValueError(f'J_P length mismatch: expected {n}, got {J.size}.')
    return J

def _valid_solution_mask(X: Array, J: Array, feasible_mask: Optional[Array]=None) -> Array:
    mask = np.isfinite(J) & np.all(np.isfinite(X), axis=1)
    if feasible_mask is not None:
        fm = np.asarray(feasible_mask, dtype=bool).reshape(-1)
        if fm.size != J.size:
            raise ValueError('feasible_mask length must match J_P.')
        mask = mask & fm
    return mask

def _rank_weights(m: int) -> Array:
    m = int(m)
    if m <= 0:
        return np.zeros(0, dtype=float)
    if m == 1:
        return np.ones(1, dtype=float)
    rank_scaled = np.arange(m, dtype=float) / float(m - 1)
    logits = -_RANK_DECAY * rank_scaled
    logits -= float(np.max(logits))
    e = np.exp(logits)
    s = float(np.sum(e))
    if s <= _EPS or not np.isfinite(s):
        return np.ones(m, dtype=float) / float(m)
    return e / s

def _weighted_center(elites: Array, weights: Array) -> Array:
    elites = np.asarray(elites, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if elites.ndim != 2:
        raise ValueError('elites must be a 2-D array.')
    if elites.shape[0] != weights.size:
        raise ValueError('weights length must match number of elites.')
    return np.sum(weights[:, None] * elites, axis=0)

def _medoid_index(elites: Array, center: Array) -> int:
    elites = np.asarray(elites, dtype=float)
    center = np.asarray(center, dtype=float).reshape(-1)
    if elites.ndim != 2:
        raise ValueError('elites must be a 2-D array.')
    if elites.shape[1] != center.size:
        raise ValueError('center dimension mismatch.')
    scale = np.std(elites, axis=0)
    scale = np.where(scale > _EPS, scale, 1.0)
    diff = (elites - center[None, :]) / scale[None, :]
    dist = np.linalg.norm(diff, axis=1)
    return int(np.argmin(dist))

def _choose_reference_index(elites: Array, weights: Array, medoid_idx: int, stagnation_ratio: float, rng: Optional[np.random.Generator]) -> Tuple[int, bool]:
    m = int(elites.shape[0])
    stag = _clip01(stagnation_ratio)
    if rng is None or m <= 1:
        return (int(medoid_idx), False)
    if stag < _RANDOM_TRIGGER:
        return (int(medoid_idx), False)
    prob = float(_RANDOM_PROB * stag)
    if float(rng.random()) < prob:
        idx = int(rng.choice(np.arange(m), p=weights))
        return (idx, True)
    return (int(medoid_idx), False)

def _fallback_reference(X: Array, J: Array, gbest: Optional[Array]) -> Array:
    if gbest is not None:
        g = np.asarray(gbest, dtype=float).reshape(-1)
        if np.all(np.isfinite(g)):
            return g.copy()
    finite = np.isfinite(J) & np.all(np.isfinite(X), axis=1)
    if np.any(finite):
        idx = int(np.argmin(J[finite]))
        Xf = X[finite]
        return Xf[idx].copy()
    return np.mean(X, axis=0).copy()

def compute_elite_guidance(X_P: Array, J_P: Array, *, gbest: Optional[Array]=None, iteration: Optional[int]=None, max_iter: Optional[int]=None, feasible_mask: Optional[Array]=None, stagnation_ratio: float=0.0, rng: Optional[np.random.Generator]=None, cfg: Optional[EliteGuidanceConfig]=None) -> Tuple[Array, Dict[str, Any]]:
    cfg = cfg or EliteGuidanceConfig()
    X = _as_2d_population(X_P)
    J = _as_1d_cost(J_P, X.shape[0])
    progress = _elite_iteration_progress(iteration, max_iter)
    lam = fusion_strength(iteration, max_iter, stagnation_ratio, cfg)
    if lam <= 0.0:
        ref = _fallback_reference(X, J, gbest)
        return (ref, {'enabled': False, 'reason': 'warmup_or_zero_lambda', 'progress': float(progress), 'lambda': 0.0, 'n_valid': 0, 'top_k': 0, 'selected_rank': -1, 'random_selected': False})
    valid_mask = _valid_solution_mask(X, J, feasible_mask)
    n_valid = int(np.count_nonzero(valid_mask))
    if n_valid <= 0:
        ref = _fallback_reference(X, J, gbest)
        return (ref, {'enabled': False, 'reason': 'no_valid_elite', 'progress': float(progress), 'lambda': 0.0, 'n_valid': 0, 'top_k': 0, 'selected_rank': -1, 'random_selected': False})
    X_valid = X[valid_mask]
    J_valid = J[valid_mask]
    order = np.argsort(J_valid)
    X_sorted = X_valid[order]
    J_sorted = J_valid[order]
    if n_valid < _MIN_ELITES and gbest is not None:
        ref = _fallback_reference(X, J, gbest)
        return (ref, {'enabled': False, 'reason': 'too_few_elites', 'progress': float(progress), 'lambda': 0.0, 'n_valid': int(n_valid), 'top_k': int(n_valid), 'selected_rank': -1, 'random_selected': False})
    top_k = int(math.ceil(float(cfg.elite_ratio) * float(n_valid)))
    top_k = max(1, top_k)
    top_k = min(int(n_valid), top_k)
    if n_valid >= _MIN_ELITES:
        top_k = max(_MIN_ELITES, top_k)
        top_k = min(top_k, n_valid)
    elites = X_sorted[:top_k]
    elite_costs = J_sorted[:top_k]
    weights = _rank_weights(top_k)
    center = _weighted_center(elites, weights)
    medoid_idx = _medoid_index(elites, center)
    selected_idx, random_selected = _choose_reference_index(elites=elites, weights=weights, medoid_idx=medoid_idx, stagnation_ratio=stagnation_ratio, rng=rng)
    elite_ref = elites[selected_idx].copy()
    if gbest is not None:
        g = np.asarray(gbest, dtype=float).reshape(-1)
        if g.size != elite_ref.size:
            raise ValueError('gbest dimension mismatch.')
        if not np.all(np.isfinite(g)):
            G_ref = elite_ref
            effective_lambda = 1.0
        else:
            G_ref = (1.0 - lam) * g + lam * elite_ref
            effective_lambda = float(lam)
    else:
        G_ref = elite_ref
        effective_lambda = 1.0
    info = {'enabled': True, 'reason': 'ok', 'progress': float(progress), 'lambda': float(effective_lambda), 'n_valid': int(n_valid), 'top_k': int(top_k), 'selected_rank': int(selected_idx), 'random_selected': bool(random_selected), 'medoid_rank': int(medoid_idx), 'best_elite_cost': float(elite_costs[0]), 'selected_elite_cost': float(elite_costs[selected_idx]), 'stagnation_ratio': float(_clip01(stagnation_ratio)), 'weights_max': float(np.max(weights)) if weights.size else 0.0, 'weights_min': float(np.min(weights)) if weights.size else 0.0}
    return (np.asarray(G_ref, dtype=float).copy(), info)

class EliteGuidanceController:

    def __init__(self, max_iter: int, cfg: Optional[EliteGuidanceConfig]=None, rng: Optional[np.random.Generator]=None) -> None:
        self.max_iter = max(int(max_iter), 1)
        self.cfg = cfg or EliteGuidanceConfig()
        self.rng = rng or np.random.default_rng()
        self.best_seen = float('inf')
        self.stagnant_iters = 0
        self.last_info: Dict[str, Any] = {}

    def reset(self) -> None:
        self.best_seen = float('inf')
        self.stagnant_iters = 0
        self.last_info = {}

    def _update_stagnation(self, best_fit: Optional[float]) -> float:
        if best_fit is None:
            return 0.0
        try:
            y = float(best_fit)
        except Exception:
            return 0.0
        if not np.isfinite(y):
            return 0.0
        if y < self.best_seen - _IMPROVE_TOL:
            self.best_seen = y
            self.stagnant_iters = 0
        else:
            self.stagnant_iters += 1
        return _clip01(self.stagnant_iters / float(max(_STAG_WINDOW, 1)))

    def step(self, *, iteration: int, X_P: Array, J_P: Array, gbest: Optional[Array], best_fit: Optional[float]=None, feasible_mask: Optional[Array]=None) -> Tuple[Array, Dict[str, Any]]:
        stag = self._update_stagnation(best_fit)
        G_ref, info = compute_elite_guidance(X_P=X_P, J_P=J_P, gbest=gbest, iteration=iteration, max_iter=self.max_iter, feasible_mask=feasible_mask, stagnation_ratio=stag, rng=self.rng, cfg=self.cfg)
        info['stagnant_iters'] = int(self.stagnant_iters)
        info['best_seen'] = float(self.best_seen) if np.isfinite(self.best_seen) else None
        self.last_info = info
        return (G_ref, info)
__all__ = ['EliteGuidanceConfig', 'EliteGuidanceController', 'iteration_progress', 'fusion_strength', 'compute_elite_guidance']
Obstacle = Tuple[float, float, float, float, str]

def min_lateral_margin_xy(x: float, y: float, obstacles: Iterable[Obstacle]) -> float:
    from evaluation.objectives import min_lateral_margin_xy as implementation
    return float(implementation(x, y, obstacles))

def segment_min_lateral_margin(A: Array, B: Array, obstacles: Iterable[Obstacle], samples: int=61) -> float:
    from evaluation.objectives import segment_min_lateral_margin as implementation
    return float(implementation(A, B, obstacles, samples=samples))

def obstacle_xy_max_radius(radius: float, kind: str) -> float:
    from evaluation.objectives import obstacle_xy_max_radius as implementation
    return float(implementation(radius, kind))
Array = np.ndarray
HeightFn = Callable[[float, float], float]
CostFn = Callable[[Array], float]
WARM_RATIO = 0.25
_MAX_WARM_RATIO = 0.25
_SEED_CAP_RATIO = 0.4
_MIN_SEED_K = 2
_REPAIR_PASSES = 8
_SEG_MARGIN_SAMPLES = 61
_PUSH_BUFFER = 1.0
_ACTIVE_GATE = 1.0
_INACTIVE_GATE = -1.0
_EPS = 1e-12
_DEFAULT_CANDIDATE_COUNT = 6
_MAX_CORRIDOR_OBSTACLES = 4

@dataclass(frozen=True)
class HotStartConfig:
    warm_ratio: float = WARM_RATIO
    candidate_count: int = _DEFAULT_CANDIDATE_COUNT

    def __post_init__(self) -> None:
        if not 0.0 < float(self.warm_ratio) <= _MAX_WARM_RATIO:
            raise ValueError(f'warm_ratio must be in (0, {_MAX_WARM_RATIO}].')
        if int(self.candidate_count) <= 0:
            raise ValueError('candidate_count must be positive.')

def _hot_is_pair(x: Any) -> bool:
    return isinstance(x, (list, tuple, np.ndarray)) and len(x) == 2 and np.isscalar(x[0]) and np.isscalar(x[1])

def _is_group_bounds(bounds: Any) -> bool:
    return isinstance(bounds, (list, tuple)) and len(bounds) in (3, 4) and all((_hot_is_pair(bd) for bd in bounds))

def _bound_for_dim(st_x: Any, dim_idx: int, default: Tuple[float, float]) -> Tuple[float, float]:
    if _hot_is_pair(st_x):
        return (float(st_x[0]), float(st_x[1]))
    if _is_group_bounds(st_x):
        bd = st_x[dim_idx]
        return (float(bd[0]), float(bd[1]))
    return (float(default[0]), float(default[1]))

def clip_by_bounds(vec: Array, bounds: Any) -> Array:
    if bounds is None:
        return vec
    vec = np.asarray(vec, dtype=float)
    if _hot_is_pair(bounds):
        np.clip(vec, float(bounds[0]), float(bounds[1]), out=vec)
        return vec
    if _is_group_bounds(bounds):
        stride = len(bounds)
        for d, bd in enumerate(bounds):
            vec[d::stride] = np.clip(vec[d::stride], float(bd[0]), float(bd[1]))
        return vec
    if isinstance(bounds, (list, tuple)) and len(bounds) == vec.size and all((_hot_is_pair(bd) for bd in bounds)):
        for i, bd in enumerate(bounds):
            vec[i] = np.clip(vec[i], float(bd[0]), float(bd[1]))
        return vec
    return vec

def _cfg(ctx: Dict[str, Any]) -> Dict[str, Any]:
    c = ctx.get('cfg', ctx.get('CFG', {}))
    return c if isinstance(c, dict) else {}

def _get_required(ctx: Dict[str, Any], key: str) -> Any:
    if key not in ctx:
        raise KeyError(f"ctx must contain key '{key}'.")
    return ctx[key]

def _get_height_limits(ctx: Dict[str, Any]) -> Tuple[float, float, float, Optional[float]]:
    c = _cfg(ctx)
    hmin = float(c.get('HMIN', ctx.get('HMIN', 0.0)))
    hmax = float(c.get('HMAX', ctx.get('HMAX', 180.0)))
    agl_min = float(c.get('AGL_MIN', ctx.get('AGL_MIN', 5.0)))
    agl_max = c.get('AGL_MAX', ctx.get('AGL_MAX', None))
    agl_max = None if agl_max is None else float(agl_max)
    return (hmin, hmax, agl_min, agl_max)

def _safety_margin(ctx: Dict[str, Any]) -> float:
    c = _cfg(ctx)
    vals = [c.get('SAFETY_MARGIN', ctx.get('SAFETY_MARGIN', None)), c.get('F1_SEG_MARGIN', ctx.get('F1_SEG_MARGIN', None)), c.get('OBS_LATERAL_CLEAR', ctx.get('OBS_LATERAL_CLEAR', None)), c.get('NODE_CLEAR_HARD', ctx.get('NODE_CLEAR_HARD', None)), c.get('SEG_CLEAR_HARD', ctx.get('SEG_CLEAR_HARD', None))]
    nums = [float(v) for v in vals if v is not None]
    return float(max(nums) if nums else 4.0)

def _get_gate_cfg(ctx: Dict[str, Any]) -> AdaptiveWaypointConfig:
    for key in ('gate_cfg', 'adaptive_cfg', 'waypoint_cfg'):
        cfg = ctx.get(key, None)
        if isinstance(cfg, AdaptiveWaypointConfig):
            return cfg
    if 'K_CAP' in ctx:
        k_cap = int(ctx['K_CAP'])
    elif 'K_MAX' in ctx:
        k_cap = int(ctx['K_MAX'])
    elif 'N_DIMS' in ctx:
        k_cap = int(ctx['N_DIMS']) // 4
    else:
        raise KeyError('hot-start context must define K_CAP, K_MAX, or N_DIMS.')
    cfg = _cfg(ctx)
    required = ('TAU', 'LAMBDA_K_MAX', 'LAMBDA_B')
    missing = [key for key in required if key not in cfg]
    if missing:
        raise KeyError('hot-start context is missing gate configuration fields: ' + ', '.join(missing))
    return AdaptiveWaypointConfig(k_cap=max(1, int(k_cap)), active_threshold=float(cfg['TAU']), lambda_k=float(cfg['LAMBDA_K_MAX']), lambda_binary=float(cfg['LAMBDA_B']))

def _seed_waypoint_count(gate_cfg: AdaptiveWaypointConfig) -> int:
    k_cap = int(gate_cfg.k_cap)
    return int(max(1, min(k_cap, max(_MIN_SEED_K, round(_SEED_CAP_RATIO * k_cap)))))

def _line_normal_xy(start: Array, goal: Array) -> Array:
    s = np.asarray(start, dtype=float).reshape(3)
    g = np.asarray(goal, dtype=float).reshape(3)
    d = g - s
    n = np.array([-d[1], d[0]], dtype=float)
    norm = float(np.linalg.norm(n))
    if norm < _EPS:
        n = np.array([0.0, 1.0], dtype=float)
        norm = 1.0
    return n / norm

def _nearest_obstacle_center(x: float, y: float, obstacles: Iterable[Obstacle]) -> Tuple[Optional[Tuple[float, float]], float]:
    best_center: Optional[Tuple[float, float]] = None
    best_margin = float('inf')
    for cx, cy, r, _h, kind in obstacles:
        rmax = obstacle_xy_max_radius(float(r), str(kind))
        margin = float(np.hypot(float(x) - float(cx), float(y) - float(cy)) - rmax)
        if margin < best_margin:
            best_margin = margin
            best_center = (float(cx), float(cy))
    return (best_center, float(best_margin))

def _push_direction(point_xy: Array, center: Optional[Tuple[float, float]], fallback: Array, rng: np.random.Generator) -> Array:
    p = np.asarray(point_xy, dtype=float).reshape(2)
    if center is None:
        v = np.asarray(fallback, dtype=float).reshape(2).copy()
    else:
        v = p - np.asarray(center, dtype=float).reshape(2)
    norm = float(np.linalg.norm(v))
    if norm < _EPS:
        v = rng.normal(0.0, 1.0, size=2)
        norm = float(np.linalg.norm(v))
    if norm < _EPS:
        v = np.asarray(fallback, dtype=float).reshape(2).copy()
        norm = float(np.linalg.norm(v))
    return v / max(norm, _EPS)

def _clip_path_xyz(path: Array, ctx: Dict[str, Any]) -> Array:
    if path.shape[0] <= 2:
        return path
    st_x = ctx.get('st_x', None)
    hmin, hmax, _agl_min, _agl_max = _get_height_limits(ctx)
    x_lo, x_hi = _bound_for_dim(st_x, 0, (0.0, 100.0))
    y_lo, y_hi = _bound_for_dim(st_x, 1, (0.0, 100.0))
    z_lo, z_hi = _bound_for_dim(st_x, 2, (hmin, hmax))
    path[1:-1, 0] = np.clip(path[1:-1, 0], x_lo, x_hi)
    path[1:-1, 1] = np.clip(path[1:-1, 1], y_lo, y_hi)
    path[1:-1, 2] = np.clip(path[1:-1, 2], max(z_lo, hmin), min(z_hi, hmax))
    return path

def _enforce_height(path: Array, height_fn: HeightFn, ctx: Dict[str, Any]) -> Array:
    if path.shape[0] <= 2:
        return path
    hmin, hmax, agl_min, agl_max = _get_height_limits(ctx)
    for i in range(1, path.shape[0] - 1):
        ground = float(height_fn(float(path[i, 0]), float(path[i, 1])))
        lo = max(float(hmin), ground + float(agl_min))
        hi = float(hmax)
        if agl_max is not None:
            hi = min(hi, ground + float(agl_max))
        if hi < lo:
            hi = lo
        path[i, 2] = float(np.clip(path[i, 2], lo, hi))
    return path

def _repair_nodes(path: Array, obstacles: Iterable[Obstacle], ctx: Dict[str, Any], rng: np.random.Generator) -> Array:
    obs = list(obstacles)
    if not obs or path.shape[0] <= 2:
        return path
    margin_need = _safety_margin(ctx)
    fallback = _line_normal_xy(path[0], path[-1])
    for i in range(1, path.shape[0] - 1):
        center, margin = _nearest_obstacle_center(float(path[i, 0]), float(path[i, 1]), obs)
        if margin < margin_need:
            direction = _push_direction(path[i, :2], center, fallback, rng)
            path[i, :2] += direction * (margin_need - margin + _PUSH_BUFFER)
    return path

def _repair_segments(path: Array, obstacles: Iterable[Obstacle], ctx: Dict[str, Any], rng: np.random.Generator) -> Array:
    obs = list(obstacles)
    if not obs or path.shape[0] <= 2:
        return path
    margin_need = _safety_margin(ctx)
    fallback = _line_normal_xy(path[0], path[-1])
    for _ in range(_REPAIR_PASSES):
        changed = False
        for j in range(path.shape[0] - 1):
            margin = float(segment_min_lateral_margin(path[j], path[j + 1], obs, samples=_SEG_MARGIN_SAMPLES))
            if margin >= margin_need:
                continue
            mid = 0.5 * (path[j] + path[j + 1])
            center, _ = _nearest_obstacle_center(float(mid[0]), float(mid[1]), obs)
            direction = _push_direction(mid[:2], center, fallback, rng)
            gap = margin_need - margin + _PUSH_BUFFER
            movable = []
            if 0 < j < path.shape[0] - 1:
                movable.append(j)
            if 0 < j + 1 < path.shape[0] - 1:
                movable.append(j + 1)
            if not movable:
                continue
            step = gap / float(len(movable))
            for idx in movable:
                path[idx, :2] += direction * step
            changed = True
        if not changed:
            break
    return path

def build_seed_waypoints(*, start: Array, goal: Array, obstacles: Iterable[Obstacle], height_fn: HeightFn, ctx: Dict[str, Any], gate_cfg: Optional[AdaptiveWaypointConfig]=None, rng: Optional[np.random.Generator]=None) -> Array:
    gate_cfg = gate_cfg or _get_gate_cfg(ctx)
    rng = rng or np.random.default_rng()
    start = np.asarray(start, dtype=float).reshape(3)
    goal = np.asarray(goal, dtype=float).reshape(3)
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(goal)):
        raise ValueError('start/goal contains non-finite values.')
    k_seed = _seed_waypoint_count(gate_cfg)
    path = np.linspace(start, goal, k_seed + 2)
    path = _clip_path_xyz(path, ctx)
    path = _enforce_height(path, height_fn, ctx)
    path = _repair_nodes(path, obstacles, ctx, rng)
    path = _clip_path_xyz(path, ctx)
    path = _enforce_height(path, height_fn, ctx)
    path = _repair_segments(path, obstacles, ctx, rng)
    path = _clip_path_xyz(path, ctx)
    path = _enforce_height(path, height_fn, ctx)
    return path[1:-1].copy()

def _segment_projection_t(start: Array, goal: Array, point_xy: Array) -> float:
    s = np.asarray(start, dtype=float).reshape(3)[:2]
    g = np.asarray(goal, dtype=float).reshape(3)[:2]
    p = np.asarray(point_xy, dtype=float).reshape(2)
    d = g - s
    denom = float(np.dot(d, d))
    if denom < _EPS:
        return 0.0
    return float(np.clip(np.dot(p - s, d) / denom, 0.0, 1.0))

def _line_unit_and_normal(start: Array, goal: Array) -> Tuple[Array, Array]:
    s = np.asarray(start, dtype=float).reshape(3)[:2]
    g = np.asarray(goal, dtype=float).reshape(3)[:2]
    d = g - s
    n = float(np.linalg.norm(d))
    if n < _EPS:
        unit = np.array([1.0, 0.0], dtype=float)
    else:
        unit = d / n
    normal = np.array([-unit[1], unit[0]], dtype=float)
    return (unit, normal)

def _key_obstacles_for_corridors(start: Array, goal: Array, obstacles: Iterable[Obstacle], ctx: Dict[str, Any]) -> list[Obstacle]:
    obs = list(obstacles)
    if not obs:
        return []
    s = np.asarray(start, dtype=float).reshape(3)[:2]
    g = np.asarray(goal, dtype=float).reshape(3)[:2]
    d = g - s
    denom = float(np.dot(d, d))
    margin = _safety_margin(ctx) + _PUSH_BUFFER
    ranked: list[tuple[float, float, Obstacle]] = []
    for ob in obs:
        cx, cy, r, _h, kind = ob
        center = np.array([float(cx), float(cy)], dtype=float)
        if denom < _EPS:
            t = 0.0
            closest = s
        else:
            t = float(np.clip(np.dot(center - s, d) / denom, 0.0, 1.0))
            closest = s + t * d
        lateral = float(np.linalg.norm(center - closest))
        safe_radius = obstacle_xy_max_radius(float(r), str(kind)) + margin
        if lateral <= safe_radius + margin:
            ranked.append((float(t), float(lateral - safe_radius), ob))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in ranked[:_MAX_CORRIDOR_OBSTACLES]]

def _resample_midpoints_from_full_path(path: Array, k_seed: int) -> Array:
    P = np.asarray(path, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3 or P.shape[0] < 2:
        return np.zeros((0, 3), dtype=float)
    k_seed = int(max(k_seed, 0))
    if k_seed <= 0:
        return np.zeros((0, 3), dtype=float)
    seg = np.linalg.norm(P[1:] - P[:-1], axis=1)
    dist = np.r_[0.0, np.cumsum(seg)]
    total = float(dist[-1])
    if total < _EPS:
        return np.repeat(P[:1], k_seed, axis=0)
    targets = np.linspace(0.0, total, k_seed + 2)[1:-1]
    out = np.empty((k_seed, 3), dtype=float)
    for i, target in enumerate(targets):
        j = int(np.searchsorted(dist, target, side='right') - 1)
        j = int(np.clip(j, 0, P.shape[0] - 2))
        span = max(float(dist[j + 1] - dist[j]), _EPS)
        alpha = float((target - dist[j]) / span)
        out[i] = P[j] + alpha * (P[j + 1] - P[j])
    return out

def _corridor_seed_waypoints(*, start: Array, goal: Array, obstacles: Iterable[Obstacle], height_fn: HeightFn, ctx: Dict[str, Any], gate_cfg: AdaptiveWaypointConfig, sign_pattern: Iterable[float], altitude_bias: float, rng: np.random.Generator) -> Optional[Array]:
    key_obs = _key_obstacles_for_corridors(start, goal, obstacles, ctx)
    if not key_obs:
        return None
    start = np.asarray(start, dtype=float).reshape(3)
    goal = np.asarray(goal, dtype=float).reshape(3)
    unit, normal = _line_unit_and_normal(start, goal)
    margin = _safety_margin(ctx) + _PUSH_BUFFER
    signs = list(sign_pattern)
    if not signs:
        signs = [1.0]
    nodes = [start.copy()]
    line_len = max(float(np.linalg.norm(goal[:2] - start[:2])), 1.0)
    for i, ob in enumerate(key_obs):
        cx, cy, r, _h, kind = ob
        center = np.array([float(cx), float(cy)], dtype=float)
        radius = obstacle_xy_max_radius(float(r), str(kind)) + margin
        t = _segment_projection_t(start, goal, center)
        base_xy = start[:2] + t * (goal[:2] - start[:2])
        side = float(signs[i % len(signs)])
        along = unit * min(radius * 0.35, 0.08 * line_len)
        detour_xy = base_xy + side * normal * radius + along
        z_line = start[2] + t * (goal[2] - start[2])
        detour = np.array([detour_xy[0], detour_xy[1], z_line + float(altitude_bias)], dtype=float)
        nodes.append(detour)
    nodes.append(goal.copy())
    path = np.asarray(nodes, dtype=float)
    path = _clip_path_xyz(path, ctx)
    path = _enforce_height(path, height_fn, ctx)
    path = _repair_nodes(path, obstacles, ctx, rng)
    path = _clip_path_xyz(path, ctx)
    path = _enforce_height(path, height_fn, ctx)
    path = _repair_segments(path, obstacles, ctx, rng)
    path = _clip_path_xyz(path, ctx)
    path = _enforce_height(path, height_fn, ctx)
    k_seed = _seed_waypoint_count(gate_cfg)
    mid = _resample_midpoints_from_full_path(path, k_seed)
    if mid.size == 0:
        return None
    return mid

def _candidate_signature(xg: Array) -> tuple:
    arr = np.asarray(xg, dtype=float).reshape(-1)
    return tuple(np.round(arr, 3).tolist())

def build_hot_start_xg_candidates(ctx: Dict[str, Any], seed: int, config: Optional[HotStartConfig]=None, gate_cfg: Optional[AdaptiveWaypointConfig]=None, rng: Optional[np.random.Generator]=None) -> list[Array]:
    config = config or HotStartConfig()
    gate_cfg = gate_cfg or _get_gate_cfg(ctx)
    rng = rng or np.random.default_rng(int(seed))
    start = np.asarray(_get_required(ctx, 'START'), dtype=float).reshape(3)
    goal = np.asarray(_get_required(ctx, 'GOAL'), dtype=float).reshape(3)
    obstacles = list(_get_required(ctx, 'obstacles'))
    height_fn = _get_required(ctx, 'height_fn')
    if not callable(height_fn):
        raise TypeError("ctx['height_fn'] must be callable.")
    hmin, hmax, _agl_min, _agl_max = _get_height_limits(ctx)
    z_span = max(float(hmax - hmin), 1.0)
    specs: list[tuple[tuple[float, ...], float]] = [((1.0,), 0.0), ((-1.0,), 0.0), ((1.0, -1.0), 0.04 * z_span), ((-1.0, 1.0), 0.04 * z_span), ((1.0,), 0.08 * z_span), ((-1.0,), 0.08 * z_span)]
    waypoints: list[Array] = [build_seed_waypoints(start=start, goal=goal, obstacles=obstacles, height_fn=height_fn, ctx=ctx, gate_cfg=gate_cfg, rng=rng)]
    for signs, z_bias in specs:
        cand = _corridor_seed_waypoints(start=start, goal=goal, obstacles=obstacles, height_fn=height_fn, ctx=ctx, gate_cfg=gate_cfg, sign_pattern=signs, altitude_bias=float(z_bias), rng=rng)
        if cand is not None:
            waypoints.append(cand)
    candidates: list[Array] = []
    seen: set[tuple] = set()
    for mid in waypoints:
        try:
            xg = pack_seed_xg(mid, gate_cfg)
            clip_by_bounds(xg, ctx.get('st_x', None))
        except Exception:
            continue
        sig = _candidate_signature(xg)
        if sig in seen:
            continue
        seen.add(sig)
        candidates.append(xg.copy())
        if len(candidates) >= int(config.candidate_count):
            break
    if not candidates:
        candidates.append(build_hot_start_xg(ctx, seed=seed, config=config, gate_cfg=gate_cfg, rng=rng))
    return candidates

def _interp_along_waypoints(points: Array, n: int) -> Array:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points must have shape (k, 3).')
    n = int(n)
    if n <= 0:
        return np.zeros((0, 3), dtype=float)
    if points.shape[0] == 0:
        return np.zeros((n, 3), dtype=float)
    if points.shape[0] == 1:
        return np.repeat(points, n, axis=0)
    old_t = np.linspace(0.0, 1.0, points.shape[0])
    new_t = np.linspace(0.0, 1.0, n)
    return np.vstack([np.interp(new_t, old_t, points[:, d]) for d in range(3)]).T

def pack_seed_xg(waypoints: Array, gate_cfg: AdaptiveWaypointConfig) -> Array:
    waypoints = np.asarray(waypoints, dtype=float)
    if waypoints.size == 0:
        waypoints = waypoints.reshape(0, 3)
    if waypoints.ndim != 2 or waypoints.shape[1] != 3:
        raise ValueError('waypoints must have shape (k, 3).')
    if not np.all(np.isfinite(waypoints)):
        raise ValueError('waypoints contains non-finite values.')
    k_cap = int(gate_cfg.k_cap)
    k_seed = min(int(waypoints.shape[0]), k_cap)
    all_xyz = _interp_along_waypoints(waypoints, k_cap)
    gates = np.full(k_cap, _INACTIVE_GATE, dtype=float)
    if k_seed > 0:
        active_idx = np.linspace(0, k_cap - 1, k_seed).round().astype(int)
        active_idx = np.unique(active_idx)
        if active_idx.size < k_seed:
            pool = [i for i in range(k_cap) if i not in set(active_idx.tolist())]
            active_idx = np.asarray(sorted(active_idx.tolist() + pool[:k_seed - active_idx.size]), dtype=int)
        all_xyz[active_idx[:k_seed]] = waypoints[:k_seed]
        gates[active_idx[:k_seed]] = _ACTIVE_GATE
    blocks = np.column_stack([all_xyz, gates])
    return blocks.reshape(-1)

def build_hot_start_xg(ctx: Dict[str, Any], seed: int, config: Optional[HotStartConfig]=None, gate_cfg: Optional[AdaptiveWaypointConfig]=None, rng: Optional[np.random.Generator]=None) -> Array:
    _ = config or HotStartConfig()
    gate_cfg = gate_cfg or _get_gate_cfg(ctx)
    start = np.asarray(_get_required(ctx, 'START'), dtype=float).reshape(3)
    goal = np.asarray(_get_required(ctx, 'GOAL'), dtype=float).reshape(3)
    obstacles = list(_get_required(ctx, 'obstacles'))
    height_fn = _get_required(ctx, 'height_fn')
    if not callable(height_fn):
        raise TypeError("ctx['height_fn'] must be callable.")
    rng = rng or np.random.default_rng()
    waypoints = build_seed_waypoints(start=start, goal=goal, obstacles=obstacles, height_fn=height_fn, ctx=ctx, gate_cfg=gate_cfg, rng=rng)
    xg = pack_seed_xg(waypoints, gate_cfg)
    clip_by_bounds(xg, ctx.get('st_x', None))
    return xg

def _safe_eval(cost_fn: CostFn, x: Array) -> float:
    try:
        y = float(cost_fn(np.asarray(x, dtype=float)))
    except Exception:
        return float('inf')
    return y if np.isfinite(y) else float('inf')

def _warm_count(n_particles: int, config: HotStartConfig) -> int:
    n_particles = int(n_particles)
    if n_particles <= 0:
        return 0
    n = int(round(float(config.warm_ratio) * n_particles))
    cap = max(1, int(round(_MAX_WARM_RATIO * n_particles)))
    return int(max(1, min(n, cap, n_particles)))

def _set_array_attr(obj: Any, attr: str, value: Array) -> None:
    if not hasattr(obj, attr):
        return
    old = getattr(obj, attr)
    value = np.asarray(value, dtype=float)
    if isinstance(old, np.ndarray) and old.shape == value.shape:
        old[:] = value
    else:
        setattr(obj, attr, value.copy())

def _random_velocity_like(x: Array, ctx: Dict[str, Any], rng: np.random.Generator) -> Array:
    x = np.asarray(x, dtype=float)
    st_v = ctx.get('st_v', None)
    if _hot_is_pair(st_v):
        lo, hi = (float(st_v[0]), float(st_v[1]))
        return rng.uniform(lo, hi, size=x.shape)
    if _is_group_bounds(st_v):
        v = np.zeros_like(x, dtype=float)
        stride = len(st_v)
        for d, bd in enumerate(st_v):
            lo, hi = (float(bd[0]), float(bd[1]))
            v[d::stride] = rng.uniform(lo, hi, size=v[d::stride].shape)
        return v
    return rng.normal(0.0, 0.01, size=x.shape)

def _jitter_xg(seed_xg: Array, ctx: Dict[str, Any], rng: np.random.Generator) -> Array:
    x = np.asarray(seed_xg, dtype=float).copy()
    if x.size % 4 != 0:
        raise ValueError('seed_xg length must be divisible by 4.')
    st_x = ctx.get('st_x', None)
    x_lo, x_hi = _bound_for_dim(st_x, 0, (0.0, 100.0))
    y_lo, y_hi = _bound_for_dim(st_x, 1, (0.0, 100.0))
    z_lo, z_hi = _bound_for_dim(st_x, 2, (0.0, 180.0))
    g_lo, g_hi = _bound_for_dim(st_x, 3, (-4.0, 4.0)) if _is_group_bounds(st_x) and len(st_x) >= 4 else (-4.0, 4.0)
    xy_sigma = max(1e-09, 0.012 * (abs(x_hi - x_lo) + abs(y_hi - y_lo)) * 0.5)
    z_sigma = max(1e-09, 0.015 * abs(z_hi - z_lo))
    g_sigma = max(1e-09, 0.05 * abs(g_hi - g_lo))
    blocks = x.reshape(-1, 4)
    blocks[:, 0] += rng.normal(0.0, xy_sigma, size=blocks.shape[0])
    blocks[:, 1] += rng.normal(0.0, xy_sigma, size=blocks.shape[0])
    blocks[:, 2] += rng.normal(0.0, z_sigma, size=blocks.shape[0])
    blocks[:, 3] += rng.normal(0.0, g_sigma, size=blocks.shape[0])
    blocks[:, 3] = np.clip(blocks[:, 3], g_lo, g_hi)
    x = blocks.reshape(-1)
    clip_by_bounds(x, st_x)
    return x

def _particles_of(opt: Any) -> list[Any]:
    particles = getattr(opt, 'particles', None)
    if particles is None:
        raise AttributeError("optimizer must have a 'particles' attribute.")
    if len(particles) <= 0:
        raise ValueError('optimizer.particles is empty.')
    return list(particles)

def _refresh_gbest(opt: Any, best_x: Array, best_value: float) -> bool:
    if not hasattr(opt, 'gbest') or not hasattr(opt, 'gvalue'):
        return False
    current = float(getattr(opt, 'gvalue', float('inf')))
    if float(best_value) < current:
        setattr(opt, 'gbest', np.asarray(best_x, dtype=float).copy())
        setattr(opt, 'gvalue', float(best_value))
        return True
    return False

def inject_hot_start(opt: Any, seed_xg: Array, *, ctx: Dict[str, Any], cost_fn: Optional[CostFn]=None, config: Optional[HotStartConfig]=None, rng: Optional[np.random.Generator]=None) -> Dict[str, Any]:
    config = config or HotStartConfig()
    particles = _particles_of(opt)
    if cost_fn is None:
        cost_fn = getattr(opt, 'func', None)
    if cost_fn is None or not callable(cost_fn):
        raise ValueError('cost_fn must be callable or opt.func must exist.')
    rng = rng or np.random.default_rng()
    seed_xg = np.asarray(seed_xg, dtype=float).copy()
    if seed_xg.ndim != 1 or seed_xg.size % 4 != 0:
        raise ValueError('seed_xg must be a 1-D xyzg vector with length divisible by 4.')
    if not np.all(np.isfinite(seed_xg)):
        raise ValueError('seed_xg contains non-finite values.')
    clip_by_bounds(seed_xg, ctx.get('st_x', None))
    n_warm = _warm_count(len(particles), config)
    best_x = None
    best_value = float('inf')
    seed_cost = None
    for i in range(n_warm):
        x = seed_xg.copy() if i == 0 else _jitter_xg(seed_xg, ctx, rng)
        y = _safe_eval(cost_fn, x)
        particle = particles[i]
        _set_array_attr(particle, 'position', x)
        _set_array_attr(particle, 'pbest', x)
        if hasattr(particle, 'pvalue'):
            setattr(particle, 'pvalue', float(y))
        if hasattr(particle, 'velocity'):
            v = _random_velocity_like(x, ctx, rng)
            _set_array_attr(particle, 'velocity', v)
        if i == 0:
            seed_cost = float(y)
        if y < best_value:
            best_value = float(y)
            best_x = x.copy()
    updated_gbest = False
    if best_x is not None and np.isfinite(best_value):
        updated_gbest = _refresh_gbest(opt, best_x, best_value)
    return {'enabled': True, 'n_warm': int(n_warm), 'seed_cost': float(seed_cost) if seed_cost is not None else None, 'best_warm_cost': float(best_value) if np.isfinite(best_value) else None, 'updated_gbest': bool(updated_gbest), 'k_cap': int(seed_xg.size // 4)}

def inject_hot_start_candidates(opt: Any, seed_xg_list: Iterable[Array], *, ctx: Dict[str, Any], cost_fn: Optional[CostFn]=None, config: Optional[HotStartConfig]=None, rng: Optional[np.random.Generator]=None) -> Dict[str, Any]:
    config = config or HotStartConfig()
    particles = _particles_of(opt)
    if cost_fn is None:
        cost_fn = getattr(opt, 'func', None)
    if cost_fn is None or not callable(cost_fn):
        raise ValueError('cost_fn must be callable or opt.func must exist.')
    rng = rng or np.random.default_rng()
    candidates: list[Array] = []
    seen: set[tuple] = set()
    for seed_xg in seed_xg_list:
        x = np.asarray(seed_xg, dtype=float).copy()
        if x.ndim != 1 or x.size % 4 != 0 or (not np.all(np.isfinite(x))):
            continue
        clip_by_bounds(x, ctx.get('st_x', None))
        sig = _candidate_signature(x)
        if sig in seen:
            continue
        seen.add(sig)
        candidates.append(x)
    if not candidates:
        raise ValueError('seed_xg_list does not contain a valid xyzg candidate.')
    scored: list[tuple[float, Array]] = []
    for x in candidates:
        scored.append((_safe_eval(cost_fn, x), x.copy()))
    scored.sort(key=lambda item: item[0])
    candidates = [x for _cost, x in scored]
    n_warm = _warm_count(len(particles), config)
    best_x = None
    best_value = float('inf')
    seed_cost = None
    for i in range(n_warm):
        base = candidates[i % len(candidates)]
        x = base.copy() if i < len(candidates) else _jitter_xg(base, ctx, rng)
        y = _safe_eval(cost_fn, x)
        particle = particles[i]
        _set_array_attr(particle, 'position', x)
        _set_array_attr(particle, 'pbest', x)
        if hasattr(particle, 'pvalue'):
            setattr(particle, 'pvalue', float(y))
        if hasattr(particle, 'velocity'):
            v = _random_velocity_like(x, ctx, rng)
            _set_array_attr(particle, 'velocity', v)
        if i == 0:
            seed_cost = float(y)
        if y < best_value:
            best_value = float(y)
            best_x = x.copy()
    updated_gbest = False
    if best_x is not None and np.isfinite(best_value):
        updated_gbest = _refresh_gbest(opt, best_x, best_value)
    return {'enabled': True, 'n_warm': int(n_warm), 'n_candidates': int(len(candidates)), 'candidate_costs': [float(c) for c, _x in scored], 'seed_cost': float(seed_cost) if seed_cost is not None else None, 'best_warm_cost': float(best_value) if np.isfinite(best_value) else None, 'updated_gbest': bool(updated_gbest), 'k_cap': int(candidates[0].size // 4), '_candidate_xg_list': [x.copy() for x in candidates]}

def apply_hot_start(opt: Any, *, ctx: Dict[str, Any], seed: int, config: Optional[HotStartConfig]=None, gate_cfg: Optional[AdaptiveWaypointConfig]=None, cost_fn: Optional[CostFn]=None, rng: Optional[np.random.Generator]=None) -> Dict[str, Any]:
    config = config or HotStartConfig()
    gate_cfg = gate_cfg or _get_gate_cfg(ctx)
    seed_xg_list = build_hot_start_xg_candidates(ctx=ctx, seed=int(seed), config=config, gate_cfg=gate_cfg, rng=rng)
    info = inject_hot_start_candidates(opt, seed_xg_list, ctx=ctx, cost_fn=cost_fn, config=config, rng=rng)
    info['seed_xg'] = seed_xg_list[0].copy()
    return info
__all__ = ['HotStartConfig', 'clip_by_bounds', 'build_seed_waypoints', 'pack_seed_xg', 'build_hot_start_xg', 'build_hot_start_xg_candidates', 'inject_hot_start', 'inject_hot_start_candidates', 'apply_hot_start']
Array = np.ndarray
EnergyFn = Callable[[Array], float]
MAX_PENALTY = 1000000.0

def _is_number(x: Any) -> bool:
    return np.isscalar(x)

def _is_pair(x: Any) -> bool:
    if not isinstance(x, (list, tuple, np.ndarray)):
        return False
    if len(x) != 2:
        return False
    return _is_number(x[0]) and _is_number(x[1])

def _as_group_bounds(bounds: Any) -> list[tuple[float, float]] | None:
    if bounds is None:
        return None
    if _is_pair(bounds):
        return None
    if isinstance(bounds, np.ndarray):
        bounds = bounds.tolist()
    if isinstance(bounds, (list, tuple)) and len(bounds) > 0:
        if all((_is_pair(bd) for bd in bounds)):
            return [(float(bd[0]), float(bd[1])) for bd in bounds]
    return None

def _clip_by_bounds(vec: Array, bounds: Any) -> Array:
    vec = np.asarray(vec, dtype=float)
    if bounds is None:
        return vec
    if _is_pair(bounds):
        lo, hi = (float(bounds[0]), float(bounds[1]))
        np.clip(vec, lo, hi, out=vec)
        return vec
    group_bounds = _as_group_bounds(bounds)
    if group_bounds is None:
        return vec
    if len(group_bounds) == vec.size:
        for i, (lo, hi) in enumerate(group_bounds):
            vec[i] = np.clip(vec[i], lo, hi)
        return vec
    stride = len(group_bounds)
    for d, (lo, hi) in enumerate(group_bounds):
        idx = np.arange(d, vec.size, stride)
        if idx.size > 0:
            vec[idx] = np.clip(vec[idx], lo, hi)
    return vec

def _sample_by_bounds(rng: np.random.Generator, n_dims: int, bounds: Any, *, fallback_low: float=0.0, fallback_high: float=1.0) -> Array:
    n_dims = int(n_dims)
    if bounds is None:
        return rng.uniform(fallback_low, fallback_high, size=n_dims).astype(float)
    if _is_pair(bounds):
        lo, hi = (float(bounds[0]), float(bounds[1]))
        return rng.uniform(lo, hi, size=n_dims).astype(float)
    group_bounds = _as_group_bounds(bounds)
    if group_bounds is None:
        return rng.uniform(fallback_low, fallback_high, size=n_dims).astype(float)
    x = np.empty(n_dims, dtype=float)
    if len(group_bounds) == n_dims:
        for i, (lo, hi) in enumerate(group_bounds):
            x[i] = rng.uniform(lo, hi)
        return x
    stride = len(group_bounds)
    for d, (lo, hi) in enumerate(group_bounds):
        idx = np.arange(d, n_dims, stride)
        if idx.size > 0:
            x[idx] = rng.uniform(lo, hi, size=idx.size)
    return x

def _sample_velocity(rng: np.random.Generator, n_dims: int, st_v: Any, st_x: Any) -> Array:
    n_dims = int(n_dims)
    if st_v is not None:
        return _sample_by_bounds(rng, n_dims, st_v, fallback_low=-0.5, fallback_high=0.5)
    if _is_pair(st_x):
        lo, hi = (float(st_x[0]), float(st_x[1]))
        span = max(abs(hi - lo), 1.0)
        return rng.uniform(-0.1 * span, 0.1 * span, size=n_dims).astype(float)
    group_bounds = _as_group_bounds(st_x)
    if group_bounds is not None:
        v = np.empty(n_dims, dtype=float)
        stride = len(group_bounds) if len(group_bounds) != n_dims else 1
        if len(group_bounds) == n_dims:
            for i, (lo, hi) in enumerate(group_bounds):
                span = max(abs(hi - lo), 1.0)
                v[i] = rng.uniform(-0.1 * span, 0.1 * span)
        else:
            for d, (lo, hi) in enumerate(group_bounds):
                span = max(abs(hi - lo), 1.0)
                idx = np.arange(d, n_dims, stride)
                if idx.size > 0:
                    v[idx] = rng.uniform(-0.1 * span, 0.1 * span, size=idx.size)
        return v
    return rng.uniform(-0.5, 0.5, size=n_dims).astype(float)

def _expand_bounds_for_diversity(st_x: Any, n_dims: int) -> Array | None:
    group_bounds = _as_group_bounds(st_x)
    if group_bounds is None:
        if _is_pair(st_x):
            lo, hi = (float(st_x[0]), float(st_x[1]))
            return np.tile(np.asarray([[lo, hi]], dtype=float), (int(n_dims), 1))
        return None
    if len(group_bounds) == int(n_dims):
        return np.asarray(group_bounds, dtype=float)
    out = np.empty((int(n_dims), 2), dtype=float)
    stride = len(group_bounds)
    for d, bd in enumerate(group_bounds):
        out[d::stride, :] = bd
    return out

def _path_length(path: Array) -> float:
    P = np.asarray(path, dtype=float)
    if P.ndim != 2 or P.shape[1] < 3 or P.shape[0] < 2:
        return float('nan')
    return float(np.sum(np.linalg.norm(P[1:, :3] - P[:-1, :3], axis=1)))

def _evaluate_path_from_ctx(path: Array, ctx: Mapping[str, Any], default: float=float('nan')) -> float:
    eval_path_fn = getattr(ctx.get('cost', None), 'evaluate_path', None)
    if not callable(eval_path_fn):
        return float(default)
    try:
        out = eval_path_fn(np.asarray(path, dtype=float), return_detail=False)
        value = float(out[0]) if isinstance(out, tuple) else float(out)
        return value if np.isfinite(value) else float(default)
    except Exception:
        return float(default)

def _path_success_from_ctx(path: Array, ctx: Mapping[str, Any], tol: float=1e-08) -> bool:
    P = np.asarray(path, dtype=float)
    fn = ctx.get('path_is_feasible', None)
    if callable(fn):
        try:
            return bool(fn(P, tol=tol))
        except TypeError:
            try:
                return bool(fn(P))
            except Exception:
                return False
        except Exception:
            return False
    try:
        from evaluation.objectives import path_is_feasible
    except Exception:
        return False
    cfg = ctx.get('cfg', {})
    try:
        return bool(path_is_feasible(P, obstacles=ctx['obstacles'], height_fn=ctx.get('height_fn', None), hmin=float(cfg.get('HMIN', ctx.get('HMIN', 0.0))), hmax=float(cfg.get('HMAX', ctx.get('HMAX', 180.0))), agl_min=float(cfg.get('AGL_MIN', ctx.get('AGL_MIN', 5.0))), agl_max=None if cfg.get('AGL_MAX', ctx.get('AGL_MAX', 120.0)) is None else float(cfg.get('AGL_MAX', ctx.get('AGL_MAX', 120.0))), safety_margin=float(cfg.get('SAFETY_MARGIN', ctx.get('SAFETY_MARGIN', 5.0))), forbid_overfly=bool(cfg.get('FORBID_OVERFLY', ctx.get('FORBID_OVERFLY', True))), overfly_margin=float(cfg.get('OVERFLY_MARGIN', ctx.get('OVERFLY_MARGIN', 5.0))), tol=float(tol)))
    except Exception:
        return False

def _safe_float(x: Any, default: float=MAX_PENALTY) -> float:
    try:
        y = float(x)
    except Exception:
        return float(default)
    if not np.isfinite(y):
        return float(default)
    return float(y)

@dataclass
class Particle:
    position: Array
    velocity: Array
    pbest: Array
    pvalue: float
    value: float = float('inf')
    feasible: bool = False
    cv: float = float('inf')
    pbest_feasible: bool = False
    pbest_cv: float = float('inf')
    stagnant_iters: int = 0

@dataclass
class EliteRecord:
    x: Array
    value: float
    cv: float

class TAWGPSO:

    def __init__(self, *, n_dims: int, n_particles: int, st_x: Any, st_v: Any, num_iter: int, func: EnergyFn, ctx: dict[str, Any] | None=None, seed: int | None=None, rng: np.random.Generator | None=None, w: float=0.58, c1: float=2.0, c2: float=2.0, use_hot_start: bool=True, use_dynamic_parameter: bool=True, use_elite_guidance: bool=True, use_stagnation_reinjection: bool=True, max_penalty: float=MAX_PENALTY, snapshot_iters: Iterable[int] | None=None, hot_start_config: Any | None=None, dynamic_config: Any | None=None, elite_config: Any | None=None) -> None:
        self.n_dims = int(n_dims)
        self.n_particles = int(n_particles)
        self.st_x = st_x
        self.st_v = st_v
        self.num_iter = int(num_iter)
        self.func = func
        self.ctx = ctx if ctx is not None else {}
        self.seed = None if seed is None else int(seed)
        self.rng = rng if rng is not None else np.random.default_rng(self.seed)
        self.w = float(w)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.max_penalty = float(max_penalty)
        self.dynamic_config = require_tawgpso_fixed_config(dynamic_config)
        self.use_hot_start = bool(use_hot_start)
        self.use_dynamic_parameter = bool(use_dynamic_parameter)
        self.use_elite_guidance = bool(use_elite_guidance)
        self.use_stagnation_reinjection = bool(use_stagnation_reinjection)
        self.snapshot_iters = set((int(i) for i in snapshot_iters or []))
        self.particles: list[Particle] = []
        self.gbest: Array | None = None
        self.gvalue: float = float('inf')
        self.gbest_feasible: bool = False
        self.gbest_cv: float = float('inf')
        self.history: list[float] = []
        self.snapshots: dict[int, dict[str, Any]] = {}
        self.evals_used: int = 0
        self.hot_start_info: dict[str, Any] = {'enabled': False}
        self.last_dynamic_info: dict[str, Any] = {}
        self.last_elite_info: dict[str, Any] = {}
        self.reinjection_info: dict[str, Any] = {'enabled': bool(self.use_stagnation_reinjection), 'count': 0, 'last_iter': 0, 'last_replaced': 0}
        self.feasible_archive: list[EliteRecord] = []
        self.reinjection_pool: list[Array] = []
        self.global_stagnant_iters: int = 0
        self.last_reinject_iter: int = 0
        if self.n_dims <= 0:
            raise ValueError('n_dims must be positive.')
        if self.n_particles <= 0:
            raise ValueError('n_particles must be positive.')
        if self.num_iter <= 0:
            raise ValueError('num_iter must be positive.')
        if not callable(self.func):
            raise TypeError('func must be callable.')
        self._set_cost_iteration(1)
        self._initialize_swarm()
        if self.use_hot_start:
            self._apply_hot_start(config=hot_start_config)
        bounds_for_div = _expand_bounds_for_diversity(self.st_x, self.n_dims)
        if self.use_dynamic_parameter and DynamicParameterScheduler is not None:
            self.dynamic_scheduler = DynamicParameterScheduler(max_iter=self.num_iter, cfg=self.dynamic_config, bounds=bounds_for_div)
        else:
            self.dynamic_scheduler = None
        if self.use_elite_guidance and EliteGuidanceController is not None:
            cfg = elite_config
            if cfg is None and EliteGuidanceConfig is not None:
                cfg = EliteGuidanceConfig()
            self.elite_controller = EliteGuidanceController(max_iter=self.num_iter, cfg=cfg, rng=self.rng)
        else:
            self.elite_controller = None

    def _set_cost_iteration(self, iteration: int | None) -> None:
        fn = self.ctx.get('set_cost_iteration', None)
        if callable(fn):
            try:
                fn(iteration, self.num_iter)
                return
            except TypeError:
                try:
                    fn(iteration)
                    return
                except Exception:
                    pass
            except Exception:
                pass
        set_iter = getattr(self.func, 'set_iteration', None)
        if callable(set_iter):
            try:
                set_iter(iteration, self.num_iter)
            except TypeError:
                set_iter(iteration)
            except Exception:
                pass

    def _score_solution(self, x: Array, *, count_eval: bool=True) -> tuple[float, bool, float]:
        x_arr = np.asarray(x, dtype=float)
        if count_eval:
            self.evals_used += 1
        detail_fn = getattr(self.func, 'detail', None)
        if callable(detail_fn):
            try:
                info = dict(detail_fn(x_arr))
                value = _safe_float(info.get('fitness', self.max_penalty), self.max_penalty)
                cv = float(max(0.0, float(info.get('F2_obstacle', 0.0)))) + float(max(0.0, float(info.get('F3_height', 0.0))))
                feasible = bool(cv <= 0.0001)
                return (float(min(value, self.max_penalty)), feasible, float(cv))
            except Exception:
                pass
        try:
            value = float(self.func(x_arr))
        except Exception:
            value = float(self.max_penalty)
        if not np.isfinite(value):
            value = float(self.max_penalty)
        try:
            path, _ = _decode_path_from_ctx(x_arr, self.ctx)
            fit = _evaluate_path_from_ctx(path, self.ctx, default=float(value))
            feasible = _path_success_from_ctx(path, self.ctx)
            cv = 0.0 if feasible else max(1.0, float(fit) / max(float(self.max_penalty), 1.0))
            value = float(min(value, fit, self.max_penalty))
            return (value, bool(feasible), float(cv))
        except Exception:
            return (float(min(value, self.max_penalty)), False, float(self.max_penalty))

    def _evaluate(self, x: Array) -> float:
        value, _feasible, _cv = self._score_solution(x, count_eval=True)
        return float(value)

    @staticmethod
    def _dominates(value_a: float, feasible_a: bool, cv_a: float, value_b: float, feasible_b: bool, cv_b: float) -> bool:
        if bool(feasible_a) != bool(feasible_b):
            return bool(feasible_a)
        if not bool(feasible_a):
            if float(cv_a) < float(cv_b) - 1e-09:
                return True
            if abs(float(cv_a) - float(cv_b)) <= 1e-09:
                return float(value_a) < float(value_b)
            return False
        return float(value_a) < float(value_b) - 1e-09

    @staticmethod
    def _risk_from_cv(cv: float) -> float:
        cv = max(float(cv), 0.0)
        return float(cv / (1.0 + cv))

    def _initialize_swarm(self) -> None:
        self.particles.clear()
        self.gbest = None
        self.gvalue = float('inf')
        self.gbest_feasible = False
        self.gbest_cv = float('inf')
        for _ in range(self.n_particles):
            x = _sample_by_bounds(self.rng, self.n_dims, self.st_x)
            _clip_by_bounds(x, self.st_x)
            v = _sample_velocity(self.rng, self.n_dims, self.st_v, self.st_x)
            _clip_by_bounds(v, self.st_v)
            value, feasible, cv = self._score_solution(x, count_eval=True)
            particle = Particle(position=x.copy(), velocity=v.copy(), pbest=x.copy(), pvalue=float(value), value=float(value), feasible=bool(feasible), cv=float(cv), pbest_feasible=bool(feasible), pbest_cv=float(cv))
            self.particles.append(particle)
            if self.gbest is None or self._dominates(value, feasible, cv, self.gvalue, self.gbest_feasible, self.gbest_cv):
                self.gvalue = float(value)
                self.gbest = x.copy()
                self.gbest_feasible = bool(feasible)
                self.gbest_cv = float(cv)
        if self.gbest is None:
            self.gbest = self.particles[0].pbest.copy()
            self.gvalue = float(self.particles[0].pvalue)
            self.gbest_feasible = bool(self.particles[0].pbest_feasible)
            self.gbest_cv = float(self.particles[0].pbest_cv)

    def _refresh_swarm_memory(self, *, count_eval: bool=False) -> None:
        self.gbest = None
        self.gvalue = float('inf')
        self.gbest_feasible = False
        self.gbest_cv = float('inf')
        for p in self.particles:
            value, feasible, cv = self._score_solution(p.position, count_eval=count_eval)
            p.value = float(value)
            p.feasible = bool(feasible)
            p.cv = float(cv)
            p.pbest = p.position.copy()
            p.pvalue = float(value)
            p.pbest_feasible = bool(feasible)
            p.pbest_cv = float(cv)
            p.stagnant_iters = 0
        self._update_global_best()
        self._update_feasible_archive()

    def _apply_hot_start(self, config: Any | None=None) -> None:
        if apply_hot_start is None:
            warnings.warn('Hot-start module is unavailable; TAWG-PSO will use random initialization only.')
            self.hot_start_info = {'enabled': False, 'reason': 'module_unavailable'}
            return
        if not self.ctx:
            self.hot_start_info = {'enabled': False, 'reason': 'missing_ctx'}
            return
        try:
            if config is None and HotStartConfig is not None:
                config = HotStartConfig()
            info = apply_hot_start(self, ctx=self.ctx, seed=int(self.seed or 0), config=config, gate_cfg=self.ctx.get('gate_cfg', None), cost_fn=self._evaluate, rng=self.rng)
            pool = info.get('_candidate_xg_list', [])
            self.reinjection_pool = [np.asarray(x, dtype=float).copy() for x in pool]
            public_info = dict(info)
            public_info.pop('_candidate_xg_list', None)
            self.hot_start_info = public_info
            self._refresh_swarm_memory(count_eval=False)
        except Exception as exc:
            warnings.warn(f'Hot-start failed and was skipped: {exc}')
            self.hot_start_info = {'enabled': False, 'reason': str(exc)}

    def _population_matrix(self, attr: str='position') -> Array:
        return np.vstack([np.asarray(getattr(p, attr), dtype=float) for p in self.particles])

    def _pbest_values(self) -> Array:
        return np.asarray([float(p.pvalue) for p in self.particles], dtype=float)

    def _record_snapshot(self, iteration: int) -> None:
        if iteration not in self.snapshot_iters:
            return
        if self.gbest is None:
            return
        self.snapshots[int(iteration)] = {'gbest_xg': self.gbest.copy(), 'gvalue': float(self.gvalue)}

    def _update_global_best(self) -> None:
        for p in self.particles:
            if self._dominates(p.pvalue, p.pbest_feasible, p.pbest_cv, self.gvalue, self.gbest_feasible, self.gbest_cv):
                self.gvalue = float(p.pvalue)
                self.gbest = p.pbest.copy()
                self.gbest_feasible = bool(p.pbest_feasible)
                self.gbest_cv = float(p.pbest_cv)

    def _archive_key(self, record: EliteRecord) -> tuple:
        return (round(float(record.value), 6), tuple(np.round(record.x[:min(12, record.x.size)], 4).tolist()))

    def _update_feasible_archive(self) -> None:
        seen = {self._archive_key(r) for r in self.feasible_archive}
        for p in self.particles:
            if not bool(p.pbest_feasible):
                continue
            rec = EliteRecord(x=p.pbest.copy(), value=float(p.pvalue), cv=float(p.pbest_cv))
            key = self._archive_key(rec)
            if key in seen:
                continue
            seen.add(key)
            self.feasible_archive.append(rec)
        self.feasible_archive.sort(key=lambda r: (float(r.value), float(r.cv)))
        max_archive = max(6, min(24, self.n_particles // 2))
        if len(self.feasible_archive) > max_archive:
            self.feasible_archive = self.feasible_archive[:max_archive]

    def _select_exemplar(self, particle: Particle, base_ref: Array) -> Array:
        if self.feasible_archive:
            risk = self._risk_from_cv(particle.cv)
            if risk > 0.15 or particle.stagnant_iters >= 12 or self.rng.random() < 0.35:
                top_k = min(len(self.feasible_archive), 6)
                idx = int(self.rng.integers(0, top_k))
                return self.feasible_archive[idx].x.copy()
        return np.asarray(base_ref, dtype=float).copy()

    def _risk_adjusted_parameters(self, particle: Particle, w_t: float, c1_t: float, c2_t: float) -> tuple[float, float, float]:
        risk = self._risk_from_cv(particle.cv)
        w_i = float(w_t + 0.18 * risk * (0.95 - float(w_t)))
        c1_i = float(c1_t * (1.0 + 0.25 * risk))
        c2_i = float(c2_t * (1.0 - 0.3 * risk))
        return (float(np.clip(w_i, 0.2, 0.95)), float(np.clip(c1_i, 0.2, 3.0)), float(np.clip(c2_i, 0.2, 3.0)))

    def _global_reference(self, iteration: int) -> Array:
        if self.gbest is None:
            raise RuntimeError('gbest is not initialized.')
        if self.elite_controller is None:
            self.last_elite_info = {'enabled': False, 'reason': 'disabled'}
            return self.gbest.copy()
        try:
            X_P = self._population_matrix('pbest')
            J_P = self._pbest_values()
            G_ref, info = self.elite_controller.step(iteration=int(iteration), X_P=X_P, J_P=J_P, gbest=self.gbest, best_fit=float(self.gvalue), feasible_mask=None)
            G_ref = np.asarray(G_ref, dtype=float).reshape(-1)
            if G_ref.size != self.n_dims or not np.all(np.isfinite(G_ref)):
                raise ValueError('invalid elite guidance vector')
            _clip_by_bounds(G_ref, self.st_x)
            self.last_elite_info = dict(info)
            return G_ref
        except Exception as exc:
            self.last_elite_info = {'enabled': False, 'reason': str(exc)}
            return self.gbest.copy()

    def _dynamic_parameters(self, iteration: int) -> tuple[float, float, float]:
        if self.dynamic_scheduler is None:
            self.last_dynamic_info = {'enabled': False, 'w': self.w, 'c1': self.c1, 'c2': self.c2}
            return (float(self.w), float(self.c1), float(self.c2))
        try:
            X = self._population_matrix('position')
            w_t, c1_t, c2_t, info = self.dynamic_scheduler.step(int(iteration), population=X, best_fit=float(self.gvalue))
            cfg = require_tawgpso_fixed_config(self.dynamic_scheduler.cfg)
            if not (cfg.w_low <= float(w_t) <= cfg.w_high and cfg.c_low <= float(c1_t) <= cfg.c_high and (cfg.c_low <= float(c2_t) <= cfg.c_high)):
                raise RuntimeError(f'Dynamic scheduler returned values outside the locked {TAWG_PSO_PARAMETER_SET} boundaries.')
            self.last_dynamic_info = dict(info)
            return (float(w_t), float(c1_t), float(c2_t))
        except Exception as exc:
            self.last_dynamic_info = {'enabled': False, 'reason': str(exc)}
            raise RuntimeError('TAWG-PSO dynamic parameter scheduling failed; refusing to fall back to unverified static parameters.') from exc

    def _update_particle(self, particle: Particle, g_ref: Array, w_t: float, c1_t: float, c2_t: float) -> None:
        r1 = self.rng.random(self.n_dims)
        r2 = self.rng.random(self.n_dims)
        exemplar = self._select_exemplar(particle, g_ref)
        w_i, c1_i, c2_i = self._risk_adjusted_parameters(particle, w_t, c1_t, c2_t)
        particle.velocity = float(w_i) * particle.velocity + float(c1_i) * r1 * (particle.pbest - particle.position) + float(c2_i) * r2 * (exemplar - particle.position)
        _clip_by_bounds(particle.velocity, self.st_v)
        particle.position = particle.position + particle.velocity
        _clip_by_bounds(particle.position, self.st_x)
        value, feasible, cv = self._score_solution(particle.position, count_eval=True)
        particle.value = float(value)
        particle.feasible = bool(feasible)
        particle.cv = float(cv)
        if self._dominates(value, feasible, cv, particle.pvalue, particle.pbest_feasible, particle.pbest_cv):
            particle.pvalue = float(value)
            particle.pbest = particle.position.copy()
            particle.pbest_feasible = bool(feasible)
            particle.pbest_cv = float(cv)
            particle.stagnant_iters = 0
        else:
            particle.stagnant_iters += 1

    def _feasible_ratio(self) -> float:
        if not self.particles:
            return 0.0
        return float(np.mean([bool(p.feasible) for p in self.particles]))

    def _particle_worst_key(self, idx: int) -> tuple:
        p = self.particles[idx]
        return (1 if not p.pbest_feasible else 0, float(p.pbest_cv), float(p.pvalue), float(p.stagnant_iters))

    def _candidate_for_reinjection(self, local_idx: int) -> Array:
        if self.reinjection_pool:
            base = self.reinjection_pool[local_idx % len(self.reinjection_pool)]
            return self._neighbor_for_reinjection(base, ratio=1.0)
        if self.feasible_archive:
            base = self.feasible_archive[local_idx % len(self.feasible_archive)].x
            return self._neighbor_for_reinjection(base, ratio=1.0)
        return _sample_by_bounds(self.rng, self.n_dims, self.st_x)

    def _replace_particle(self, idx: int, x: Array) -> None:
        x = np.asarray(x, dtype=float).reshape(-1).copy()
        _clip_by_bounds(x, self.st_x)
        v = _sample_velocity(self.rng, self.n_dims, self.st_v, self.st_x)
        _clip_by_bounds(v, self.st_v)
        value, feasible, cv = self._score_solution(x, count_eval=True)
        p = self.particles[int(idx)]
        p.position = x.copy()
        p.velocity = v.copy()
        p.pbest = x.copy()
        p.pvalue = float(value)
        p.value = float(value)
        p.feasible = bool(feasible)
        p.cv = float(cv)
        p.pbest_feasible = bool(feasible)
        p.pbest_cv = float(cv)
        p.stagnant_iters = 0

    def _maybe_reinject(self, iteration: int) -> None:
        if not self.use_stagnation_reinjection:
            self.reinjection_info.setdefault('enabled', False)
            return
        if self.n_particles < 6:
            return
        if int(iteration) - int(self.last_reinject_iter) < 35:
            return
        feasible_ratio = self._feasible_ratio()
        should_reinject = self.global_stagnant_iters >= 35 or (int(iteration) >= 60 and feasible_ratio < 0.35 and (self.global_stagnant_iters >= 12))
        if not should_reinject:
            return
        n_replace = max(1, int(round(0.15 * self.n_particles)))
        n_replace = min(n_replace, max(1, self.n_particles // 4))
        ranked = sorted(range(self.n_particles), key=self._particle_worst_key, reverse=True)
        chosen = ranked[:n_replace]
        for j, idx in enumerate(chosen):
            self._replace_particle(idx, self._candidate_for_reinjection(j))
        self.last_reinject_iter = int(iteration)
        self.global_stagnant_iters = 0
        self._update_global_best()
        self._update_feasible_archive()
        self.reinjection_info = {'enabled': True, 'last_iter': int(iteration), 'count': int(self.reinjection_info.get('count', 0)) + 1, 'last_replaced': int(len(chosen)), 'feasible_ratio_before': float(feasible_ratio), 'pool_size': int(len(self.reinjection_pool)), 'archive_size': int(len(self.feasible_archive))}

    def _neighbor_for_reinjection(self, x: Array, ratio: float) -> Array:
        x2 = np.asarray(x, dtype=float).copy()
        if x2.size % 4 != 0:
            noise = self.rng.normal(0.0, 0.01, size=x2.size)
            x2 += noise * max(float(ratio), 0.001)
            return _clip_by_bounds(x2, self.st_x)
        blocks = x2.reshape(-1, 4)
        k_cap = blocks.shape[0]
        n_change = int(self.rng.integers(1, max(2, min(k_cap, 3)) + 1))
        idx = self.rng.choice(k_cap, size=n_change, replace=False)
        group_bounds = _as_group_bounds(self.st_x)
        if group_bounds is not None and len(group_bounds) >= 4:
            x_span = max(group_bounds[0][1] - group_bounds[0][0], 1.0)
            y_span = max(group_bounds[1][1] - group_bounds[1][0], 1.0)
            z_span = max(group_bounds[2][1] - group_bounds[2][0], 1.0)
            g_span = max(group_bounds[3][1] - group_bounds[3][0], 1.0)
        else:
            x_span = y_span = z_span = g_span = 1.0
        r = max(float(ratio), 0.001)
        scales = np.asarray([0.02 * x_span, 0.02 * y_span, 0.015 * z_span, 0.03 * g_span], dtype=float) * r
        blocks[idx, :] += self.rng.normal(0.0, scales, size=(n_change, 4))
        return _clip_by_bounds(blocks.reshape(-1), self.st_x)

    def solve(self) -> tuple[Array, float, list[float]]:
        self._record_snapshot(0)
        for it in range(1, self.num_iter + 1):
            self._set_cost_iteration(it)
            w_t, c1_t, c2_t = self._dynamic_parameters(it)
            g_ref = self._global_reference(it)
            old_value = float(self.gvalue)
            old_feasible = bool(self.gbest_feasible)
            old_cv = float(self.gbest_cv)
            for particle in self.particles:
                self._update_particle(particle, g_ref, w_t, c1_t, c2_t)
            self._update_global_best()
            improved = self._dominates(self.gvalue, self.gbest_feasible, self.gbest_cv, old_value, old_feasible, old_cv)
            if improved:
                self.global_stagnant_iters = 0
            else:
                self.global_stagnant_iters += 1
            self._update_feasible_archive()
            self._maybe_reinject(it)
            self.history.append(float(self.gvalue))
            self._record_snapshot(it)
        if self.gbest is None:
            raise RuntimeError('TAWG-PSO failed to produce a valid global best solution.')
        return (self.gbest.copy(), float(self.gvalue), list(self.history))

def _load_default_ctx() -> dict[str, Any]:
    from configuration.protocol import get_ctx_shared
    return get_ctx_shared(reset=True)

def _require_ctx_keys(ctx: Mapping[str, Any]) -> None:
    required = ('cost', 'decode_solution', 'N_DIMS', 'st_x', 'st_v', 'pso_params', 'num_iter')
    for key in required:
        if key not in ctx:
            raise KeyError(f'ctx missing required key: {key!r}')

def _get_cost_from_ctx(ctx: Mapping[str, Any]) -> EnergyFn:
    cost = ctx.get('cost', None)
    if cost is None or not callable(cost):
        raise KeyError("ctx must contain callable ctx['cost'].")
    return cost

def _get_pso_params_from_ctx(ctx: Mapping[str, Any]) -> dict[str, float | int]:
    params = dict(ctx.get('pso_params', {}))
    if not params:
        raise KeyError("ctx must contain ctx['pso_params'].")
    return {'n_particles': int(params.get('n_particles', ctx.get('n_particles', 50))), 'w': float(params.get('w', 0.58)), 'c1': float(params.get('c1', 2.0)), 'c2': float(params.get('c2', 2.0))}

def _decode_path_from_ctx(x_best: Array, ctx: Mapping[str, Any]) -> tuple[Array, int]:
    decoder = ctx.get('decode_solution', None)
    if not callable(decoder):
        raise RuntimeError("ctx must provide callable ctx['decode_solution'].")
    out = decoder(np.asarray(x_best, dtype=float))
    if isinstance(out, dict):
        path = np.asarray(out['path'], dtype=float)
        k_eff = int(out.get('K_eff', out.get('k_eff', max(path.shape[0] - 2, 0))))
        return (path, k_eff)
    if isinstance(out, tuple):
        path = np.asarray(out[0], dtype=float)
        k_eff = int(out[1]) if len(out) > 1 else max(path.shape[0] - 2, 0)
        return (path, k_eff)
    path = np.asarray(out, dtype=float)
    return (path, int(max(path.shape[0] - 2, 0)))

def _print_single_run_metrics(result: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any] | None:
    try:
        from evaluation.metrics import print_single_run_metrics
        return print_single_run_metrics('TAWG-PSO', result, ctx, recompute_fitness=False)
    except Exception as exc:
        warnings.warn(f'Failed to summarize TAWG-PSO metrics with experiment_metrics.py: {exc}')
        meta = result.get('meta', {})
        print(f"[TAWG-PSO] K_eff={int(meta.get('K_eff', 0))}, path_len={float(meta.get('path_length', float('nan'))):.4f} m, gvalue={float(meta.get('gvalue', float('nan'))):.6f}, time={float(result.get('time_s', float('nan'))):.3f} s")
        return None

def _plot_with_experiment_plotter(res: dict[str, Any], ctx: dict[str, Any], *, save_dir: str | None=None, show: bool=True, draw_terrain: bool=True, draw_obstacles: bool=True, y_log: bool=False) -> dict[str, str]:
    try:
        from visualization.plotting import plot_result_bundle
    except Exception as exc:
        raise RuntimeError('Cannot import visualization.plotting.plot_result_bundle.') from exc
    return plot_result_bundle(res=res, ctx=ctx, save_dir=save_dir, prefix='TAWG-PSO', show=show, draw_terrain=draw_terrain, draw_obstacles=draw_obstacles, y_log=y_log, dpi=300)

def solve_tawgpso_path_once(*, ctx: dict[str, Any] | None=None, seed: int=2026001, rng: np.random.Generator | None=None, num_iter: int | None=None, n_particles: int | None=None, use_hot_start: bool=True, use_dynamic_parameter: bool=True, use_elite_guidance: bool=True, use_stagnation_reinjection: bool=True, hot_start_config: Any | None=None, dynamic_config: Any | None=None, elite_config: Any | None=None, snapshot_iters: Iterable[int] | None=None, plot: bool=False, plot_save_dir: str | None=None, plot_show: bool=True, plot_draw_terrain: bool=True, plot_draw_obstacles: bool=True, plot_y_log: bool=False, print_metrics: bool=True) -> dict[str, Any]:
    if ctx is None:
        ctx = _load_default_ctx()
    _require_ctx_keys(ctx)
    cost = _get_cost_from_ctx(ctx)
    params = _get_pso_params_from_ctx(ctx)
    fixed_dynamic_config = require_tawgpso_fixed_config(get_tawgpso_dynamic_config() if dynamic_config is None else dynamic_config)
    T = int(num_iter if num_iter is not None else ctx.get('num_iter', 400))
    N = int(n_particles if n_particles is not None else params['n_particles'])
    n_dims = int(ctx['N_DIMS'])
    set_iter = ctx.get('set_cost_iteration', None)
    if callable(set_iter):
        try:
            set_iter(1, T)
        except TypeError:
            set_iter(1)
        except Exception:
            pass
    if snapshot_iters is None:
        snapshot_iters = ctx.get('DEFAULT_SNAPSHOT_ITERS', ())
    run_rng = rng if rng is not None else np.random.default_rng(int(seed))
    t0 = time.perf_counter()
    opt = TAWGPSO(n_dims=n_dims, n_particles=N, st_x=ctx['st_x'], st_v=ctx['st_v'], num_iter=T, func=cost, ctx=ctx, seed=int(seed), rng=run_rng, w=float(params['w']), c1=float(params['c1']), c2=float(params['c2']), use_hot_start=use_hot_start, use_dynamic_parameter=use_dynamic_parameter, use_elite_guidance=use_elite_guidance, use_stagnation_reinjection=use_stagnation_reinjection, hot_start_config=hot_start_config, dynamic_config=fixed_dynamic_config, elite_config=elite_config, snapshot_iters=snapshot_iters, max_penalty=MAX_PENALTY)
    gbest, gvalue, history = opt.solve()
    search_time_s = float(time.perf_counter() - t0)
    final_eval_t0 = time.perf_counter()
    if callable(set_iter):
        try:
            set_iter(T, T)
        except TypeError:
            set_iter(T)
        except Exception:
            pass
    raw_path, raw_k_eff = _decode_path_from_ctx(gbest, ctx)
    raw_path = np.asarray(raw_path, dtype=float)
    raw_path_length = _path_length(raw_path)
    raw_fitness = _evaluate_path_from_ctx(raw_path, ctx, default=float(gvalue))
    raw_success = _path_success_from_ctx(raw_path, ctx)
    path = np.asarray(raw_path, dtype=float).copy()
    k_eff = int(raw_k_eff)
    path_len = _path_length(path)
    final_cost = _evaluate_path_from_ctx(path, ctx, default=float(gvalue))
    final_success = _path_success_from_ctx(path, ctx)
    final_eval_time_s = float(time.perf_counter() - final_eval_t0)
    time_s = float(time.perf_counter() - t0)
    res: dict[str, Any] = {'algo': 'TAWG-PSO', 'seed': int(seed), 'path': np.asarray(path, dtype=float), 'history': list(history), 'time_s': float(time_s), 'search_time_s': float(search_time_s), 'final_eval_time_s': float(final_eval_time_s), 'meta': {'algo': 'TAWG-PSO', 'seed': int(seed), 'K_eff': int(k_eff), 'K_cap': int(ctx.get('K_CAP', ctx.get('K_MAX', n_dims // 4))), 'n_particles': int(N), 'num_iter': int(T), 'path_length': float(path_len), 'gvalue': float(final_cost), 'search_time_s': float(search_time_s), 'final_eval_time_s': float(final_eval_time_s), 'gbest_value': float(gvalue), 'gbest_feasible': bool(opt.gbest_feasible), 'gbest_feasible_int': int(bool(opt.gbest_feasible)), 'gbest_cv': float(opt.gbest_cv), 'evals_used': int(opt.evals_used), 'feasible_archive_size': int(len(opt.feasible_archive)), 'global_stagnant_iters': int(opt.global_stagnant_iters), 'reinjection_count': int(opt.reinjection_info.get('count', 0)), 'last_reinject_iter': int(opt.reinjection_info.get('last_iter', 0)), 'last_reinject_replaced': int(opt.reinjection_info.get('last_replaced', 0)), 'hot_start_n_candidates': int(opt.hot_start_info.get('n_candidates', 0)), 'hot_start_n_warm': int(opt.hot_start_info.get('n_warm', 0)), 'raw_K_eff': int(raw_k_eff), 'raw_path_length': float(raw_path_length), 'raw_fitness': float(raw_fitness), 'raw_success': bool(raw_success), 'raw_success_int': int(bool(raw_success)), 'final_K_eff': int(k_eff), 'final_path_length': float(path_len), 'final_fitness': float(final_cost), 'final_success': bool(final_success), 'final_success_int': int(bool(final_success)), 'use_hot_start': bool(use_hot_start), 'use_hot_start_int': int(bool(use_hot_start)), 'use_dynamic_parameter': bool(use_dynamic_parameter), 'use_dynamic_parameter_int': int(bool(use_dynamic_parameter)), 'use_elite_guidance': bool(use_elite_guidance), 'use_elite_guidance_int': int(bool(use_elite_guidance)), 'use_stagnation_reinjection': bool(use_stagnation_reinjection), 'use_stagnation_reinjection_int': int(bool(use_stagnation_reinjection)), 'dynamic_parameter_set': TAWG_PSO_PARAMETER_SET, 'dynamic_w_high': float(opt.dynamic_scheduler.cfg.w_high) if opt.dynamic_scheduler is not None else float(opt.dynamic_config.w_high), 'dynamic_w_low': float(opt.dynamic_scheduler.cfg.w_low) if opt.dynamic_scheduler is not None else float(opt.dynamic_config.w_low), 'dynamic_c_high': float(opt.dynamic_scheduler.cfg.c_high) if opt.dynamic_scheduler is not None else float(opt.dynamic_config.c_high), 'dynamic_c_low': float(opt.dynamic_scheduler.cfg.c_low) if opt.dynamic_scheduler is not None else float(opt.dynamic_config.c_low)}, 'gbest_xg': np.asarray(gbest, dtype=float).copy(), 'snapshots': dict(opt.snapshots), 'figures': {}, 'diagnostics': {'hot_start': dict(opt.hot_start_info), 'reinjection': dict(opt.reinjection_info), 'last_dynamic': dict(opt.last_dynamic_info), 'last_elite': dict(opt.last_elite_info), 'optimizer': {'gbest_feasible': bool(opt.gbest_feasible), 'gbest_cv': float(opt.gbest_cv), 'feasible_archive_size': int(len(opt.feasible_archive)), 'global_stagnant_iters': int(opt.global_stagnant_iters), 'reinjection_pool_size': int(len(opt.reinjection_pool))}, 'raw_final_audit': {'raw_K_eff': int(raw_k_eff), 'raw_path_length': float(raw_path_length), 'raw_fitness': float(raw_fitness), 'raw_success': bool(raw_success), 'final_K_eff': int(k_eff), 'final_path_length': float(path_len), 'final_fitness': float(final_cost), 'final_success': bool(final_success)}}}
    if print_metrics:
        record = _print_single_run_metrics(res, ctx)
        if record is not None:
            res['record'] = record
    if plot:
        figures = _plot_with_experiment_plotter(res, ctx, save_dir=plot_save_dir, show=plot_show, draw_terrain=plot_draw_terrain, draw_obstacles=plot_draw_obstacles, y_log=plot_y_log)
        res['figures'] = figures
        if figures:
            print(f"[TAWG-PSO] figures saved: path={figures.get('path')}, convergence={figures.get('convergence')}")
        elif plot_show:
            print('[TAWG-PSO] path and convergence figures displayed.')
    return res
run_once = solve_tawgpso_path_once

def main(seed: int=2026001, num_iter: int=400) -> None:
    ctx = _load_default_ctx()
    solve_tawgpso_path_once(ctx=ctx, seed=int(seed), rng=np.random.default_rng(int(seed)), num_iter=int(num_iter), plot=True, plot_save_dir=None, plot_show=True, plot_draw_terrain=True, plot_draw_obstacles=True, plot_y_log=False, print_metrics=True)
if __name__ == '__main__':
    import sys
    seed_arg = int(sys.argv[1]) if len(sys.argv) >= 2 else 2026001
    iter_arg = int(sys.argv[2]) if len(sys.argv) >= 3 else 400
    main(seed=seed_arg, num_iter=iter_arg)
