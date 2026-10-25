"""Run-level validation, feasibility, fitness, timing, energy, and aggregate statistics."""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np

try:
    from evaluation.objectives import path_is_feasible as _path_is_feasible
except Exception:
    _path_is_feasible = None
try:
    from evaluation.energy import compute_energy_index as _compute_energy_index
    from evaluation.energy import EnergyConfig as _EnergyConfig
except Exception:
    _compute_energy_index = None
    _EnergyConfig = None

def as_path(path: Any) -> np.ndarray:
    P = np.asarray(path, dtype=float)
    if P.ndim != 2 or P.shape[1] < 3:
        raise ValueError('path must have shape (n_points, 3) or (n_points, >=3).')
    P = P[:, :3]
    if P.shape[0] < 2:
        raise ValueError('path must contain at least start and goal.')
    if not np.all(np.isfinite(P)):
        raise ValueError('path contains non-finite values.')
    return P

def compute_path_length(path: Any) -> float:
    P = as_path(path)
    seg = P[1:] - P[:-1]
    return float(np.sum(np.linalg.norm(seg, axis=1)))

def evaluate_path_fitness(path: Any, ctx: dict[str, Any], *, default: float=float('nan')) -> float:
    P = as_path(path)
    eval_path_fn = getattr(ctx.get('cost', None), 'evaluate_path', None)
    if not callable(eval_path_fn):
        return float(default)
    try:
        out = eval_path_fn(P, return_detail=False)
        value = float(out[0]) if isinstance(out, tuple) else float(out)
        return value if np.isfinite(value) else float(default)
    except Exception:
        return float(default)

def _safe_angle_between(v1: np.ndarray, v2: np.ndarray, eps: float=1e-12) -> float:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < eps or n2 < eps:
        return 0.0
    c = float(np.dot(v1, v2) / (n1 * n2))
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.arccos(c))

def _fallback_compute_energy(path: Any, k_eff: int | None=None) -> float:
    P = as_path(path)
    if P.shape[0] < 2:
        return float('nan')
    k_dist = 1.0
    k_turn = 8.0
    k_climb = 2.5
    k_pitch = 5.0
    V = P[1:] - P[:-1]
    seg_len = np.linalg.norm(V, axis=1)
    path_length = float(np.sum(seg_len))
    dz = np.diff(P[:, 2])
    total_climb = float(np.sum(np.maximum(dz, 0.0)))
    turn_sum = 0.0
    if V.shape[0] >= 2:
        for i in range(1, V.shape[0]):
            turn_sum += abs(_safe_angle_between(V[i - 1], V[i]))
    horizontal = np.sqrt(V[:, 0] ** 2 + V[:, 1] ** 2)
    pitch = np.arctan2(V[:, 2], horizontal)
    if pitch.shape[0] >= 2:
        pitch_change_sum = float(np.sum(np.abs(np.diff(pitch))))
    else:
        pitch_change_sum = 0.0
    energy = k_dist * path_length + k_turn * turn_sum + k_climb * total_climb + k_pitch * pitch_change_sum
    return float(energy)

def compute_energy(path: Any, *, k_eff: int | None=None, energy_config: Any | None=None) -> float:
    P = as_path(path)
    if _compute_energy_index is not None:
        try:
            info = _compute_energy_index(path=P, k_eff=k_eff, config=energy_config, strict=False)
            value = float(info.get('energy_total', float('nan')))
            if np.isfinite(value):
                return value
        except Exception:
            pass
    return _fallback_compute_energy(P, k_eff=k_eff)

def compute_k_eff(result: dict[str, Any]) -> int:
    meta = dict(result.get('meta', {}))
    for key in ('K_eff', 'k_eff'):
        if key in meta:
            try:
                return int(meta[key])
            except Exception:
                pass
    P = as_path(result['path'])
    return int(max(P.shape[0] - 2, 0))

def check_success(path: Any, ctx: dict[str, Any], *, tol: float=1e-08) -> bool:
    P = as_path(path)
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
    if _path_is_feasible is None:
        return False
    cfg = ctx.get('cfg', {})
    try:
        return bool(_path_is_feasible(P, obstacles=ctx['obstacles'], height_fn=ctx.get('height_fn', None), hmin=float(cfg.get('HMIN', ctx.get('HMIN', 0.0))), hmax=float(cfg.get('HMAX', ctx.get('HMAX', 180.0))), agl_min=float(cfg.get('AGL_MIN', ctx.get('AGL_MIN', 5.0))), agl_max=None if cfg.get('AGL_MAX', ctx.get('AGL_MAX', 120.0)) is None else float(cfg.get('AGL_MAX', ctx.get('AGL_MAX', 120.0))), safety_margin=float(cfg.get('SAFETY_MARGIN', ctx.get('SAFETY_MARGIN', 4.0))), forbid_overfly=bool(cfg.get('FORBID_OVERFLY', ctx.get('FORBID_OVERFLY', True))), overfly_margin=float(cfg.get('OVERFLY_MARGIN', ctx.get('OVERFLY_MARGIN', 5.0))), tol=float(tol)))
    except Exception:
        return False

def compute_fitness(result: dict[str, Any], ctx: dict[str, Any], *, recompute: bool=False) -> float:
    meta = dict(result.get('meta', {}))
    if recompute and result.get('gbest_xg', None) is not None:
        cost = ctx.get('cost', None)
        if callable(cost):
            try:
                num_iter = int(meta.get('num_iter', ctx.get('num_iter', 0)))
                set_iter = ctx.get('set_cost_iteration', None)
                if callable(set_iter) and num_iter > 0:
                    set_iter(num_iter, num_iter)
                value = float(cost(np.asarray(result['gbest_xg'], dtype=float)))
                if np.isfinite(value):
                    return value
            except Exception:
                pass
    for key in ('final_fitness', 'gvalue', 'fitness', 'fit'):
        if key in meta:
            try:
                value = float(meta[key])
                if np.isfinite(value):
                    return value
            except Exception:
                pass
    if result.get('gbest_xg', None) is not None:
        cost = ctx.get('cost', None)
        if callable(cost):
            try:
                value = float(cost(np.asarray(result['gbest_xg'], dtype=float)))
                if np.isfinite(value):
                    return value
            except Exception:
                pass
    return float('nan')

def validate_result(result: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise TypeError('result must be a dict.')
    if 'path' not in result:
        raise KeyError("result missing key: 'path'.")
    if 'time_s' not in result:
        raise KeyError("result missing key: 'time_s'.")
    out = dict(result)
    out['path'] = as_path(out['path'])
    out['time_s'] = float(out['time_s'])
    out['meta'] = dict(out.get('meta', {}))
    out['history'] = list(out.get('history', []))
    if out.get('gbest_xg', None) is not None:
        out['gbest_xg'] = np.asarray(out['gbest_xg'], dtype=float).copy()
    return out

def _safe_optional_float(value: Any, default: float=float('nan')) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)

def _get_optional_float(key: str, *sources: dict[str, Any], default: float=float('nan')) -> float:
    for source in sources:
        if key not in source:
            continue
        value = _safe_optional_float(source.get(key), default=float('nan'))
        if np.isfinite(value):
            return float(value)
    return float(default)

def extract_timing_metrics(result: dict[str, Any]) -> dict[str, float]:
    res = validate_result(result)
    meta = dict(res.get('meta', {}))
    runtime_s = float(res['time_s'])
    search_time_s = _get_optional_float('search_time_s', res, meta)
    final_eval_time_s = _get_optional_float('final_eval_time_s', res, meta)
    if not np.isfinite(final_eval_time_s) and np.isfinite(search_time_s):
        remain = runtime_s - float(search_time_s)
        if np.isfinite(remain) and remain >= 0.0:
            final_eval_time_s = float(remain)
    return {'runtime_s': float(runtime_s), 'search_time_s': float(search_time_s), 'final_eval_time_s': float(final_eval_time_s)}

def summarize_result(result: dict[str, Any], ctx: dict[str, Any], *, recompute_fitness: bool=False, success_tol: float=1e-08, energy_config: Any | None=None) -> dict[str, Any]:
    res = validate_result(result)
    meta = dict(res['meta'])
    P = res['path']
    algo = str(meta.get('algo', result.get('algo', 'UNKNOWN')))
    seed = meta.get('seed', result.get('seed', None))
    seed = None if seed is None else int(seed)
    success = check_success(P, ctx, tol=success_tol)
    path_len_raw = compute_path_length(P)
    path_len = float(path_len_raw) if bool(success) else float('nan')
    timing = extract_timing_metrics(res)
    k_eff = compute_k_eff(res)
    if bool(success):
        energy = compute_energy(P, k_eff=k_eff, energy_config=energy_config)
    else:
        energy = float('nan')
    fitness = compute_fitness(res, ctx, recompute=recompute_fitness)
    record = {'algo': algo, 'seed': seed, 'path_length': float(path_len), 'success': bool(success), 'success_int': int(bool(success)), 'runtime_s': float(timing['runtime_s']), 'search_time_s': float(timing['search_time_s']), 'final_eval_time_s': float(timing['final_eval_time_s']), 'K_eff': int(k_eff), 'energy': float(energy), 'fitness': float(fitness), 'num_iter': int(meta.get('num_iter', ctx.get('num_iter', 0))), 'n_particles': None if meta.get('n_particles', ctx.get('n_particles', None)) is None else int(meta.get('n_particles', ctx.get('n_particles', 0))), 'gvalue': float(meta.get('gvalue', fitness))}
    for key in ('evals_used', 'eval_budget', 'accepted', 'improved', 'outer_steps'):
        if key in meta:
            record[key] = meta[key]
    diagnostic_keys = ('raw_K_eff', 'raw_path_length', 'raw_fitness', 'raw_success', 'raw_success_int', 'gbest_feasible', 'gbest_feasible_int', 'gbest_cv', 'feasible_archive_size', 'global_stagnant_iters', 'reinjection_count', 'last_reinject_iter', 'last_reinject_replaced', 'hot_start_n_candidates', 'hot_start_n_warm', 'final_K_eff', 'final_path_length', 'final_fitness', 'final_success', 'final_success_int', 'use_hot_start', 'use_hot_start_int', 'use_dynamic_parameter', 'use_dynamic_parameter_int', 'use_elite_guidance', 'use_elite_guidance_int', 'use_stagnation_reinjection', 'use_stagnation_reinjection_int', 'dynamic_parameter_set', 'dynamic_w_high', 'dynamic_w_low', 'dynamic_c_high', 'dynamic_c_low')
    for key in diagnostic_keys:
        if key in meta:
            record[key] = meta[key]
    return record

def summarize_results(results: Iterable[dict[str, Any]], ctx: dict[str, Any], *, recompute_fitness: bool=False, success_tol: float=1e-08, energy_config: Any | None=None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for result in results:
        records.append(summarize_result(result, ctx, recompute_fitness=recompute_fitness, success_tol=success_tol, energy_config=energy_config))
    return records

def _finite_array(values: Iterable[Any]) -> np.ndarray:
    arr = []
    for v in values:
        try:
            x = float(v)
            if np.isfinite(x):
                arr.append(x)
        except Exception:
            pass
    return np.asarray(arr, dtype=float)

def _mean_std(values: Iterable[Any]) -> tuple[float, float]:
    arr = _finite_array(values)
    if arr.size == 0:
        return (float('nan'), float('nan'))
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0
    return (mean, std)

def _mean_var(values: Iterable[Any]) -> tuple[float, float]:
    arr = _finite_array(values)
    if arr.size == 0:
        return (float('nan'), float('nan'))
    mean = float(np.mean(arr))
    var = float(np.var(arr, ddof=1)) if arr.size >= 2 else 0.0
    return (mean, var)

def _success_value(record: dict[str, Any], success_key: str='success_int') -> bool:
    if success_key in record:
        try:
            return bool(int(record.get(success_key, 0)))
        except Exception:
            return bool(record.get(success_key, False))
    if success_key == 'success_int' and 'success' in record:
        return bool(record.get('success', False))
    return False

def _metric_values(records: Sequence[dict[str, Any]], metric: str, *, success_key: str | None=None) -> list[Any]:
    if success_key is None:
        return [x.get(metric, float('nan')) for x in records]
    return [x.get(metric, float('nan')) for x in records if _success_value(x, success_key=success_key)]

def aggregate_records(records: Sequence[dict[str, Any]], *, group_key: str='algo') -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        algo = str(rec.get(group_key, 'UNKNOWN'))
        groups[algo].append(dict(rec))
    summary: list[dict[str, Any]] = []
    all_run_metrics = ('runtime_s', 'search_time_s', 'final_eval_time_s', 'K_eff', 'fitness')
    for algo, items in groups.items():
        row: dict[str, Any] = {group_key: algo, 'n_runs': int(len(items))}
        success_values = [int(_success_value(x, success_key='success_int')) for x in items]
        row['success_count'] = int(np.sum(success_values))
        row['success_rate'] = float(100.0 * np.mean(success_values)) if len(success_values) > 0 else float('nan')
        for metric in all_run_metrics:
            mean, std = _mean_std(_metric_values(items, metric))
            row[f'{metric}_mean'] = float(mean)
            row[f'{metric}_std'] = float(std)
        path_mean, path_std = _mean_std(_metric_values(items, 'path_length', success_key='success_int'))
        row['path_length_mean'] = float(path_mean)
        row['path_length_std'] = float(path_std)
        row['path_length_success_count'] = int(row['success_count'])
        energy_mean, energy_var = _mean_var(_metric_values(items, 'energy', success_key='success_int'))
        row['energy_mean'] = float(energy_mean)
        row['energy_var'] = float(energy_var)
        row['energy_success_count'] = int(row['success_count'])
        diagnostic_flags = (('raw_success_int', 'raw_success'), ('gbest_feasible_int', 'gbest_feasible'), ('final_success_int', 'final_success'), ('use_hot_start_int', 'use_hot_start'), ('use_dynamic_parameter_int', 'use_dynamic_parameter'), ('use_elite_guidance_int', 'use_elite_guidance'), ('use_stagnation_reinjection_int', 'use_stagnation_reinjection'))
        for value_key, out_key in diagnostic_flags:
            if any((value_key in x for x in items)):
                values = [int(bool(x.get(value_key, 0))) for x in items]
                row[f'{out_key}_count'] = int(np.sum(values))
                row[f'{out_key}_rate'] = float(100.0 * np.mean(values))
        diagnostic_means = ('raw_path_length', 'raw_fitness', 'gbest_cv', 'feasible_archive_size', 'global_stagnant_iters', 'reinjection_count', 'last_reinject_iter', 'last_reinject_replaced', 'hot_start_n_candidates', 'hot_start_n_warm', 'final_path_length', 'final_fitness')
        for metric in diagnostic_means:
            if any((metric in x for x in items)):
                success_key = None
                if metric == 'raw_path_length':
                    success_key = 'raw_success_int'
                elif metric == 'final_path_length':
                    success_key = 'final_success_int'
                mean, std = _mean_std(_metric_values(items, metric, success_key=success_key))
                row[f'{metric}_mean'] = float(mean)
                row[f'{metric}_std'] = float(std)
        summary.append(row)
    summary.sort(key=lambda x: str(x.get(group_key, '')))
    return summary

def format_mean_std(mean: float, std: float, digits: int=3) -> str:
    try:
        if not np.isfinite(float(mean)) or not np.isfinite(float(std)):
            return 'nan'
        return f'{float(mean):.{digits}f}+/-{float(std):.{digits}f}'
    except Exception:
        return 'nan'

def format_float(value: float, digits: int=3) -> str:
    try:
        if not np.isfinite(float(value)):
            return 'nan'
        return f'{float(value):.{digits}f}'
    except Exception:
        return 'nan'

def format_single_run_metrics(record: dict[str, Any], *, label: str | None=None, digits: int=3) -> str:
    prefix = f'[{label}] ' if label else ''
    return prefix + f"success={int(record.get('success_int', 0))}, K_eff={int(record.get('K_eff', 0))}, path_len={format_float(record.get('path_length', float('nan')), digits=4)} m, energy={format_float(record.get('energy', float('nan')), digits=4)}, fitness={format_float(record.get('fitness', float('nan')), digits=6)}, runtime={format_float(record.get('runtime_s', float('nan')), digits=digits)} s, search={format_float(record.get('search_time_s', float('nan')), digits=digits)} s, final_eval={format_float(record.get('final_eval_time_s', float('nan')), digits=digits)} s"

def print_single_run_metrics(label: str, result: dict[str, Any], ctx: dict[str, Any], *, recompute_fitness: bool=False, success_tol: float=1e-08, energy_config: Any | None=None) -> dict[str, Any]:
    record = summarize_result(result, ctx, recompute_fitness=recompute_fitness, success_tol=success_tol, energy_config=energy_config)
    print(format_single_run_metrics(record, label=label))
    return dict(record)

def make_table_rows(summary: Sequence[dict[str, Any]], *, digits: int=3) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s in summary:
        row = {'Algorithm': s.get('algo', 'UNKNOWN'), 'Success rate (%)': format_float(s.get('success_rate', float('nan')), digits=digits), 'Path length': format_mean_std(s.get('path_length_mean', float('nan')), s.get('path_length_std', float('nan')), digits=digits), 'Runtime (s)': format_mean_std(s.get('runtime_s_mean', float('nan')), s.get('runtime_s_std', float('nan')), digits=digits), 'Search time (s)': format_mean_std(s.get('search_time_s_mean', float('nan')), s.get('search_time_s_std', float('nan')), digits=digits), 'Final eval time (s)': format_mean_std(s.get('final_eval_time_s_mean', float('nan')), s.get('final_eval_time_s_std', float('nan')), digits=digits), 'K_eff': format_mean_std(s.get('K_eff_mean', float('nan')), s.get('K_eff_std', float('nan')), digits=digits), 'Energy': format_float(s.get('energy_mean', float('nan')), digits=digits), 'Energy variance': format_float(s.get('energy_var', float('nan')), digits=digits), 'Fitness': format_mean_std(s.get('fitness_mean', float('nan')), s.get('fitness_std', float('nan')), digits=digits)}
        rows.append(row)
    return rows

def _to_csv_value(v: Any) -> Any:
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (np.ndarray, list, tuple, dict)):
        return str(v)
    return v

def save_csv(rows: Sequence[dict[str, Any]], filepath: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    if not rows:
        with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
            f.write('')
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _to_csv_value(row.get(k, '')) for k in fieldnames})

def main() -> None:
    print('[experiment_metrics] ready')
    print('[experiment_metrics] energy metric: enabled')
    print('[experiment_metrics] energy output: energy_mean + energy_var only')
if __name__ == '__main__':
    main()
