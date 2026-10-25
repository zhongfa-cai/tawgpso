"""Frozen TAWG-PSO task parameters and context construction for the synthetic terrain experiments."""

from __future__ import annotations
import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
import copy
import inspect
import random
from typing import Any, Callable
import numpy as np
from environments.synthetic.terrain import GRID_LIMIT, load_dem
from environments.synthetic.terrain import make_height_fn, make_obstacles_for_cost
from evaluation.objectives import DEFAULT_WEIGHTS, build_cost_fixed_waypoints, build_cost_sipso_adaptive
from algorithm.tawg_pso import AdaptiveWaypointConfig
BASE_SEED = 2026001
N_RUNS_SMOKE = 30
N_RUNS_TABLE = 51
N_RUNS_CURVE = 50
N_PARTICLES = 51
NUM_ITER = 400
DEFAULT_SNAPSHOT_ITERS = (25, 50, 100, 200, 300, 350, 400)
DIM = 3
N_GROUP = DIM + 1
K_CAP = 26
TAU = 0.45
LAMBDA_K_MAX = 0.02
LAMBDA_B = 0.0
N_DIMS = N_GROUP * K_CAP
G_BOUNDS = (-4.0, 4.0)
ST_V = ((-1.5, 1.5), (-1.5, 1.5), (-0.8, 0.8), (-0.15, 0.15))
BASELINE_N_GROUP = DIM
BASELINE_WAYPOINTS = 15
BASELINE_N_DIMS = BASELINE_N_GROUP * BASELINE_WAYPOINTS
BASELINE_ST_V = ((-1.5, 1.5), (-1.5, 1.5), (-0.8, 0.8))
_BASE_CFG: dict[str, Any] = dict(K_CAP=K_CAP, OBSTACLE_RADIUS_EXPAND=0.0, FORBID_OVERFLY=True, OVERFLY_MARGIN=5.0, HMIN=0.0, HMAX=200.0, AGL_MIN=5.0, AGL_MAX=120, SAFETY_MARGIN=5.0, TAU=TAU, GATE_SHARPNESS=6.0, LAMBDA_K_MIN=0.0, LAMBDA_K_MAX=LAMBDA_K_MAX, LAMBDA_B=LAMBDA_B, GATE_SCHEDULE_POWER=2.0, BASELINE_WAYPOINTS=BASELINE_WAYPOINTS, BASELINE_SORT_WAYPOINTS=True)
_BASE_PSO_PARAMS: dict[str, Any] = dict(n_particles=N_PARTICLES, w=0.58, c1=2.0, c2=2.0, num_iter=NUM_ITER)

def _freeze(x: Any) -> Any:
    if isinstance(x, dict):
        return tuple(sorted(((str(k), _freeze(v)) for k, v in x.items())))
    if isinstance(x, (list, tuple)):
        return tuple((_freeze(v) for v in x))
    if isinstance(x, np.ndarray):
        return tuple((_freeze(v) for v in x.tolist()))
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    return x

def seed_all(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
_seed_all = seed_all

def make_seed(run_id: int, base_seed: int=BASE_SEED) -> int:
    return int(base_seed) + int(run_id)

def make_seed_list(n_runs: int, base_seed: int=BASE_SEED) -> list[int]:
    return [make_seed(i, base_seed=base_seed) for i in range(int(n_runs))]

def _extract_k_eff(path: np.ndarray, gate_info: Any) -> int:
    if isinstance(gate_info, dict):
        for key in ('K_eff', 'k_eff', 'n_active', 'active_count'):
            if key in gate_info:
                try:
                    return int(gate_info[key])
                except Exception:
                    pass
        if 'active_mask' in gate_info:
            try:
                return int(np.sum(np.asarray(gate_info['active_mask'], dtype=bool)))
            except Exception:
                pass
        if 'active_indices' in gate_info:
            try:
                return int(len(gate_info['active_indices']))
            except Exception:
                pass
    path = np.asarray(path, dtype=float)
    return int(max(path.shape[0] - 2, 0))

def make_cfg(overrides: dict[str, Any] | None=None) -> dict[str, Any]:
    cfg = copy.deepcopy(_BASE_CFG)
    if overrides:
        cfg.update(copy.deepcopy(overrides))
    return cfg

def make_pso_params(cfg: dict[str, Any] | None=None, overrides: dict[str, Any] | None=None) -> dict[str, Any]:
    if cfg is None:
        cfg = make_cfg()
    params = copy.deepcopy(_BASE_PSO_PARAMS)
    params['st_x'] = ((0.0, float(GRID_LIMIT)), (0.0, float(GRID_LIMIT)), (float(cfg['HMIN']), float(cfg['HMAX'])), (float(G_BOUNDS[0]), float(G_BOUNDS[1])))
    params['st_v'] = ST_V
    if overrides:
        params.update(copy.deepcopy(overrides))
    return params

def make_baseline_pso_params(cfg: dict[str, Any] | None=None, overrides: dict[str, Any] | None=None) -> dict[str, Any]:
    if cfg is None:
        cfg = make_cfg()
    params = copy.deepcopy(_BASE_PSO_PARAMS)
    params['n_waypoints'] = int(cfg.get('BASELINE_WAYPOINTS', BASELINE_WAYPOINTS))
    params['st_x'] = ((0.0, float(GRID_LIMIT)), (0.0, float(GRID_LIMIT)), (float(cfg['HMIN']), float(cfg['HMAX'])))
    params['st_v'] = BASELINE_ST_V
    if overrides:
        params.update(copy.deepcopy(overrides))
    params['n_waypoints'] = int(params.get('n_waypoints', cfg.get('BASELINE_WAYPOINTS', BASELINE_WAYPOINTS)))
    return params

def make_gate_cfg(cfg: dict[str, Any] | None=None, overrides: dict[str, Any] | None=None) -> AdaptiveWaypointConfig:
    if cfg is None:
        cfg = make_cfg()
    values: dict[str, Any] = dict(k_cap=int(cfg['K_CAP']), active_threshold=float(cfg['TAU']), lambda_k=float(cfg['LAMBDA_K_MAX']), lambda_binary=float(cfg['LAMBDA_B']))
    if overrides:
        key_map = {'k_cap': 'K_CAP', 'active_threshold': 'TAU', 'lambda_k': 'LAMBDA_K_MAX', 'lambda_binary': 'LAMBDA_B'}
        conflicts: list[str] = []
        for gate_key, override_value in overrides.items():
            cfg_key = key_map.get(gate_key)
            if cfg_key is None:
                continue
            cfg_value = values[gate_key]
            if isinstance(cfg_value, (int, float)) and isinstance(override_value, (int, float)):
                equal = bool(np.isclose(float(cfg_value), float(override_value)))
            else:
                equal = cfg_value == override_value
            if not equal:
                conflicts.append(f'{gate_key}={override_value!r} conflicts with cfg[{cfg_key!r}]={cfg_value!r}')
        if conflicts:
            raise ValueError('Gate parameters must be set through cfg_overrides only; divergent gate_overrides would create parameter drift: ' + '; '.join(conflicts))
        values.update(copy.deepcopy(overrides))
    sig = inspect.signature(AdaptiveWaypointConfig)
    kwargs: dict[str, Any] = {}
    for name, p in sig.parameters.items():
        if name in ('self', 'args', 'kwargs'):
            continue
        if name in values:
            kwargs[name] = values[name]
            continue
        if p.default is not inspect._empty:
            continue
        raise TypeError(f'AdaptiveWaypointConfig requires unsupported field: {name!r}. Please add this field in make_gate_cfg().')
    return AdaptiveWaypointConfig(**kwargs)

def build_cost_function(*, cfg: dict[str, Any], start: np.ndarray, goal: np.ndarray, obstacles: list[tuple], height_fn: Callable[[float, float], float], gate_cfg: AdaptiveWaypointConfig, max_iter: int) -> Callable[[np.ndarray], float]:
    return build_cost_sipso_adaptive(start=np.asarray(start, dtype=float), goal=np.asarray(goal, dtype=float), obstacles=obstacles, height_fn=height_fn, gate_cfg=gate_cfg, hmin=float(cfg['HMIN']), hmax=float(cfg['HMAX']), agl_min=float(cfg['AGL_MIN']), agl_max=None if cfg['AGL_MAX'] is None else float(cfg['AGL_MAX']), safety_margin=float(cfg['SAFETY_MARGIN']), forbid_overfly=bool(cfg['FORBID_OVERFLY']), overfly_margin=float(cfg['OVERFLY_MARGIN']), weights=DEFAULT_WEIGHTS, max_iter=int(max_iter))

def build_baseline_cost_function(*, cfg: dict[str, Any], start: np.ndarray, goal: np.ndarray, obstacles: list[tuple], height_fn: Callable[[float, float], float]) -> Callable[[np.ndarray], float]:
    return build_cost_fixed_waypoints(start=np.asarray(start, dtype=float), goal=np.asarray(goal, dtype=float), obstacles=obstacles, height_fn=height_fn, hmin=float(cfg['HMIN']), hmax=float(cfg['HMAX']), agl_min=float(cfg['AGL_MIN']), agl_max=None if cfg['AGL_MAX'] is None else float(cfg['AGL_MAX']), safety_margin=float(cfg['SAFETY_MARGIN']), forbid_overfly=bool(cfg['FORBID_OVERFLY']), overfly_margin=float(cfg['OVERFLY_MARGIN']), weights=DEFAULT_WEIGHTS, sort_waypoints=bool(cfg.get('BASELINE_SORT_WAYPOINTS', True)))

def decode_solution_with_ctx(xg: np.ndarray, ctx: dict[str, Any]) -> tuple[np.ndarray, int]:
    cost = ctx['cost']
    decode_fn = getattr(cost, 'decode_path', None)
    if not callable(decode_fn):
        raise RuntimeError("ctx['cost'] must provide decode_path(vector). Use the TAWG-PSO adaptive cost for TAWG-PSO and the fixed-xyz baseline cost for baseline algorithms.")
    path, gate_info = decode_fn(np.asarray(xg, dtype=float))
    path = np.asarray(path, dtype=float)
    k_eff = _extract_k_eff(path, gate_info)
    return (path, int(k_eff))

def set_cost_iteration(ctx: dict[str, Any], iteration: int | None, max_iter: int | None=None) -> None:
    fn = getattr(ctx['cost'], 'set_iteration', None)
    if callable(fn):
        try:
            fn(iteration, max_iter)
        except TypeError:
            fn(iteration)
_CTX_SHARED: dict[str, Any] | None = None
_CTX_BASELINE_SHARED: dict[str, Any] | None = None

def _ctx_fingerprint(*, cfg: dict[str, Any], pso_params: dict[str, Any], gate_cfg: AdaptiveWaypointConfig) -> tuple:
    return (_freeze(cfg), _freeze(pso_params), _freeze(gate_cfg), int(DIM), int(N_GROUP), int(cfg['K_CAP']), int(N_GROUP) * int(cfg['K_CAP']))

def build_ctx_shared(*, cfg_overrides: dict[str, Any] | None=None, pso_overrides: dict[str, Any] | None=None, gate_overrides: dict[str, Any] | None=None) -> dict[str, Any]:
    cfg = make_cfg(overrides=cfg_overrides)
    pso_params = make_pso_params(cfg=cfg, overrides=pso_overrides)
    gate_cfg = make_gate_cfg(cfg=cfg, overrides=gate_overrides)
    k_cap = int(cfg['K_CAP'])
    n_dims = int(N_GROUP) * k_cap
    X, Y, Z, START, GOAL = load_dem(clearance=float(cfg['AGL_MIN']))
    height_fn = make_height_fn(X, Y, Z)
    obstacles = make_obstacles_for_cost(radius_expand=float(cfg['OBSTACLE_RADIUS_EXPAND']))
    cost = build_cost_function(cfg=cfg, start=np.asarray(START, dtype=float), goal=np.asarray(GOAL, dtype=float), obstacles=obstacles, height_fn=height_fn, gate_cfg=gate_cfg, max_iter=int(pso_params['num_iter']))
    fp = _ctx_fingerprint(cfg=cfg, pso_params=pso_params, gate_cfg=gate_cfg)
    ctx: dict[str, Any] = dict(X=X, Y=Y, Z=Z, START=np.asarray(START, dtype=float), GOAL=np.asarray(GOAL, dtype=float), height_fn=height_fn, obstacles=obstacles, environment_source='environment.dem_terrain', cost=cost, gate_cfg=gate_cfg, fingerprint=fp, DIM=int(DIM), N_GROUP=int(N_GROUP), K_CAP=k_cap, N_DIMS=n_dims, tau=float(cfg['TAU']), st_x=pso_params['st_x'], st_v=pso_params['st_v'], g_bounds=tuple(G_BOUNDS), n_particles=int(pso_params['n_particles']), num_iter=int(pso_params['num_iter']), pso_params=dict(n_particles=int(pso_params['n_particles']), w=float(pso_params['w']), c1=float(pso_params['c1']), c2=float(pso_params['c2'])), cfg=copy.deepcopy(cfg), BASE_SEED=int(BASE_SEED), N_RUNS_SMOKE=int(N_RUNS_SMOKE), N_RUNS_TABLE=int(N_RUNS_TABLE), N_RUNS_CURVE=int(N_RUNS_CURVE), DEFAULT_SNAPSHOT_ITERS=tuple(DEFAULT_SNAPSHOT_ITERS))
    ctx['decode_solution'] = lambda xg, _ctx=ctx: decode_solution_with_ctx(xg, _ctx)
    ctx['set_cost_iteration'] = lambda iteration, max_iter=None, _ctx=ctx: set_cost_iteration(_ctx, iteration, max_iter)
    return ctx

def get_ctx_shared(reset: bool=False, *, cfg_overrides: dict[str, Any] | None=None, pso_overrides: dict[str, Any] | None=None, gate_overrides: dict[str, Any] | None=None) -> dict[str, Any]:
    global _CTX_SHARED
    cfg = make_cfg(overrides=cfg_overrides)
    pso_params = make_pso_params(cfg=cfg, overrides=pso_overrides)
    gate_cfg = make_gate_cfg(cfg=cfg, overrides=gate_overrides)
    expected_fp = _ctx_fingerprint(cfg=cfg, pso_params=pso_params, gate_cfg=gate_cfg)
    if reset or _CTX_SHARED is None:
        _CTX_SHARED = build_ctx_shared(cfg_overrides=cfg_overrides, pso_overrides=pso_overrides, gate_overrides=gate_overrides)
        return _CTX_SHARED
    if _CTX_SHARED.get('fingerprint') != expected_fp:
        _CTX_SHARED = build_ctx_shared(cfg_overrides=cfg_overrides, pso_overrides=pso_overrides, gate_overrides=gate_overrides)
    return _CTX_SHARED

def build_baseline_ctx_from_ctx(base_ctx: dict[str, Any], *, baseline_overrides: dict[str, Any] | None=None) -> dict[str, Any]:
    cfg = copy.deepcopy(base_ctx.get('cfg', make_cfg()))
    if baseline_overrides:
        cfg.update(copy.deepcopy(baseline_overrides))
    tawg_k_cap = cfg.pop('K_CAP', base_ctx.get('K_CAP', None))
    base_pso = dict(base_ctx.get('pso_params', {}))
    base_pso['n_particles'] = int(base_ctx.get('n_particles', base_pso.get('n_particles', N_PARTICLES)))
    base_pso['num_iter'] = int(base_ctx.get('num_iter', base_pso.get('num_iter', NUM_ITER)))
    baseline_params = make_baseline_pso_params(cfg=cfg, overrides=base_pso)
    n_waypoints = int(baseline_params.get('n_waypoints', cfg.get('BASELINE_WAYPOINTS', BASELINE_WAYPOINTS)))
    n_dims = int(BASELINE_N_GROUP * n_waypoints)
    cost = build_baseline_cost_function(cfg=cfg, start=np.asarray(base_ctx['START'], dtype=float), goal=np.asarray(base_ctx['GOAL'], dtype=float), obstacles=list(base_ctx['obstacles']), height_fn=base_ctx['height_fn'])
    ctx = dict(base_ctx)
    ctx['cfg'] = cfg
    ctx['cost'] = cost
    ctx['encoding'] = 'fixed_xyz'
    ctx['baseline_encoding'] = 'fixed_xyz'
    ctx['TAWG_K_CAP'] = tawg_k_cap
    ctx.pop('K_CAP', None)
    ctx.pop('gate_cfg', None)
    ctx.pop('g_bounds', None)
    ctx['DIM'] = int(DIM)
    ctx['N_GROUP'] = int(BASELINE_N_GROUP)
    ctx['N_WAYPOINTS'] = int(n_waypoints)
    ctx['K_FIXED'] = int(n_waypoints)
    ctx['N_DIMS'] = int(n_dims)
    ctx['st_x'] = baseline_params['st_x']
    ctx['st_v'] = baseline_params['st_v']
    ctx['n_particles'] = int(baseline_params['n_particles'])
    ctx['num_iter'] = int(baseline_params['num_iter'])
    ctx['pso_params'] = dict(n_particles=int(baseline_params['n_particles']), w=float(baseline_params['w']), c1=float(baseline_params['c1']), c2=float(baseline_params['c2']))
    ctx['fingerprint'] = ('baseline-fixed-xyz', _freeze(base_ctx.get('fingerprint', None)), _freeze(cfg), _freeze(baseline_params), int(BASELINE_N_GROUP), int(n_waypoints), int(n_dims))
    ctx['decode_solution'] = lambda xyz, _ctx=ctx: decode_solution_with_ctx(xyz, _ctx)
    ctx['set_cost_iteration'] = lambda iteration, max_iter=None, _ctx=ctx: set_cost_iteration(_ctx, iteration, max_iter)
    return ctx

def ensure_baseline_ctx(ctx: dict[str, Any], *, baseline_overrides: dict[str, Any] | None=None) -> dict[str, Any]:
    if str(ctx.get('encoding', '')) == 'fixed_xyz':
        return ctx
    cost = ctx.get('cost', None)
    if str(getattr(cost, 'encoding', '')) == 'fixed_xyz':
        return ctx
    return build_baseline_ctx_from_ctx(ctx, baseline_overrides=baseline_overrides)

def get_baseline_ctx_shared(reset: bool=False, *, cfg_overrides: dict[str, Any] | None=None, pso_overrides: dict[str, Any] | None=None, gate_overrides: dict[str, Any] | None=None, baseline_overrides: dict[str, Any] | None=None) -> dict[str, Any]:
    global _CTX_BASELINE_SHARED
    base_ctx = get_ctx_shared(reset=reset, cfg_overrides=cfg_overrides, pso_overrides=pso_overrides, gate_overrides=gate_overrides)
    candidate = build_baseline_ctx_from_ctx(base_ctx, baseline_overrides=baseline_overrides)
    if reset or _CTX_BASELINE_SHARED is None:
        _CTX_BASELINE_SHARED = candidate
    elif _CTX_BASELINE_SHARED.get('fingerprint') != candidate.get('fingerprint'):
        _CTX_BASELINE_SHARED = candidate
    return _CTX_BASELINE_SHARED

def get_run_settings(*, n_runs: int=N_RUNS_TABLE, base_seed: int=BASE_SEED, num_iter: int=NUM_ITER, n_particles: int=N_PARTICLES) -> dict[str, Any]:
    return dict(n_runs=int(n_runs), base_seed=int(base_seed), seeds=make_seed_list(n_runs=n_runs, base_seed=base_seed), num_iter=int(num_iter), n_particles=int(n_particles), snapshot_iters=tuple(DEFAULT_SNAPSHOT_ITERS), environment_source='environment.dem_terrain')

def get_common_pso_params(ctx: dict[str, Any] | None=None) -> dict[str, Any]:
    if ctx is None:
        ctx = get_ctx_shared(reset=False)
    return dict(ctx['pso_params'])

def get_common_pso_full_params(ctx: dict[str, Any] | None=None) -> dict[str, Any]:
    if ctx is None:
        ctx = get_ctx_shared(reset=False)
    params = dict(ctx['pso_params'])
    params['st_x'] = ctx['st_x']
    params['st_v'] = ctx['st_v']
    params['num_iter'] = int(ctx['num_iter'])
    return params

def get_algorithm_common_kwargs(ctx: dict[str, Any] | None=None) -> dict[str, Any]:
    if ctx is None:
        ctx = get_ctx_shared(reset=False)
    p = get_common_pso_full_params(ctx)
    return dict(n_dims=int(ctx['N_DIMS']), n_particles=int(p['n_particles']), st_x=p['st_x'], st_v=p['st_v'], w=float(p['w']), c1=float(p['c1']), c2=float(p['c2']), num_iter=int(p['num_iter']), func=ctx['cost'])

def main() -> None:
    ctx = get_ctx_shared(reset=True)
    print('[experiment_protocol]')
    print(f"  environment  = {ctx['environment_source']}")
    print(f"  N_DIMS       = {ctx['N_DIMS']}")
    print(f"  K_CAP        = {ctx['K_CAP']}")
    print(f"  n_particles  = {ctx['n_particles']}")
    print(f"  num_iter     = {ctx['num_iter']}")
    print(f"  tau          = {ctx['tau']}")
    print(f"  st_x         = {ctx['st_x']}")
    print(f"  st_v         = {ctx['st_v']}")
    print(f'  seeds[:5]    = {make_seed_list(5)}')
    print('  ctx cfg      =', ctx['cfg'])
    print('  SAFETY_MARGIN =', ctx['cfg'].get('SAFETY_MARGIN'))
    print('  OBSTACLE_RADIUS_EXPAND =', ctx['cfg'].get('OBSTACLE_RADIUS_EXPAND'))
    rng = np.random.default_rng(BASE_SEED)
    st_x = ctx['st_x']
    xg = np.empty((ctx['K_CAP'], ctx['N_GROUP']), dtype=float)
    xg[:, 0] = rng.uniform(st_x[0][0], st_x[0][1], size=ctx['K_CAP'])
    xg[:, 1] = rng.uniform(st_x[1][0], st_x[1][1], size=ctx['K_CAP'])
    xg[:, 2] = rng.uniform(st_x[2][0], st_x[2][1], size=ctx['K_CAP'])
    xg[:, 3] = rng.uniform(st_x[3][0], st_x[3][1], size=ctx['K_CAP'])
    xg = xg.reshape(-1)
    fit = float(ctx['cost'](xg))
    path, k_eff = ctx['decode_solution'](xg)
    print(f'  smoke_cost   = {fit:.6f}')
    print(f'  path_shape   = {path.shape}')
    print(f'  K_eff        = {k_eff}')
if __name__ == '__main__':
    main()
