"""Christmas-Island-based adapted S1-S6 terrain and threat-layout definitions."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

import numpy as np
from PIL import Image

BaseItem = tuple[tuple[float, float], float, float]

@dataclass(frozen=True)
class ThreatSpec:
    center: tuple[float, float]
    radius: float
    height: float = 160.0

    def as_base_item(self) -> BaseItem:
        return (self.center, float(self.radius), float(self.height))
THREATS: Final[dict[int, ThreatSpec]] = {1: ThreatSpec(center=(132.0, 168.0), radius=13.5), 2: ThreatSpec(center=(142.0, 124.0), radius=13.0), 3: ThreatSpec(center=(80.0, 111.0), radius=14.0), 4: ThreatSpec(center=(100.0, 78.0), radius=14.5), 5: ThreatSpec(center=(70.0, 67.0), radius=13.0), 6: ThreatSpec(center=(120.0, 45.0), radius=13.5)}
SCENARIO_COMBINATIONS: Final[dict[str, tuple[int, ...]]] = {'s1': (1, 2, 3, 4, 5, 6), 's2': (1, 2, 4, 5, 6), 's3': (1, 2, 3, 4, 6), 's4': (3, 4, 5, 6), 's5': (3, 4, 5), 's6': (3, 4)}
SCENARIO_LABELS: Final[dict[str, str]] = {'s1': 'six-threat corridor scenario', 's2': 'five-threat right-corridor scenario', 's3': 'five-threat upper-corridor scenario', 's4': 'four-threat lower-corridor scenario', 's5': 'three-threat central scenario', 's6': 'two-threat baseline scenario'}

def normalize_scenario_name(name: str) -> str:
    key = str(name).strip().lower().replace('-', '_')
    if key.startswith('scenario_'):
        key = key.removeprefix('scenario_')
    elif key.startswith('scenario') and key.removeprefix('scenario').isdigit():
        key = key.removeprefix('scenario')
    if key.isdigit():
        key = f's{key}'
    return key

def build_base(name: str) -> list[BaseItem]:
    key = normalize_scenario_name(name)
    if key not in SCENARIO_COMBINATIONS:
        valid = ', '.join(sorted(SCENARIO_COMBINATIONS))
        raise KeyError(f'Unknown scenario {name!r}. Expected one of: {valid}.')
    return [THREATS[i].as_base_item() for i in SCENARIO_COMBINATIONS[key]]

def describe(name: str) -> dict[str, object]:
    key = normalize_scenario_name(name)
    ids = SCENARIO_COMBINATIONS[key]
    return {'name': key, 'label': SCENARIO_LABELS[key], 'threat_ids': ids, 'n_threats': len(ids), 'base': build_base(key)}
GRID_LIMIT = 200.0
GRID_STEP = 0.5
TERRAIN_Z_MIN = 0.0
TERRAIN_Z_MAX = 60.0
START = np.asarray([40.0, 22.0, 80.0], dtype=float)
GOAL = np.asarray([160.0, 178.0, 100.0], dtype=float)
ASSET_DIR = Path(__file__).with_name('assets')

def scenario_names() -> tuple[str, ...]:
    return tuple((f's{i}' for i in range(1, 7)))

def _axis() -> np.ndarray:
    n = int(round(GRID_LIMIT / GRID_STEP))
    return np.linspace(0.0, GRID_LIMIT, n + 1, dtype=np.float64)

def _clean_terrain_array(arr: np.ndarray) -> np.ndarray:
    H = np.asarray(arr, dtype=np.float64)
    H[~np.isfinite(H)] = 0.0
    H[H < 0.0] = 0.0
    return H

def _normalize_terrain(H: np.ndarray) -> np.ndarray:
    valid = H[np.isfinite(H) & (H > 0.0)]
    if valid.size == 0:
        return np.zeros_like(H, dtype=np.float64)
    lo, hi = np.percentile(valid, [2.0, 98.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(valid))
        hi = float(np.max(valid))
    if hi <= lo:
        return np.zeros_like(H, dtype=np.float64)
    Z = np.clip((H - lo) / (hi - lo), 0.0, 1.0)
    return TERRAIN_Z_MIN + Z * (TERRAIN_Z_MAX - TERRAIN_Z_MIN)

def _resize_float_image(H: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.asarray(H, dtype=np.float32), mode='F')
    resized = image.resize((int(shape[1]), int(shape[0])), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float64)

@lru_cache(maxsize=6)
def load_scenario_dem(scenario_id: str='s1') -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scenario_name = normalize_scenario_name(scenario_id)
    terrain_file = ASSET_DIR / f'{scenario_name}.tif'
    if not terrain_file.exists():
        raise FileNotFoundError(f'Missing scenario terrain asset: {terrain_file}.')
    axis = _axis()
    X, Y = np.meshgrid(axis, axis)
    with Image.open(terrain_file) as img:
        H_raw = _clean_terrain_array(np.asarray(img))
    H_resized = _resize_float_image(H_raw, X.shape)
    Z = _normalize_terrain(H_resized)
    return (X, Y, Z, START.copy(), GOAL.copy())

def load_scenario_obstacles(scenario_id: str, *, radius_expand: float=0.0) -> list[tuple[float, float, float, float, str]]:
    scenario_name = normalize_scenario_name(scenario_id)
    expand = float(radius_expand)
    obstacles: list[tuple[float, float, float, float, str]] = []
    for center, radius, height in build_base(scenario_name):
        cx, cy = center
        obstacles.append((float(cx), float(cy), max(0.0, float(radius) + expand), float(height), 'cylinder'))
    return obstacles

def load_scenario(scenario_id: str, *, radius_expand: float=0.0) -> dict[str, Any]:
    scenario_name = normalize_scenario_name(scenario_id)
    X, Y, Z, start, goal = load_scenario_dem(scenario_name)
    return {'scenario': scenario_name, 'meta': describe(scenario_name), 'X': X, 'Y': Y, 'Z': Z, 'START': start, 'GOAL': goal, 'obstacles': load_scenario_obstacles(scenario_name, radius_expand=radius_expand)}

def plot_scenario_topview(scenario_id: str, *, save_path: str | Path | None=None, show: bool=False, dpi: int=300) -> str | None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    scene = load_scenario(scenario_id)
    X = scene['X']
    Y = scene['Y']
    Z = scene['Z']
    start = scene['START']
    goal = scene['GOAL']
    obstacles = scene['obstacles']
    fig, ax = plt.subplots(figsize=(6.2, 5.4), dpi=120)
    extent = [float(X.min()), float(X.max()), float(Y.min()), float(Y.max())]
    ax.imshow(Z, extent=extent, origin='lower', cmap='summer', interpolation='bilinear')
    for cx, cy, radius, _height, _kind in obstacles:
        for scale in (1.0, 0.72, 0.44):
            ax.add_patch(Circle((float(cx), float(cy)), float(radius) * scale, fill=False, edgecolor='black', linewidth=0.9))
        ax.plot(float(cx), float(cy), 'ko', markersize=2.8)
    ax.plot(float(start[0]), float(start[1]), 'ks', markersize=5.5)
    ax.plot(float(goal[0]), float(goal[1]), 'ko', markersize=5.5)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(str(scene['meta']['label']))
    fig.tight_layout()
    saved = None
    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=int(dpi), bbox_inches='tight')
        saved = str(out)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return saved

def save_all_scenario_previews(output_dir: str | Path='outputs/scenario_preview', *, show: bool=False, dpi: int=300) -> list[str]:
    out_dir = Path(output_dir)
    saved: list[str] = []
    for name in scenario_names():
        path = out_dir / f'{name}_topview.png'
        saved_path = plot_scenario_topview(name, save_path=path, show=show, dpi=dpi)
        if saved_path is not None:
            print(saved_path)
            saved.append(saved_path)
    return saved

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description='Render S1-S6 scenario top-view figures.')
    parser.add_argument('--scenario', default='s1', help='Scenario name: s1-s6, or all.')
    parser.add_argument('--save-dir', type=Path, default=None)
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    names = scenario_names() if str(args.scenario).lower() == 'all' else (normalize_scenario_name(args.scenario),)
    for name in names:
        out = None if args.save_dir is None else args.save_dir / f'{name}_topview.png'
        saved = plot_scenario_topview(name, save_path=out, show=not bool(args.no_show), dpi=int(args.dpi))
        if saved is not None:
            print(saved)
if __name__ == '__main__':
    main()
