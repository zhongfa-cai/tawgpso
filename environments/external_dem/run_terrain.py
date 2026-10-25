"""Render CI-EXT or MSH-EXT in native metric coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from environments.external_dem.external_dem import load_external_scene, scene_audit_payload


def render(scene_name: str, save_path: Path | None, show: bool) -> None:
    scene = load_external_scene(scene_name)
    terrain = scene.terrain
    logical_rows = int(round(scene.y_max_m / terrain.resolution_m)) + 1
    logical_cols = int(round(scene.x_max_m / terrain.resolution_m)) + 1
    stride = max(1, int(np.ceil(max(logical_rows, logical_cols) / 1400)))
    array_view = terrain.array[:logical_rows:stride, :logical_cols:stride]
    mask_view = terrain.mask[:logical_rows:stride, :logical_cols:stride]
    display = np.where(mask_view, array_view, np.nan)

    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=130)
    image = ax.imshow(
        display,
        extent=(0.0, scene.x_max_m, 0.0, scene.y_max_m),
        origin="lower",
        cmap="terrain",
        interpolation="nearest",
        aspect="equal",
    )
    for cx, cy, radius, _height, _kind in scene.obstacles_physical:
        ax.add_patch(Circle((cx, cy), radius, fill=False, color="red", linewidth=1.0))
    ax.plot(scene.start_physical[0], scene.start_physical[1], "ks", label="Start")
    ax.plot(scene.goal_physical[0], scene.goal_physical[1], "k*", markersize=10, label="Goal")
    ax.set(xlabel="x [m]", ylabel="y [m]", title=f"{scene.scene_id} native-metric DEM")
    ax.legend(loc="best")
    fig.colorbar(image, ax=ax, label="Elevation [m]")
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    print(json.dumps(scene_audit_payload(scene), indent=2))
    if save_path is not None:
        print(save_path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a frozen external DEM scene.")
    parser.add_argument("--scene", choices=("ci_ext", "msh_ext"), default="ci_ext")
    parser.add_argument("--save", type=Path, default=None, help="Optionally save the displayed terrain figure.")
    parser.add_argument("--no-show", action="store_true", help="Disable the interactive figure window.")
    args = parser.parse_args()
    render(args.scene, args.save, not bool(args.no_show))


if __name__ == "__main__":
    main()
