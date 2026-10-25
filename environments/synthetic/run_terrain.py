"""Render one of the three deterministic synthetic environments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from environments.synthetic.terrain import load_dem, render_dem_3d


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a synthetic TAWG-PSO terrain.")
    parser.add_argument("--environment", choices=("sparse_1", "medium_1", "dense_1"), default="medium_1")
    parser.add_argument("--save", type=Path, default=None, help="Optionally save the displayed terrain figure.")
    parser.add_argument("--no-show", action="store_true", help="Disable the interactive figure window.")
    args = parser.parse_args()

    os.environ["OBSTACLE_SET"] = args.environment
    X, Y, Z, start, goal = load_dem(clearance=5.0)
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
    render_dem_3d(
        X,
        Y,
        Z,
        start_xy=(float(start[0]), float(start[1])),
        start_z=float(start[2]),
        goal_xy=(float(goal[0]), float(goal[1])),
        goal_z=float(goal[2]),
        save_path=None if args.save is None else str(args.save),
        show=not bool(args.no_show),
    )
    print(f"environment={args.environment} shape={Z.shape} z_range=({Z.min():.6f}, {Z.max():.6f})")
    if args.save is not None:
        print(args.save.resolve())


if __name__ == "__main__":
    main()
