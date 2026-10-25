"""Render one or all frozen S1-S6 scenario environments."""

from __future__ import annotations

import argparse
from pathlib import Path

from environments.scenarios.scenes import normalize_scenario_name, plot_scenario_topview, scenario_names


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the S1-S6 scenario terrains.")
    parser.add_argument("--scenario", default="s1", help="s1 through s6, or all")
    parser.add_argument("--save-dir", type=Path, default=None, help="Optionally save the displayed terrain figures.")
    parser.add_argument("--no-show", action="store_true", help="Disable interactive figure windows.")
    args = parser.parse_args()

    names = scenario_names() if args.scenario.lower() == "all" else (normalize_scenario_name(args.scenario),)
    paths = []
    for name in names:
        path = None if args.save_dir is None else args.save_dir / f"{name}_topview.png"
        saved = plot_scenario_topview(name, save_path=path, show=not bool(args.no_show))
        if saved is not None:
            paths.append(saved)
    for path in paths:
        print(Path(path).resolve())


if __name__ == "__main__":
    main()
