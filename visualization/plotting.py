"""Minimal plotting support for the reproducible medium-environment entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_result_bundle(
    res: dict[str, Any],
    ctx: dict[str, Any],
    save_dir: str | Path | None = None,
    prefix: str = "TAWG-PSO",
    show: bool = True,
    draw_terrain: bool = True,
    draw_obstacles: bool = True,
    y_log: bool = False,
    dpi: int = 300,
) -> dict[str, str]:
    del draw_obstacles
    output = None if save_dir is None else Path(save_dir)
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
    path = np.asarray(res["path"], dtype=float)
    figures: dict[str, str] = {}

    path_fig = plt.figure(figsize=(8.0, 6.2))
    ax = path_fig.add_subplot(111, projection="3d")
    if draw_terrain:
        X = np.asarray(ctx["X"])
        Y = np.asarray(ctx["Y"])
        Z = np.asarray(ctx["Z"])
        stride = max(1, int(np.ceil(max(Z.shape) / 180)))
        ax.plot_surface(X[::stride, ::stride], Y[::stride, ::stride], Z[::stride, ::stride], cmap="terrain", alpha=0.72, linewidth=0)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], "r-o", linewidth=2.0, markersize=3.2, label="TAWG-PSO")
    ax.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]", title="TAWG-PSO path in medium_1")
    ax.legend(loc="best")
    path_fig.tight_layout()
    if output is not None:
        path_figure = output / f"{prefix.lower()}_path.png"
        path_fig.savefig(path_figure, dpi=dpi, bbox_inches="tight")
        figures["path"] = str(path_figure)

    convergence_fig, ax = plt.subplots(figsize=(7.2, 4.6))
    history = np.asarray(res.get("history", []), dtype=float)
    ax.plot(np.arange(1, history.size + 1), history, color="#1f77b4", linewidth=1.5)
    if y_log and np.all(history > 0):
        ax.set_yscale("log")
    ax.set(xlabel="Iteration", ylabel="Best fitness", title="TAWG-PSO convergence")
    ax.grid(alpha=0.25)
    convergence_fig.tight_layout()
    if output is not None:
        convergence_figure = output / f"{prefix.lower()}_convergence.png"
        convergence_fig.savefig(convergence_figure, dpi=dpi, bbox_inches="tight")
        figures["convergence"] = str(convergence_figure)
    if show:
        plt.show()
    plt.close(path_fig)
    plt.close(convergence_fig)
    return figures
