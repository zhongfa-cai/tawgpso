"""Run one reproducible TAWG-PSO path-planning trial on medium_1."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SEED = 2026001
DEFAULT_PARTICLES = 50
DEFAULT_ITERATIONS = 400
DEFAULT_EVAL_BUDGET = 20100


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def run(seed: int, particles: int, iterations: int, eval_budget: int, save_dir: Path | None, show: bool) -> dict[str, Any]:
    os.environ["OBSTACLE_SET"] = "medium_1"
    from algorithm.tawg_pso import solve_tawgpso_path_once
    from configuration.protocol import get_ctx_shared
    from evaluation.energy import compute_energy_metrics
    from evaluation.metrics import print_single_run_metrics
    from evaluation.objectives import EvaluationBudget, compute_path_diagnostics

    ctx = get_ctx_shared(
        reset=True,
        pso_overrides={"n_particles": int(particles), "num_iter": int(iterations)},
    )
    budget = EvaluationBudget(ctx["cost"], int(eval_budget))
    ctx["cost"] = budget
    ctx["set_cost_iteration"] = budget.set_iteration
    ctx["eval_budget"] = int(eval_budget)
    result = solve_tawgpso_path_once(
        ctx=ctx,
        seed=int(seed),
        rng=np.random.default_rng(int(seed)),
        num_iter=int(iterations),
        n_particles=int(particles),
        plot=True,
        plot_save_dir=None if save_dir is None else str(save_dir),
        plot_show=bool(show),
        print_metrics=False,
    )
    result["meta"].update(budget.metadata(result["meta"].get("evals_used")))
    result["evals_used"] = int(budget.n_evals)
    record = print_single_run_metrics("TAWG-PSO / medium_1", result, ctx, recompute_fitness=False)
    path = np.asarray(result["path"], dtype=float)
    energy = compute_energy_metrics(path, k_eff=int(record["K_eff"]))
    diagnostics = compute_path_diagnostics(path, ctx)
    cost_output = ctx["cost"].evaluate_path(path, return_detail=True)
    cost_detail = dict(cost_output[1]) if isinstance(cost_output, tuple) else {}

    summary = {
        "environment": "medium_1",
        "seed": int(seed),
        "n_particles": int(particles),
        "num_iter": int(iterations),
        "eval_budget": int(eval_budget),
        **record,
        "turn_angle_sum_deg": float(energy["turn_angle_sum_deg"]),
        "pitch_change_sum_deg": float(energy["pitch_change_sum_deg"]),
        "total_climb_m": float(energy["total_climb"]),
        "total_descent_m": float(energy["total_descent"]),
        **diagnostics,
        "fitness_components": cost_detail,
        "figures": result.get("figures", {}),
    }
    if save_dir is not None:
        import csv
        import json
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "metrics.json").write_text(json.dumps(_json_value(summary), indent=2) + "\n", encoding="utf-8")
        with (save_dir / "path.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("point", "x_m", "y_m", "z_m"))
            for index, point in enumerate(path):
                writer.writerow((index, float(point[0]), float(point[1]), float(point[2])))
        print(f"metrics={save_dir / 'metrics.json'}")
        print(f"path={save_dir / 'path.csv'}")
    print(
        f"min_clearance={diagnostics['min_clearance']:.6f} m, "
        f"min_agl={diagnostics['min_agl']:.6f} m, "
        f"max_agl={diagnostics['max_agl']:.6f} m, "
        f"collisions={diagnostics['collision_count']}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TAWG-PSO once on the frozen medium_1 task.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--particles", type=int, default=DEFAULT_PARTICLES)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--eval-budget", type=int, default=DEFAULT_EVAL_BUDGET)
    parser.add_argument("--save-dir", type=Path, default=None, help="Optionally save figures, metrics, and path data.")
    parser.add_argument("--no-show", action="store_true", help="Disable interactive figure windows.")
    args = parser.parse_args()
    run(args.seed, args.particles, args.iterations, args.eval_budget, args.save_dir, not bool(args.no_show))


if __name__ == "__main__":
    main()
