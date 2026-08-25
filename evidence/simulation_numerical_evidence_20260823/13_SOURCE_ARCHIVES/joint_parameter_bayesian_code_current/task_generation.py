from __future__ import annotations

from dataclasses import replace

import numpy as np

from benchmark_core import StressScenario, generate_tasks, solve_ph


DEFAULT_SEEDS = [101, 202, 303, 404, 555]


def generate_comparison_tasks(
    seed: int,
    count: int,
    distribution: str = "nominal",
    minimum_initial_error_ph: float = 0.0,
):
    tasks = generate_tasks(seed, count, StressScenario("nominal"))
    rng = np.random.default_rng(seed + 8_104_729)
    output = []
    for task in tasks:
        if distribution == "nominal":
            concentration = 0.1
        elif distribution == "variable_concentration":
            concentration = float(np.exp(rng.uniform(np.log(0.03), np.log(0.25))))
        else:
            raise ValueError(f"Unknown task distribution: {distribution}")
        initial_ph = float(
            np.round(
                solve_ph(
                    task.pka_values,
                    task.initial_volume_ml,
                    concentration,
                ),
                2,
            )
        )
        target = float(task.target_ph)
        if minimum_initial_error_ph > 0.0:
            for _ in range(100):
                if abs(target - initial_ph) >= minimum_initial_error_ph:
                    break
                target = float(np.round(rng.uniform(2.0, 11.0), 2))
        output.append(
            replace(
                task,
                analyte_conc_m=concentration,
                initial_ph=initial_ph,
                target_ph=target,
            )
        )
    return output
