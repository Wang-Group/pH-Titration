"""Replay the archived local-response study using its exact locked tasks.

No training. The supplied diagnostic and PF source are unchanged; an added
snapshot wrapper records the solution state needed to recalculate the curves.
"""
from __future__ import annotations

import os
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BLOCK = ROOT / "evidence/simulation_numerical_evidence_20260823/06_POSTERIOR_RECOVERY/local_response_reproduction_20260906"
PACKAGE = BLOCK / "original_package"
RESULTS = PACKAGE / "formal_results/new_pf_local_response"
SEEDS = (101, 202, 303, 404, 555)
_diagnostic = None


def diagnostic():
    global _diagnostic
    if _diagnostic is None:
        # The ZIP omitted reference/original_bayesian_controller.py. Use only
        # the explicitly archived dependency, never the current training API.
        sys.path.insert(0, str(BLOCK / "dependency"))
        spec = importlib.util.spec_from_file_location(
            "archived_local_response", PACKAGE / "pf_local_response_diagnostics.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        original_snapshot = module.snapshot

        def snapshot_with_state(task, controller, observations, checkpoint_type):
            row = original_snapshot(task, controller, observations, checkpoint_type)
            row.update(state_total_volume_ml=float(controller.total_volume),
                       state_base_moles=float(controller.base_added_moles),
                       state_acid_moles=float(controller.acid_added_moles),
                       controller_steps=int(controller.steps_taken))
            return row

        module.snapshot = snapshot_with_state
        _diagnostic = module
    return _diagnostic


def replay_task(payload):
    benchmark_seed, record = payload
    module = diagnostic()
    record = dict(record)
    record["pka_values"] = tuple(record["pka_values"])
    rows = module.run_task((module.ControlTask(**record), 1000, benchmark_seed))
    for row in rows:
        row["benchmark_seed"] = benchmark_seed
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-tasks-per-seed", type=int, default=300,
                        help="Values below 300 are a technical smoke test on a locked subset")
    args = parser.parse_args()
    if args.workers < 1 or not 1 <= args.limit_tasks_per_seed <= 300:
        parser.error("workers must be positive and limit-tasks-per-seed must be in 1..300")
    from scripts.audit_pf_local_curves import verify_files
    verify_files()
    output = args.output.resolve()
    if output.is_relative_to((ROOT / "evidence").resolve()):
        parser.error("Write new results outside the archived evidence directory")
    output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for seed in SEEDS:
        tasks = [json.loads(line) for line in
                 (RESULTS / f"seed_{seed}_tasks.jsonl").read_text().splitlines() if line.strip()]
        jobs.extend((seed, task) for task in tasks[:args.limit_tasks_per_seed])
    module = diagnostic()
    all_rows = []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, rows in enumerate(pool.map(replay_task, jobs, chunksize=2), 1):
            all_rows.extend(rows)
            if index % 50 == 0 or index == len(jobs):
                print(f"Local-response replay: {index}/{len(jobs)} tasks; "
                      f"elapsed {time.monotonic() - started:.1f}s", flush=True)
    per_seed, aggregate = module.summarize(all_rows, list(SEEDS))
    module.write_csv(output / "all_local_response_rows.csv", all_rows)
    module.write_csv(output / "per_seed_summary.csv", per_seed)
    module.write_csv(output / "aggregate_summary.csv", aggregate)
    config = {
        "status": "COMPLETE", "tasks": len(jobs), "snapshots": len(all_rows),
        "scope": "full study" if args.limit_tasks_per_seed == 300 else "smoke test only",
        "tasks_per_seed": args.limit_tasks_per_seed, "seeds": list(SEEDS),
        "particles": 1000, "workers": args.workers, "python": platform.python_version(),
        "numpy": module.np.__version__, "elapsed_seconds": time.monotonic() - started,
        "windows_ml": list(module.WINDOWS_ML), "grid_points_per_window": 81,
        "replay_source": "unchanged supplied run_task; additional state columns only",
    }
    (output / "REPLAY_CONFIG.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    from scripts.audit_pf_local_curves import compare_replay
    report = compare_replay(output)
    (output / "REPLAY_AUDIT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
