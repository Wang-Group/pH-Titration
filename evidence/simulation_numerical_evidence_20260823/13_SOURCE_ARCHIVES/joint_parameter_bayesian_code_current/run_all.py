from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


PROFILES = {
    "quick": {
        "seeds": [101, 202],
        "nominal_tasks": 30,
        "variable_control_tasks": 20,
        "curve_tasks": 10,
        "pymc_tasks": 1,
        "draws": 20,
        "chains": 1,
        "particles": 200,
        "control_workers": 4,
        "curve_workers": 2,
        "pymc_workers": 2,
    },
    "standard": {
        "seeds": [101, 202, 303, 404, 555],
        "nominal_tasks": 500,
        "variable_control_tasks": 200,
        "curve_tasks": 100,
        "pymc_tasks": 1,
        "draws": 100,
        "chains": 1,
        "particles": 1000,
        "control_workers": 10,
        "curve_workers": 5,
        "pymc_workers": 5,
    },
    "full": {
        "seeds": [101, 202, 303, 404, 555],
        "nominal_tasks": 3000,
        "variable_control_tasks": 1000,
        "curve_tasks": 300,
        "pymc_tasks": 3,
        "draws": 300,
        "chains": 1,
        "particles": 1000,
        "control_workers": 10,
        "curve_workers": 5,
        "pymc_workers": 5,
    },
}


def run(command, cwd, log_path):
    print("\nRUN:", " ".join(str(item) for item in command), flush=True)
    started = time.time()
    status = "PASS"
    return_code = 0
    try:
        subprocess.run(command, cwd=cwd, check=True)
    except subprocess.CalledProcessError as exc:
        status = "FAIL"
        return_code = int(exc.returncode)
        raise
    finally:
        record = {
            "command": [str(item) for item in command],
            "status": status,
            "return_code": return_code,
            "started_unix": started,
            "elapsed_seconds": time.time() - started,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser(description="One-click joint Bayesian comparison runner")
    parser.add_argument("--profile", choices=PROFILES, default="standard")
    parser.add_argument("--skip-pymc", action="store_true")
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    settings = PROFILES[args.profile]
    seeds = [str(value) for value in settings["seeds"]]
    run_dir = base / "results" / args.profile
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "MASTER_RUN_LOG.jsonl"

    common = ["--seeds", *seeds, "--particles", str(settings["particles"])]
    run(
        [
            sys.executable,
            "run_pf_multiseed_control.py",
            *common,
            "--tasks-per-seed",
            str(settings["nominal_tasks"]),
            "--workers",
            str(settings["control_workers"]),
            "--distribution",
            "nominal",
            "--output-dir",
            str(run_dir / "pf_control_nominal"),
        ],
        base,
        log_path,
    )
    run(
        [
            sys.executable,
            "run_pf_multiseed_control.py",
            *common,
            "--tasks-per-seed",
            str(settings["variable_control_tasks"]),
            "--workers",
            str(settings["control_workers"]),
            "--distribution",
            "variable_concentration",
            "--output-dir",
            str(run_dir / "pf_control_variable_concentration"),
        ],
        base,
        log_path,
    )
    run(
        [
            sys.executable,
            "run_pf_curve_recovery.py",
            *common,
            "--tasks-per-seed",
            str(settings["curve_tasks"]),
            "--workers",
            str(settings["curve_workers"]),
            "--output-dir",
            str(run_dir / "pf_curve_recovery"),
        ],
        base,
        log_path,
    )
    if not args.skip_pymc:
        run(
            [
                sys.executable,
                "run_pymc_comparison.py",
                "--seeds",
                *seeds,
                "--particles",
                str(settings["particles"]),
                "--tasks-per-seed",
                str(settings["pymc_tasks"]),
                "--draws",
                str(settings["draws"]),
                "--chains",
                str(settings["chains"]),
                "--workers",
                str(settings["pymc_workers"]),
                "--output-dir",
                str(run_dir / "pymc_comparison"),
            ],
            base,
            log_path,
        )
    run([sys.executable, "build_master_report.py", "--run-dir", str(run_dir)], base, log_path)
    print(f"\nFinished. Open: {run_dir / 'MASTER_RESULTS_SUMMARY.md'}", flush=True)


if __name__ == "__main__":
    main()
