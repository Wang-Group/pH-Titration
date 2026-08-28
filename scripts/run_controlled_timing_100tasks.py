from __future__ import annotations

"""Launch the controlled 100-task timing workers in isolated processes."""

import argparse
import csv
import ctypes
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "benchmark_controlled_observation_to_action_100tasks.py"
DEFAULT_OUTPUT = (
    ROOT
    / "evidence"
    / "simulation_numerical_evidence_20260823"
    / "01_PRIMARY_5x3000_BENCHMARK"
    / "formal_matched_evaluation"
    / "controlled_timing_100tasks_20260828"
)
METHODS = ("imitation", "ppo", "pf_1000", "pf_10000", "pf_100000", "pymc")


class FileTime(ctypes.Structure):
    _fields_ = (("low", ctypes.c_uint32), ("high", ctypes.c_uint32))


def filetime_value(value: FileTime) -> int:
    return (int(value.high) << 32) | int(value.low)


def sample_windows_cpu_utilization(interval_seconds: float = 1.0) -> float | None:
    if sys.platform != "win32":
        return None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_system_times = kernel32.GetSystemTimes
    get_system_times.argtypes = (
        ctypes.POINTER(FileTime),
        ctypes.POINTER(FileTime),
        ctypes.POINTER(FileTime),
    )
    get_system_times.restype = ctypes.c_int

    def snapshot() -> tuple[int, int, int]:
        idle, kernel, user = FileTime(), FileTime(), FileTime()
        if not get_system_times(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            raise OSError(ctypes.get_last_error(), "GetSystemTimes failed")
        return filetime_value(idle), filetime_value(kernel), filetime_value(user)

    before = snapshot()
    time.sleep(interval_seconds)
    after = snapshot()
    idle_delta = after[0] - before[0]
    total_delta = (after[1] - before[1]) + (after[2] - before[2])
    if total_delta <= 0:
        return None
    return 100.0 * (1.0 - idle_delta / total_delta)


def wait_for_idle(threshold_percent: float, consecutive_samples: int = 3) -> list[float]:
    samples: list[float] = []
    consecutive = 0
    while len(samples) < 60:
        utilization = sample_windows_cpu_utilization()
        if utilization is None:
            return samples
        samples.append(utilization)
        consecutive = consecutive + 1 if utilization <= threshold_percent else 0
        if consecutive >= consecutive_samples:
            return samples
        print(
            f"Waiting for system idle: CPU utilization {utilization:.1f}% "
            f"(target <= {threshold_percent:.1f}%)",
            flush=True,
        )
    raise RuntimeError(
        f"System did not remain below {threshold_percent:.1f}% CPU utilization "
        "for three consecutive samples within 60 seconds"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cpu-index", type=int, default=2)
    parser.add_argument("--idle-threshold-percent", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--draws", type=int, default=300)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument(
        "--pymc-python",
        type=Path,
        default=Path(
            "C:/Users/ZSY/Desktop/FDTD/joint_parameter_bayesian_processed_20260811/"
            ".venv/Scripts/python.exe"
        ),
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if "pymc" in args.methods and not args.pymc_python.is_file():
        raise FileNotFoundError(args.pymc_python)

    environment = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        environment[variable] = "1"
    environment["PYTHONHASHSEED"] = str(args.seed)

    executions: list[dict] = []
    for method in args.methods:
        idle_samples = wait_for_idle(args.idle_threshold_percent)
        method_output = output / method
        executable = args.pymc_python.resolve() if method == "pymc" else Path(sys.executable).resolve()
        command = [
            str(executable),
            str(WORKER),
            "--method",
            method,
            "--output-dir",
            str(method_output),
            "--repeats",
            str(args.repeats),
            "--cpu-index",
            str(args.cpu_index),
            "--seed",
            str(args.seed),
            "--draws",
            str(args.draws),
            "--chains",
            str(args.chains),
        ]
        print(f"Launching controlled worker: {method}", flush=True)
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            check=True,
            stdout=None,
            stderr=None,
        )
        executions.append(
            {
                "method": method,
                "python_executable": str(executable),
                "returncode": completed.returncode,
                "prelaunch_cpu_utilization_percent": idle_samples,
            }
        )

    summaries = []
    task_keys: set[tuple[int, int]] | None = None
    for method in args.methods:
        rows = read_csv(output / method / "raw.csv")
        keys = {(int(row["benchmark_seed"]), int(row["task_id"])) for row in rows}
        if len(keys) != 100:
            raise RuntimeError(f"{method}: expected 100 task keys, found {len(keys)}")
        if task_keys is None:
            task_keys = keys
        elif keys != task_keys:
            raise RuntimeError(f"{method}: task keys do not match the other methods")
        summary = read_csv(output / method / "summary.csv")[0]
        summaries.append(summary)

    write_csv(output / "CONTROLLED_RESULT_SUMMARY.csv", summaries)
    config = {
        "study_id": "controlled_observation_to_action_timing_100tasks",
        "methods": list(args.methods),
        "unique_task_cases": len(task_keys or ()),
        "repeats_per_task": args.repeats,
        "formal_measurements_per_method": 100 * args.repeats,
        "cpu_index": args.cpu_index,
        "seed": args.seed,
        "idle_threshold_percent": args.idle_threshold_percent,
        "draws_per_k": args.draws,
        "chains": args.chains,
        "launcher_python": sys.executable,
        "pymc_python": str(args.pymc_python.resolve()),
        "platform": platform.platform(),
        "active_power_scheme": "not recorded by launcher",
        "worker_sha256": sha256(WORKER),
        "executions": executions,
    }
    (output / "CONTROLLED_RUN_CONFIG.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
