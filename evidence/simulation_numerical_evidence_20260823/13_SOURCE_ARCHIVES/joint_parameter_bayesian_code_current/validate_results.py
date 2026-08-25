from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


SEEDS = (101, 202, 303, 404, 555)
PF_METHODS = (
    "pf_pka_only_k3",
    "pf_pka_conc_k3",
    "pf_pka_conc_variable_k",
)
PYMC_METHODS = (
    "pymc_pka_only_k3",
    "pymc_pka_conc_k3",
    "pymc_pka_conc_variable_k",
)
CHECKPOINTS = ("after_step4", "after_step8", "last_decision", "final")
PROFILES = {
    "quick": {"seeds": (101, 202), "nominal": 30, "variable": 20, "curve": 10, "pymc": 1, "particles": 200, "draws": 20},
    "standard": {"seeds": SEEDS, "nominal": 500, "variable": 200, "curve": 100, "pymc": 1, "particles": 1000, "draws": 100},
    "full": {"seeds": SEEDS, "nominal": 3000, "variable": 1000, "curve": 300, "pymc": 3, "particles": 1000, "draws": 300},
}


class Validator:
    def __init__(self, run_dir: Path, profile: str):
        self.run_dir = run_dir
        self.profile = profile
        self.spec = PROFILES[profile]
        self.seeds = self.spec["seeds"]
        self.errors: list[str] = []
        self.checks: list[dict] = []

    def check(self, condition: bool, name: str, detail: str = ""):
        status = "PASS" if condition else "FAIL"
        self.checks.append({"name": name, "status": status, "detail": detail})
        if not condition:
            self.errors.append(f"{name}: {detail}" if detail else name)

    def read_json(self, path: Path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.check(False, f"read {path.name}", str(exc))
            return {}

    def read_csv(self, path: Path):
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                return list(csv.DictReader(handle))
        except Exception as exc:
            self.check(False, f"read {path.name}", str(exc))
            return []

    @staticmethod
    def finite(row, fields):
        try:
            return all(math.isfinite(float(row[field])) for field in fields)
        except (KeyError, TypeError, ValueError):
            return False

    def validate_groups(self, rows, key_fields, expected_methods, label: str):
        groups = {}
        for row in rows:
            key = tuple(row[field] for field in key_fields)
            groups.setdefault(key, []).append(row)
        valid = True
        truth_fields = ("acid_type", "true_pair_count", "true_pkas", "true_concentration_m")
        for group in groups.values():
            if {row.get("method") for row in group} != set(expected_methods):
                valid = False
            for field in truth_fields:
                if len({row.get(field) for row in group}) != 1:
                    valid = False
        self.check(valid, f"{label} exact method sets and task truth pairing", f"groups={len(groups)}")

    def validate_settings(self, payload, tasks: int, particles: int, distribution: str, label: str):
        settings = payload.get("settings", {})
        self.check(settings.get("seeds") == list(self.seeds), f"{label} seeds", str(settings.get("seeds")))
        self.check(settings.get("tasks_per_seed") == tasks, f"{label} tasks/seed", str(settings.get("tasks_per_seed")))
        self.check(settings.get("particles") == particles, f"{label} particles", str(settings.get("particles")))
        self.check(settings.get("distribution") == distribution, f"{label} distribution", str(settings.get("distribution")))

    def validate_parameter_rows(self, rows, label: str):
        valid = True
        for row in rows:
            try:
                method = row["method"]
                pka = [float(value) for value in json.loads(row["estimated_pkas"])]
                probs = [float(value) for value in json.loads(row["pair_probabilities"])] if "pair_probabilities" in row else [float(row[f"pair_probability_k{k}"]) for k in (1, 2, 3)]
                estimated_k = int(row["estimated_pair_count"])
                concentration = float(row["estimated_concentration_m"])
                if not pka or len(pka) != estimated_k:
                    valid = False
                if any(not math.isfinite(value) or value < 1.5 - 1e-12 or value > 9.0 + 1e-12 for value in pka):
                    valid = False
                if any(pka[index] > pka[index + 1] for index in range(len(pka) - 1)):
                    valid = False
                if len(probs) != 3 or any(not math.isfinite(value) or value < -1e-12 for value in probs):
                    valid = False
                if not math.isclose(sum(probs), 1.0, rel_tol=0.0, abs_tol=1e-8):
                    valid = False
                if not math.isfinite(concentration) or concentration < 0.02 - 1e-12 or concentration > 0.30 + 1e-12:
                    valid = False
                if "pka_only_k3" in method and not math.isclose(concentration, 0.1, abs_tol=1e-10):
                    valid = False
                if method.endswith("k3") and "variable_k" not in method and estimated_k != 3:
                    valid = False
                if "variable_k" in method and estimated_k != 1 + max(range(3), key=probs.__getitem__):
                    valid = False
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                valid = False
        self.check(valid, f"{label} posterior bounds and probabilities", f"rows={len(rows)}")

    def validate_control(self, dirname: str, tasks: int, distribution: str):
        path = self.run_dir / dirname
        label = dirname
        payload = self.read_json(path / "summary.json")
        complete = self.read_json(path / "RUN_COMPLETE.json")
        expected = len(self.seeds) * tasks * len(PF_METHODS)
        expected_shards = len(self.seeds) * len(PF_METHODS)
        self.validate_settings(payload, tasks, self.spec["particles"], distribution, label)
        self.check(complete.get("status") == "PASS", f"{label} completion marker", str(complete))
        self.check(complete.get("task_rows") == expected, f"{label} marker row count", str(complete.get("task_rows")))
        self.check(complete.get("completed_shards") == expected_shards, f"{label} marker shard count", str(complete.get("completed_shards")))
        rows = self.read_csv(path / "pf_control_per_task.csv")
        self.check(len(rows) == expected, f"{label} CSV row count", f"{len(rows)} expected {expected}")
        counts = Counter((int(row["seed"]), int(row["task_id"])) for row in rows)
        expected_keys = {(seed, task_id) for seed in self.seeds for task_id in range(1, tasks + 1)}
        self.check(set(counts) == expected_keys and set(counts.values()) == {len(PF_METHODS)}, f"{label} matched task keys", f"keys={len(counts)}")
        method_counts = Counter(row.get("method") for row in rows)
        self.check(method_counts == Counter({method: len(self.seeds) * tasks for method in PF_METHODS}), f"{label} method balance", str(method_counts))
        self.validate_groups(rows, ("seed", "task_id"), PF_METHODS, label)
        numeric = ("true_concentration_m", "initial_ph", "target_ph", "final_ph", "steps", "overshoots", "decision_time_mean_ms", "estimated_concentration_m")
        self.check(all(self.finite(row, numeric) for row in rows), f"{label} finite task metrics", f"rows={len(rows)}")
        self.validate_parameter_rows(rows, label)

    def validate_curve(self):
        path = self.run_dir / "pf_curve_recovery"
        label = "pf_curve_recovery"
        tasks = self.spec["curve"]
        payload = self.read_json(path / "summary.json")
        complete = self.read_json(path / "RUN_COMPLETE.json")
        distribution = "variable_concentration (log-uniform 0.03-0.25 M)"
        self.validate_settings(payload, tasks, self.spec["particles"], distribution, label)
        expected = len(self.seeds) * tasks * len(PF_METHODS) * len(CHECKPOINTS)
        self.check(complete.get("status") == "PASS", f"{label} completion marker", str(complete))
        self.check(complete.get("checkpoint_rows") == expected, f"{label} marker row count", str(complete.get("checkpoint_rows")))
        self.check(complete.get("completed_shards") == len(self.seeds), f"{label} marker shard count", str(complete.get("completed_shards")))
        rows = self.read_csv(path / "pf_curve_recovery_per_task_checkpoint.csv")
        self.check(len(rows) == expected, f"{label} CSV row count", f"{len(rows)} expected {expected}")
        counts = Counter((int(row["seed"]), int(row["task_id"]), row["checkpoint"]) for row in rows)
        expected_keys = {(seed, task_id, checkpoint) for seed in self.seeds for task_id in range(1, tasks + 1) for checkpoint in CHECKPOINTS}
        self.check(set(counts) == expected_keys and set(counts.values()) == {len(PF_METHODS)}, f"{label} matched task/checkpoint keys", f"keys={len(counts)}")
        self.validate_groups(rows, ("seed", "task_id", "checkpoint"), PF_METHODS, label)
        numeric = ("true_concentration_m", "observed_updates", "estimated_concentration_m", "pka_penalized_mae", "local_rmse_0p10ml_ph", "full_curve_rmse_0_33ml_ph")
        self.check(all(self.finite(row, numeric) for row in rows), f"{label} finite task metrics", f"rows={len(rows)}")
        self.validate_parameter_rows(rows, label)

    def validate_pymc(self):
        path = self.run_dir / "pymc_comparison"
        label = "pymc_comparison"
        tasks = self.spec["pymc"]
        payload = self.read_json(path / "summary.json")
        complete = self.read_json(path / "RUN_COMPLETE.json")
        distribution = "variable_concentration (log-uniform 0.03-0.25 M)"
        self.validate_settings(payload, tasks, self.spec["particles"], distribution, label)
        settings = payload.get("settings", {})
        self.check(settings.get("draws") == self.spec["draws"], f"{label} draws", str(settings.get("draws")))
        self.check(settings.get("chains") == 1, f"{label} chains", str(settings.get("chains")))
        expected = len(self.seeds) * tasks * (len(PF_METHODS) + len(PYMC_METHODS))
        expected_shards = len(self.seeds) * tasks
        self.check(complete.get("status") == "PASS", f"{label} completion marker", str(complete))
        self.check(complete.get("rows") == expected, f"{label} marker row count", str(complete.get("rows")))
        self.check(complete.get("completed_task_shards") == expected_shards, f"{label} marker shard count", str(complete.get("completed_task_shards")))
        rows = self.read_csv(path / "pymc_pf_per_task.csv")
        self.check(len(rows) == expected, f"{label} CSV row count", f"{len(rows)} expected {expected}")
        counts = Counter((int(row["seed"]), int(row["task_id"])) for row in rows)
        expected_keys = {(seed, task_id) for seed in self.seeds for task_id in range(1, tasks + 1)}
        self.check(set(counts) == expected_keys and set(counts.values()) == {6}, f"{label} matched task keys", f"keys={len(counts)}")
        methods = PF_METHODS + PYMC_METHODS
        method_counts = Counter(row.get("method") for row in rows)
        self.check(method_counts == Counter({method: expected_shards for method in methods}), f"{label} method balance", str(method_counts))
        self.validate_groups(rows, ("seed", "task_id"), methods, label)
        numeric = ("true_concentration_m", "estimated_concentration_m", "inference_runtime_seconds", "pka_penalized_mae", "local_rmse_0p10ml_ph", "full_curve_rmse_0_33ml_ph")
        self.check(all(self.finite(row, numeric) for row in rows), f"{label} finite task metrics", f"rows={len(rows)}")
        evidence_valid = True
        for row in rows:
            if row["backend"] != "pymc_smc":
                continue
            evidence = [float(row[f"log_evidence_k{k}"]) for k in (1, 2, 3)]
            if "variable_k" in row["method"]:
                evidence_valid &= all(math.isfinite(value) for value in evidence)
            else:
                evidence_valid &= math.isnan(evidence[0]) and math.isnan(evidence[1]) and math.isfinite(evidence[2])
            evidence_valid &= int(row["smc_draws"]) == self.spec["draws"] and int(row["smc_chains"]) == 1
        self.check(evidence_valid, f"{label} SMC evidence and sampler settings")
        self.validate_parameter_rows(rows, label)

    def validate_master_log(self):
        path = self.run_dir / "MASTER_RUN_LOG.jsonl"
        records = []
        try:
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except Exception as exc:
            self.check(False, "master run log readable", str(exc))
            return
        final_records = records[-5:]
        expected_scripts = (
            "run_pf_multiseed_control.py",
            "run_pf_multiseed_control.py",
            "run_pf_curve_recovery.py",
            "run_pymc_comparison.py",
            "build_master_report.py",
        )
        observed_scripts = tuple(Path(record.get("command", ["", ""])[1]).name for record in final_records if len(record.get("command", [])) > 1)
        self.check(len(records) >= 5 and observed_scripts == expected_scripts, "master run final command sequence", str(observed_scripts))
        self.check(all(record.get("status") == "PASS" and record.get("return_code") == 0 for record in final_records), "master run final command statuses", str([record.get("status") for record in final_records]))
        self.check((self.run_dir / "MASTER_RESULTS_SUMMARY.md").is_file(), "master summary exists")

    def run(self):
        self.validate_control("pf_control_nominal", self.spec["nominal"], "nominal")
        self.validate_control("pf_control_variable_concentration", self.spec["variable"], "variable_concentration")
        self.validate_curve()
        self.validate_pymc()
        self.validate_master_log()
        status = "PASS" if not self.errors else "FAIL"
        report = {
            "status": status,
            "profile": self.profile,
            "run_directory": str(self.run_dir),
            "checks_passed": sum(item["status"] == "PASS" for item in self.checks),
            "checks_failed": len(self.errors),
            "checks": self.checks,
            "errors": self.errors,
        }
        (self.run_dir / "VALIDATION_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        lines = [
            "# Independent result validation",
            "",
            f"Status: **{status}**",
            "",
            f"Profile: `{self.profile}`",
            "",
            f"Checks passed: {report['checks_passed']}; failed: {report['checks_failed']}.",
            "",
            "| Check | Status | Detail |",
            "| --- | --- | --- |",
        ]
        for item in self.checks:
            detail = str(item["detail"]).replace("|", "\\|")
            lines.append(f"| {item['name']} | {item['status']} | {detail} |")
        (self.run_dir / "VALIDATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return status


def main():
    parser = argparse.ArgumentParser(description="Strict independent validator for comparison results")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    args = parser.parse_args()
    status = Validator(args.run_dir.resolve(), args.profile).run()
    print(status)
    raise SystemExit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
