"""Regression tests for independent primary PPO result validation."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_primary_ppo_five_seeds as audit


class PrimaryPPOAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.row = audit.read_csv(audit.BLOCK / "results/ppo_303_benchmark_101.csv")[0]
        task_id = int(cls.row["task_id"])
        tasks = [json.loads(line) for line in
                 (audit.FORMAL / "tasks/seed_101_tasks.jsonl").read_text().splitlines()]
        cls.task = next(t for t in tasks if t["task_id"] == task_id)

    def test_released_results_and_all_summaries(self):
        report = audit.audit()
        self.assertEqual(report["evaluations"], 75000)
        self.assertEqual(report["ppo_higher_than_imitation_cells"], 25)
        self.assertEqual(report["selected_model_rows_compared"], 15000)
        self.assertAlmostEqual(report["training_seed_success_mean_percent"], 91.7933333333333)
        self.assertAlmostEqual(report["training_seed_success_sd_percent"], 1.52535606043675)

    def test_valid_task(self):
        audit.validate_task_row(self.row, self.task, 303, 101)

    def test_reject_incorrect_success_or_steps_or_dose_or_identity(self):
        changes = {
            "true_success": str(1 - int(self.row["true_success"])),
            "steps": "51", "total_volume_ml": "900", "task_seed": "101",
            "stop_reason": "running", "training_seed": "555",
            "final_abs_error": "-0.01",
        }
        for field, value in changes.items():
            with self.subTest(field=field):
                row = copy.copy(self.row)
                row[field] = value
                with self.assertRaises(ValueError):
                    audit.validate_task_row(row, self.task, 303, 101)

    def test_reject_duplicate_keys(self):
        with self.assertRaises(ValueError):
            audit.indexed([self.row, self.row], ("task_seed", "task_id"))

    def test_summary_mean_is_conditional_on_success(self):
        good, bad = copy.copy(self.row), copy.copy(self.row)
        good.update(true_success="1", steps="3")
        bad.update(true_success="0", steps="50")
        result = audit.summarize([good, bad])
        self.assertEqual(result["success_rate_percent"], 50)
        self.assertEqual(result["successful_steps_mean"], 3)
        self.assertEqual(result["steps_mean"], 26.5)

    def test_text_hash_allows_crlf_not_content_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            block = Path(temporary)
            data = b"original\n"
            file = block / "data.txt"
            file.write_bytes(b"original\r\n")
            (block / "MANIFEST_SHA256.csv").write_text(
                "sha256,bytes,path\n" + hashlib.sha256(data).hexdigest() + ",9,data.txt\n")
            self.assertEqual(audit.verify_archive(block), 1)
            file.write_bytes(b"changed!\r\n")
            with self.assertRaises(ValueError):
                audit.verify_archive(block)


if __name__ == "__main__":
    unittest.main()
