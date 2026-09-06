import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_pf_local_curves as audit
from scripts.build_reproducibility_release import excluded


class LocalCurveAuditTests(unittest.TestCase):
    def test_all_original_summaries_and_independent_replay(self):
        report = audit.audit()
        self.assertEqual(report["unique_tasks"], 1500)
        self.assertEqual(report["snapshots"], 12000)
        self.assertEqual(report["terminal_threshold_met"], 1404)
        self.assertEqual(report["terminal_horizon_fallbacks"], 96)
        self.assertEqual(report["independent_replay"]["matched_snapshots"], 12000)
        self.assertEqual(report["independent_replay"]["mismatched_fields"], 0)

    def test_all_archive_members_survive_release_filter(self):
        for entry in audit.read_csv(audit.BLOCK / "MANIFEST_SHA256.csv"):
            with self.subTest(path=entry["path"]):
                self.assertFalse(excluded(audit.BLOCK / entry["path"]))

    def test_review_documents_are_still_excluded(self):
        for name in ("reviewer_response.md", "manuscript.docx", "response_letter.md"):
            self.assertTrue(excluded(audit.BLOCK / name))

    def test_reject_duplicate_snapshot_and_changed_rmse(self):
        row = audit.read_csv(audit.RESULTS / "all_local_response_rows.csv")[0]
        with self.assertRaises(ValueError):
            audit.indexed([row, row])
        modified = copy.copy(row)
        modified["local_rmse_0p1_ml"] = "0.0"
        with self.assertRaises(ValueError):
            audit.compare_fields(modified, row)

    def test_seed_statistics_use_the_recorded_task_denominator(self):
        rows = [{f"local_rmse_{window}_ml": value for window in audit.WINDOWS}
                for value in (.02, .06, .4)]
        result = audit.seed_summary(rows)
        self.assertEqual(result["tasks"], 3)
        self.assertAlmostEqual(result["local_rmse_0p1_ml_mean"], .16)
        self.assertEqual(result["local_rmse_0p1_ml_median"], .06)
        self.assertAlmostEqual(result["local_rmse_0p1_ml_le_0p10_percent"], 200 / 3)


if __name__ == "__main__":
    unittest.main()
