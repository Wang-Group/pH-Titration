"""Public-tree exclusions and strict LF/CRLF integrity regressions."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.verify_source import (
    hash_byte_candidates, matches_manifest_entry, matches_sha256_allowing_crlf,
    validate_public_paths, verify_public_layout,
)


class ReleaseIntegrityTests(unittest.TestCase):
    def check_match(self, stored, expected, suffix=".json", matched=True):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / ("sample" + suffix)
            path.write_bytes(stored)
            digest = hashlib.sha256(expected).hexdigest()
            self.assertEqual(matches_sha256_allowing_crlf(path, digest), matched)
            self.assertEqual(matches_manifest_entry(path, {"sha256": digest, "bytes": str(len(expected))}), matched)

    def test_crlf_manifest_matches_lf_download(self):
        self.check_match(b'{\n  "status": "PASS"\n}\n', b'{\r\n  "status": "PASS"\r\n}\r\n')

    def test_lf_manifest_matches_crlf_checkout(self):
        self.check_match(b'{\r\n  "status": "PASS"\r\n}\r\n', b'{\n  "status": "PASS"\n}\n')

    def test_exact_binary_and_text_bytes(self):
        for data, suffix in ((b"a\r\nb\nc\r", ".txt"), (b"\x00\xff\r\n", ".pth")):
            self.check_match(data, data, suffix)

    def test_does_not_hide_content_whitespace_bom_or_final_newline_changes(self):
        expected = b'{\r\n  "value": 1\r\n}\r\n'
        for changed in (b'{\n  "value": 2\n}\n', b'{\n "value": 1\n}\n',
                        b'{\n  "value": 1\n}', b'\xef\xbb\xbf{\n  "value": 1\n}\n',
                        b'{\r  "value": 1\r}\r'):
            with self.subTest(changed=changed):
                self.check_match(changed, expected, matched=False)

    def test_binary_newline_changes_are_not_allowed(self):
        for suffix in (".pth", ".npz", ".png", ".zip"):
            self.check_match(b"valid-as-utf8\n", b"valid-as-utf8\r\n", suffix, matched=False)
        self.check_match(b"\x00\n", b"\x00\r\n", ".json", matched=False)
        self.check_match(b"\xff\n", b"\xff\r\n", ".txt", matched=False)

    def test_recorded_size_is_not_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "record.csv"
            path.write_bytes(b"a,b\n1,2\n")
            self.assertFalse(matches_manifest_entry(path, {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": "999"}))

    def test_sensor_package_complete_actual_archive_regression(self):
        block = ROOT / "evidence/simulation_numerical_evidence_20260823/08_SENSOR_STRESS/reproduction_package_20260902"
        relative = "controllers_release/PACKAGE_COMPLETE.json"
        expected = next(line.split("  ", 1)[0] for line in
                        (block / "source_archive_SHA256SUMS.txt").read_text().splitlines()
                        if line.endswith("  " + relative))
        source = block / "controller_source" / relative
        lf = source.read_bytes().replace(b"\r\n", b"\n")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "PACKAGE_COMPLETE.json"
            path.write_bytes(lf)
            self.assertNotEqual(hashlib.sha256(lf).hexdigest(), expected)
            self.assertTrue(matches_sha256_allowing_crlf(path, expected))

    def test_private_directory_paths_are_rejected(self):
        for path in ("ph4github_analysiscopy", "ph4github_analysiscopy/report.pdf",
                     "PH4GITHUB_ANALYSISCOPY\\draft.docx"):
            with self.assertRaises(ValueError):
                validate_public_paths([path])
        validate_public_paths(["controllers/controller_api.py", "evidence/reproduction/results.csv"])

    def test_archived_sensor_snapshots_preserve_original_bytes(self):
        block = ROOT / "evidence/simulation_numerical_evidence_20260823/08_SENSOR_STRESS/reproduction_package_20260902"
        manifest = dict(line.split("  ", 1)[::-1] for line in
                        (block / "source_archive_SHA256SUMS.txt").read_text().splitlines()
                        if "  " in line)
        paths = {
            "controller_source/controllers_release/evidence/RL_EFFECTIVENESS_AUDIT.md":
                "controllers_release/evidence/RL_EFFECTIVENESS_AUDIT.md",
            "runner/controllers_release/evidence/RL_EFFECTIVENESS_AUDIT.md":
                "controllers_release/evidence/RL_EFFECTIVENESS_AUDIT.md",
            "runner/study_source/BAYESIAN_RULE_ABLATION.ipynb":
                "study_source/BAYESIAN_RULE_ABLATION.ipynb",
            "runner/study_source/reference/original_bayesian_controller.py":
                "study_source/reference/original_bayesian_controller.py",
        }
        for path, key in paths.items():
            with self.subTest(path=path):
                self.assertEqual(hashlib.sha256((block / path).read_bytes()).hexdigest(), manifest[key])

    def test_archive_layout_without_git_is_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            verify_public_layout(root)
            (root / "ph4github_analysiscopy").mkdir()
            with self.assertRaises(ValueError):
                verify_public_layout(root)

    def test_current_git_index_excludes_private_working_copy(self):
        verify_public_layout(ROOT)


if __name__ == "__main__":
    unittest.main()
