"""LFS downloads must validate exact bytes before replacing any pointer."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import fetch_lfs_assets as lfs


class LfsAssetTests(unittest.TestCase):
    @staticmethod
    def pointer(payload):
        return (f"version https://git-lfs.github.com/spec/v1\n"
                f"oid sha256:{hashlib.sha256(payload).hexdigest()}\n"
                f"size {len(payload)}\n").encode()

    def test_pointer_formats(self):
        original = self.pointer(b"model")
        expected = hashlib.sha256(b"model").hexdigest(), 5
        self.assertEqual(lfs.parse_pointer(original), expected)
        self.assertEqual(lfs.parse_pointer(original.replace(b"\n", b"\r\n")), expected)
        self.assertIsNone(lfs.parse_pointer(b"actual model data"))
        with self.assertRaises(ValueError):
            lfs.parse_pointer(original.replace(b"sha256:", b"sha1:"))

    def test_fetch_deduplicates_and_preserves_real_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "controllers").mkdir()
            (root / "evidence").mkdir()
            (root / "release_archives").mkdir()
            payload = b"\x00\xffmodel\r\n"
            pointer = self.pointer(payload)
            first = root / "controllers/model.pth"
            second = root / "evidence/model.pth"
            real = root / "controllers/actual.pth"
            optional = root / "release_archives/snapshot.zip"
            for path in (first, second, optional):
                path.write_bytes(pointer)
            real.write_bytes(b"keep unchanged")
            with patch.object(lfs, "urlopen", return_value=io.BytesIO(payload)) as request:
                self.assertEqual(lfs.fetch(root, ref="test-commit"), 2)
                request.assert_called_once()
                self.assertIn("/test-commit/", request.call_args.args[0].full_url)
            self.assertEqual(first.read_bytes(), payload)
            self.assertEqual(second.read_bytes(), payload)
            self.assertEqual(real.read_bytes(), b"keep unchanged")
            self.assertEqual(optional.read_bytes(), pointer)
            self.assertEqual(lfs.find_lfs_pointers(root), [])
            self.assertEqual(len(lfs.find_lfs_pointers(root, True)), 1)

    def test_corrupt_or_incomplete_download_does_not_replace_pointer(self):
        for downloaded in (b"wrong", b"sh", b"too long"):
            with self.subTest(downloaded=downloaded), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "controllers").mkdir()
                target = root / "controllers/model.pth"
                original = self.pointer(b"model")
                target.write_bytes(original)
                with patch.object(lfs, "urlopen", return_value=io.BytesIO(downloaded)):
                    with self.assertRaises(ValueError):
                        lfs.fetch(root)
                self.assertEqual(target.read_bytes(), original)

    def test_size_limit_prevents_download_and_file_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "controllers").mkdir()
            target = root / "controllers/model.pth"
            original = self.pointer(b"model")
            target.write_bytes(original)
            with patch.object(lfs, "urlopen") as request:
                with self.assertRaises(ValueError):
                    lfs.fetch(root, max_download_mb=0)
                request.assert_not_called()
            self.assertEqual(target.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
