"""Download hash-verified Git LFS assets missing from a GitHub source ZIP.

Only valid LFS pointer files in controllers/ and evidence/ are replaced.
Existing models/data are never overwritten. release_archives/ is optional.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import tempfile
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def parse_pointer(data):
    if not data.startswith(POINTER_PREFIX):
        return None
    match = re.fullmatch(
        rb"version https://git-lfs.github.com/spec/v1\r?\n"
        rb"oid sha256:([0-9a-f]{64})\r?\nsize ([0-9]+)\r?\n?", data)
    if match is None:
        raise ValueError("Malformed Git LFS pointer")
    return match[1].decode("ascii"), int(match[2])


def find_lfs_pointers(root, include_release_archives=False):
    root = Path(root).resolve()
    pointers = []
    names = ["controllers", "evidence"]
    if include_release_archives:
        names.append("release_archives")
    for name in names:
        for path in sorted((root / name).rglob("*")):
            if not path.is_file() or path.stat().st_size > 1024:
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError(f"Asset path escapes source directory: {path}")
            original = path.read_bytes()
            pointer = parse_pointer(original)
            if pointer is not None:
                oid, size = pointer
                pointers.append({"path": path, "oid": oid, "size": size, "original": original})
    return pointers


def download_object(pointer, root, ref, output):
    relative = pointer["path"].relative_to(root).as_posix()
    url = ("https://media.githubusercontent.com/media/Wang-Group/pH-Titration/"
           + quote(ref, safe="/") + "/" + quote(relative, safe="/"))
    digest = hashlib.sha256()
    received = 0
    request = Request(url, headers={"User-Agent": "pH-Titration-reproduction"})
    with urlopen(request, timeout=90) as response, output.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            received += len(chunk)
            if received > pointer["size"]:
                raise ValueError(f"Downloaded object exceeds recorded size: {relative}")
            digest.update(chunk)
            handle.write(chunk)
    if received != pointer["size"] or digest.hexdigest() != pointer["oid"]:
        raise ValueError(f"Downloaded Git LFS object failed size/SHA-256 validation: {relative}")


def fetch(root=ROOT, ref="main", include_release_archives=False, max_download_mb=512):
    root = Path(root).resolve()
    pointers = find_lfs_pointers(root, include_release_archives)
    unique = {}
    for pointer in pointers:
        previous = unique.setdefault(pointer["oid"], pointer["size"])
        if previous != pointer["size"]:
            raise ValueError("Inconsistent sizes for the same Git LFS object")
    total = sum(unique.values())
    if total > max_download_mb * 1024 * 1024:
        raise ValueError("Downloads exceed --max-download-mb; no files changed")
    print(f"Missing LFS files: {len(pointers)}; unique download bytes: {total}", flush=True)
    with tempfile.TemporaryDirectory(prefix="ph-lfs-") as temporary:
        cache = Path(temporary).resolve()
        for index, pointer in enumerate(pointers, 1):
            cached = cache / pointer["oid"]
            if not cached.exists():
                download_object(pointer, root, ref, cached)
            target = pointer["path"]
            if target.read_bytes() != pointer["original"]:
                raise ValueError(f"Pointer changed during download; refusing overwrite: {target}")
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".lfs-", delete=False) as handle:
                replacement = Path(handle.name)
            try:
                shutil.copyfile(cached, replacement)
                if target.read_bytes() != pointer["original"]:
                    raise ValueError(f"Pointer changed before replacement: {target}")
                os.replace(replacement, target)
            finally:
                replacement.unlink(missing_ok=True)
            print(f"[{index}/{len(pointers)}] Verified {target.relative_to(root).as_posix()}", flush=True)
    return len(pointers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Extracted repository root")
    parser.add_argument("--ref", default="main", help="Commit/tag/branch used for the source ZIP")
    parser.add_argument("--include-release-archives", action="store_true")
    parser.add_argument("--max-download-mb", type=int, default=512)
    args = parser.parse_args()
    if not args.root.is_dir() or args.max_download_mb < 1:
        parser.error("root must exist and max-download-mb must be positive")
    fetch(args.root, args.ref, args.include_release_archives, args.max_download_mb)


if __name__ == "__main__":
    main()
