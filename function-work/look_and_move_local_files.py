#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import fnmatch
import shutil
from pathlib import Path


def split_name_and_suffix(name_pattern: str) -> tuple[str, str]:
    """
    Split a query string into fuzzy name part and suffix part.
    Supports multi-suffix extensions like .nii.gz.
    """
    lower = name_pattern.lower().strip()
    dot_index = lower.find(".")
    if dot_index == -1:
        return lower, ""
    return lower[:dot_index], lower[dot_index:]


def fuzzy_match_filename(filename: str, query: str) -> bool:
    """
    Match by fuzzy filename + suffix:
    1) wildcard query (e.g. *mprage*.nii.gz) via fnmatch on full name
    2) otherwise:
       - name part: substring fuzzy match
       - suffix part: endswith match
    """
    name = filename.lower()
    q = query.lower().strip()
    if not q:
        return False

    if any(ch in q for ch in ["*", "?", "[", "]"]):
        return fnmatch.fnmatch(name, q)

    q_name, q_suffix = split_name_and_suffix(q)
    if q_name and q_name not in name:
        return False
    if q_suffix and not name.endswith(q_suffix):
        return False
    return True


def unique_destination_path(dest_file: Path) -> Path:
    if not dest_file.exists():
        return dest_file

    stem = dest_file.stem
    suffix = "".join(dest_file.suffixes)
    if suffix:
        stem = dest_file.name[: -len(suffix)]
    parent = dest_file.parent

    idx = 1
    while True:
        candidate = parent / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def collect_matches(scan_path: Path, query: str) -> list[Path]:
    matches: list[Path] = []
    for p in scan_path.rglob("*"):
        if p.is_file() and fuzzy_match_filename(p.name, query):
            matches.append(p)
    return matches


def copy_matches(matches: list[Path], scan_path: Path, out_path: Path) -> list[Path]:
    copied: list[Path] = []
    out_path.mkdir(parents=True, exist_ok=True)

    for src in matches:
        rel = src.relative_to(scan_path)
        dest = out_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest = unique_destination_path(dest)
        shutil.copy2(src, dest)
        copied.append(dest)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recursively scan files by fuzzy filename pattern and copy them to an output folder."
    )
    parser.add_argument("scan_path", type=str, help="Directory to recursively scan")
    parser.add_argument("file_name", type=str, help="Target file name pattern (supports fuzzy/wildcard)")
    parser.add_argument("output_path", type=str, help="Output directory path")
    args = parser.parse_args()

    scan_path = Path(args.scan_path).expanduser().resolve()
    out_path = Path(args.output_path).expanduser().resolve()
    query = args.file_name.strip()

    if not scan_path.exists() or not scan_path.is_dir():
        raise SystemExit(f"[ERROR] scan_path is not a valid directory: {scan_path}")

    matches = collect_matches(scan_path, query)
    if not matches:
        print(f"[INFO] No file matched query: {query}")
        return

    copied = copy_matches(matches, scan_path, out_path)

    print(f"[INFO] Query: {query}")
    print(f"[INFO] Matched files: {len(matches)}")
    print(f"[INFO] Copied files: {len(copied)}")
    print(f"[INFO] Output path: {out_path}")


if __name__ == "__main__":
    main()
