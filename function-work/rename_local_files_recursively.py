#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path


def build_new_name(root: Path, nii_file: Path) -> str:
    """
    Use all parent folder names (relative to root) joined by '_' as new .nii.gz filename.
    Example:
      root/002_S_0413/mprage/2020-01-01/I123/t1_preproc.nii.gz
      -> 002_S_0413_mprage_2020-01-01_I123.nii.gz
    """
    rel_parent = nii_file.parent.relative_to(root)
    parts = [p for p in rel_parent.parts if p]
    if not parts:
        # If .nii.gz appears directly under root, fallback to original stem.
        return f"{nii_file.stem}.nii.gz"
    return "_".join(parts) + ".nii.gz"


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    base = path.name[:-7] if path.name.endswith(".nii.gz") else path.stem
    idx = 1
    while True:
        candidate = path.with_name(f"{base}_{idx}.nii.gz")
        if not candidate.exists():
            return candidate
        idx += 1


def rename_nii_gz_files(root: Path, dry_run: bool = False) -> tuple[int, int]:
    files = sorted(root.rglob("*.nii.gz"))
    renamed = 0
    skipped = 0

    for f in files:
        new_name = build_new_name(root, f)
        target = f.with_name(new_name)

        if f.name == new_name:
            skipped += 1
            continue

        target = unique_path(target)
        if dry_run:
            print(f"[DRY-RUN] {f} -> {target}")
            renamed += 1
            continue

        f.rename(target)
        print(f"[RENAMED] {f} -> {target}")
        renamed += 1

    return renamed, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recursively rename all .nii.gz files using parent folder names."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="./ADNI-output",
        help="ADNI-output root folder path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview rename operations without changing files",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"[ERROR] Invalid root directory: {root}")

    renamed, skipped = rename_nii_gz_files(root, dry_run=args.dry_run)
    print(f"[DONE] root={root}")
    print(f"[DONE] renamed={renamed}, skipped_already_named={skipped}")


if __name__ == "__main__":
    main()
