from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path


def convert_nii_to_niigz(root_dir: Path) -> tuple[int, int]:
    converted_count = 0
    skipped_count = 0

    for nii_path in root_dir.rglob("*.nii"):
        gz_path = nii_path.with_suffix(nii_path.suffix + ".gz")

        if gz_path.exists():
            skipped_count += 1
            print(f"Skip existing file: {gz_path}")
            continue

        with nii_path.open("rb") as source, gzip.open(gz_path, "wb") as target:
            shutil.copyfileobj(source, target)

        nii_path.unlink()
        converted_count += 1
        print(f"Converted and removed source: {nii_path} -> {gz_path}")

    return converted_count, skipped_count

def count_nii_gz_file(root_dir: Path) -> int:
    count = 0
    for nii_path in root_dir.rglob("*.nii.gz"):
        count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="input_dir: Recursively convert all .nii files under a folder to .nii.gz files.\n"
                    "input_dir_for_looking: count nii.gz files."
    )
    parser.add_argument("input_dir", help="Path to the input folder")
    parser.add_argument("input_dir_for_looking", help = "Path to count .nii.gz files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(args.input_dir).expanduser().resolve()

    if not root_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {root_dir}")

    if not root_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {root_dir}")

    if args.input_dri is not None:
        converted_count, skipped_count = convert_nii_to_niigz(root_dir)
        print(
            f"Finished. Converted {converted_count} file(s), skipped {skipped_count} existing .nii.gz file(s)."
        )
    if args.input_dir_for_looking is not None:
        count = count_nii_gz_file(root_dir)
        print(f"-- {count} -- founded!")


if __name__ == "__main__":
    main()
