from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def find_mask_directories(root_dir: Path) -> list[Path]:
    mask_dirs = [path for path in root_dir.rglob("*") if path.is_dir() and "Mask" in path.name]
    return sorted(mask_dirs, key=lambda path: len(path.parts), reverse=True)


def remove_mask_directories(root_dir: Path) -> int:
    removed_count = 0

    for mask_dir in find_mask_directories(root_dir):
        if not mask_dir.exists():
            continue

        shutil.rmtree(mask_dir)
        removed_count += 1
        print(f"Removed directory: {mask_dir}")

    return removed_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively find and remove folders whose names contain 'Mask'."
    )
    parser.add_argument("input_dir", help="Path to the folder to scan")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(args.input_dir).expanduser().resolve()

    if not root_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {root_dir}")

    if not root_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {root_dir}")

    removed_count = remove_mask_directories(root_dir)
    print(f"Finished. Removed {removed_count} folder(s).")


if __name__ == "__main__":
    main()
