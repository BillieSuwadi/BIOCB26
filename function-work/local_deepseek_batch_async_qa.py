#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure local import works when script is launched from another cwd.
import sys

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from local_deepseek_image_description import ask_nii_question


# Leave question list here for easy customization.
QUESTION_POOL: List[str] = [
    "Illustrate the image through a descriptive explanation。",
    "Present a compact description of the photo key features.",
    "Explain the various aspects of the image before you.",
    "Share a comprehensive rundown of the presented image.",
]


def collect_nii_gz_files(scan_path: Path) -> List[Tuple[str, str, Path]]:
    """
    Recursively collect all .nii.gz files.
    Returns tuple list: (id_without_ext, filename, absolute_path).
    """
    files: List[Tuple[str, str, Path]] = []
    for p in scan_path.rglob("*.nii.gz"):
        filename = p.name
        file_id = filename[: -len(".nii.gz")]
        files.append((file_id, filename, p.resolve()))
    files.sort(key=lambda x: str(x[2]))
    return files


def _ask_one(file_id: str, filename: str, file_path: Path, question: str) -> Dict:
    answer = ask_nii_question(str(file_path), question)
    return {
        "id": file_id,
        "filename": filename,
        "path": str(file_path),
        "question": question,
        "answer": answer,
    }


def run_async_qa(
    scan_path: str,
    seed: int | None = 42,
) -> List[Dict]:
    """
    Main callable function for other scripts.
    1) recursively find .nii.gz
    2) randomly pick one question per file
    3) synchronously call DeepSeek function file-by-file
    4) keep all results in memory (for next-step JSON injection)
    """
    root = Path(scan_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"scan_path is not a valid directory: {root}")
    if not QUESTION_POOL:
        raise ValueError("QUESTION_POOL is empty.")

    if seed is not None:
        random.seed(seed)

    targets = collect_nii_gz_files(root)
    if not targets:
        return []

    results: List[Dict] = []
    for file_id, filename, file_path in targets:
        q = random.choice(QUESTION_POOL)
        try:
            result = _ask_one(file_id, filename, file_path, q)
        except Exception as exc:
            result = {
                "id": file_id,
                "filename": filename,
                "path": str(file_path),
                "question": q,
                "answer": f"[ERROR] {exc}",
            }
        results.append(result)

    results.sort(key=lambda x: x["path"])
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recursively scan .nii.gz files and ask random questions synchronously via local_deepseek_image_description."
    )
    parser.add_argument("scan_path", type=str, help="Folder path to recursively scan .nii.gz files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for question selection")
    parser.add_argument(
        "--print_json",
        action="store_true",
        help="Print collected records in JSON format (does not save file).",
    )
    args = parser.parse_args()

    records = run_async_qa(
        scan_path=args.scan_path,
        seed=args.seed,
    )

    print(f"[INFO] total files: {len(records)}")
    if records:
        print(f"[INFO] sample id: {records[0]['id']}")
        print(f"[INFO] sample filename: {records[0]['filename']}")
        print(f"[INFO] sample question: {records[0]['question']}")
        print(f"[INFO] sample answer: {records[0]['answer'][:200]}")

    if args.print_json:
        print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
