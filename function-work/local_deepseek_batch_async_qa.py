#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
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
    "请简要描述这个MRI文件可能包含的解剖结构信息。",
    "这个NIfTI文件的维度和体素间距信息可能说明了什么？",
    "从医学影像处理角度，这个文件在分析前需要哪些预处理？",
    "请说明该图像可能的质量风险和注意事项。",
    "如果用于阿尔兹海默症研究，这个文件可以支持哪些基础分析？",
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
    max_workers: int = 4,
    seed: int | None = 42,
) -> List[Dict]:
    """
    Main callable function for other scripts.
    1) recursively find .nii.gz
    2) randomly pick one question per file
    3) asynchronously call DeepSeek function
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
    future_to_meta: Dict[Future, Tuple[str, str, Path, str]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for file_id, filename, file_path in targets:
            q = random.choice(QUESTION_POOL)
            fut = executor.submit(_ask_one, file_id, filename, file_path, q)
            future_to_meta[fut] = (file_id, filename, file_path, q)

        for fut in as_completed(future_to_meta):
            file_id, filename, file_path, q = future_to_meta[fut]
            try:
                result = fut.result()
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
        description="Recursively scan .nii.gz files and ask random questions via local_deepseek_image_description."
    )
    parser.add_argument("scan_path", type=str, help="Folder path to recursively scan .nii.gz files")
    parser.add_argument("--max_workers", type=int, default=4, help="Thread pool size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for question selection")
    parser.add_argument(
        "--print_json",
        action="store_true",
        help="Print collected records in JSON format (does not save file).",
    )
    args = parser.parse_args()

    records = run_async_qa(
        scan_path=args.scan_path,
        max_workers=max(1, args.max_workers),
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
