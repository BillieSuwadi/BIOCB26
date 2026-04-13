#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Tuple

import requests


API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-r1:70b"
REQUEST_TIMEOUT = 120

# Thread-local session: safe for multi-threaded callers and avoids cross-thread state sharing.
_thread_local = threading.local()


def _get_session() -> requests.Session:
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        sess.headers.update({"Content-Type": "application/json"})
        _thread_local.session = sess
    return sess


def _validate_path(nii_path: str | Path) -> Path:
    p = Path(nii_path).expanduser().resolve()
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"NIfTI file not found: {p}")
    name_lower = p.name.lower()
    if not (name_lower.endswith(".nii.gz") or name_lower.endswith(".nii")):
        raise ValueError(f"Only .nii or .nii.gz is supported, got: {p.name}")
    return p


def _extract_basic_info(nii_file: Path) -> Dict[str, Any]:
    """
    Best-effort metadata extraction. If no imaging libs are available, return path-only info.
    """
    info: Dict[str, Any] = {
        "file_name": nii_file.name,
        "absolute_path": str(nii_file),
        "file_size_bytes": nii_file.stat().st_size,
    }

    # Try nibabel first.
    try:
        import nibabel as nib  # type: ignore

        img = nib.load(str(nii_file))
        info["shape"] = tuple(int(v) for v in img.shape)
        if hasattr(img.header, "get_zooms"):
            info["zooms"] = tuple(float(v) for v in img.header.get_zooms())
        return info
    except Exception:
        pass

    # Fallback to SimpleITK.
    try:
        import SimpleITK as sitk  # type: ignore

        img = sitk.ReadImage(str(nii_file))
        info["shape_xyz"] = tuple(int(v) for v in img.GetSize())
        info["spacing_xyz"] = tuple(float(v) for v in img.GetSpacing())
        info["dimension"] = int(img.GetDimension())
    except Exception:
        # Keep minimal info if imaging libs are missing.
        pass

    return info


def _build_prompt(file_info: Dict[str, Any], question: str) -> str:
    return (
        "You are a medical imaging assistant. "
        "Use the provided NIfTI file metadata and answer the user question. "
        "If data is insufficient, state the limitation clearly.\n\n"
        f"NIfTI metadata:\n{json.dumps(file_info, ensure_ascii=False, indent=2)}\n\n"
        f"Question:\n{question}\n\n"
        "Answer in concise English please."
    )


def _request_once(payload: Dict[str, Any]) -> Tuple[bool, str]:
    try:
        session = _get_session()
        resp = session.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response", "")
        if not isinstance(text, str) or not text.strip():
            return False, "empty response from model"
        return True, text.strip()
    except Exception as exc:
        return False, str(exc)


def ask_nii_question(nii_path: str, question: str) -> str:
    """
    Callable function for other scripts.
    Args:
      nii_path: path to .nii.gz/.nii file
      question: user question text
    Returns:
      Answer string
    """
    if not question or not question.strip():
        return "[ERROR] question is empty."

    try:
        nii_file = _validate_path(nii_path)
    except Exception as exc:
        return f"[ERROR] invalid input path: {exc}"

    file_info = _extract_basic_info(nii_file)
    prompt = _build_prompt(file_info=file_info, question=question.strip())
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.2,
    }

    # Simple retry for transient local service failures.
    for attempt in range(2):
        ok, msg = _request_once(payload)
        if ok:
            return msg
        if attempt == 1:
            return f"[ERROR] deepseek request failed: {msg}"

    return "[ERROR] unexpected failure."


if __name__ == "__main__":
    # Simple local debug usage:
    # python local_deepseek_image_description.py /path/to/file.nii.gz "请描述这个MRI文件的基础信息"
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("nii_path", type=str)
    parser.add_argument("question", type=str)
    args = parser.parse_args()
    print(ask_nii_question(args.nii_path, args.question))
