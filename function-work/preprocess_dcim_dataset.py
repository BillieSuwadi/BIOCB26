#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ADNI DICOM preprocessing pipeline
- Traverse ADNI-like folder structure:
  ADNI1/<SUBJECT_ID>/<SEQUENCE>/<DATE_TIME>/<IMAGESET_ID>/(DICOM files)
- Convert DICOM series -> NIfTI
- Pick target sequence (MPRAGE/MP-RAGE/MPRAGE_REPEAT)
- Resample to isotropic spacing
- Optional N4 bias correction
- Intensity normalization
- Center crop/pad to fixed shape
- Save .nii.gz + optional .npy

Usage:
  python preprocess_adni_dicom.py \
    --root /path/to/ADNI1 \
    --out /path/to/output \
    --target_spacing 1.0 1.0 1.0 \
    --target_shape 160 192 160 \
    --do_n4 1
"""

from __future__ import annotations
import argparse
import re
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
from tqdm import tqdm

import SimpleITK as sitk
import dicom2nifti


# ----------------------------
# Helpers: folder traversal
# ----------------------------

SEQ_PRIORITY = [
    # Higher priority first
    "accelerated_sagittal_mprage",
    "mprage",          # includes MPRAGE, MP-RAGE
    "mprage_repeat",
]

def normalize_seq_name(name: str) -> str:
    n = name.lower().replace(" ", "_").replace("-", "_")
    n = re.sub(r"__+", "_", n)
    return n

def seq_score(seq_folder_name: str) -> int:
    n = normalize_seq_name(seq_folder_name)
    # heuristic scoring
    if "accelerated" in n and "mprage" in n:
        return 300
    if "mprage_repeat" in n or ("mprage" in n and "repeat" in n):
        return 250
    if "mprage" in n or "mp_rage" in n:
        return 200
    return 0

def find_dicoms_in_imageset(imageset_dir: Path) -> List[Path]:
    # DICOM files sometimes have no extension
    files = [p for p in imageset_dir.rglob("*") if p.is_file()]
    # Filter obvious non-dicom
    out = []
    for f in files:
        if f.name.startswith("."):
            continue
        if f.suffix.lower() in [".json", ".txt", ".csv", ".nii", ".gz", ".png", ".jpg"]:
            continue
        out.append(f)
    return out

def collect_candidate_series(root: Path) -> List[Dict]:
    """
    Return list of candidates with keys:
      subject_id, seq_name, session_name, imageset_id, imageset_dir
    """
    cands = []
    # root: ADNI1
    for subject_dir in root.iterdir():
        if not subject_dir.is_dir():
            continue
        subject_id = subject_dir.name  # e.g. 002_S_0413
        for seq_dir in subject_dir.iterdir():
            if not seq_dir.is_dir():
                continue
            for session_dir in seq_dir.iterdir():
                if not session_dir.is_dir():
                    continue
                for imageset_dir in session_dir.iterdir():
                    if not imageset_dir.is_dir():
                        continue
                    # Heuristic: imageset_id like I115005
                    imageset_id = imageset_dir.name
                    if not re.match(r"^I\d+", imageset_id):
                        # still allow, but lower chance
                        pass
                    dicoms = find_dicoms_in_imageset(imageset_dir)
                    if len(dicoms) < 10:
                        continue
                    cands.append({
                        "subject_id": subject_id,
                        "seq_name": seq_dir.name,
                        "session_name": session_dir.name,
                        "imageset_id": imageset_id,
                        "imageset_dir": imageset_dir,
                        "score": seq_score(seq_dir.name),
                        "n_files": len(dicoms),
                    })
    return cands

def pick_best_candidate(cands: List[Dict]) -> Optional[Dict]:
    if not cands:
        return None
    # prioritize by score, then by number of files (more slices often means full 3D)
    cands_sorted = sorted(cands, key=lambda x: (x["score"], x["n_files"]), reverse=True)
    return cands_sorted[0]


# ----------------------------
# Image processing (SimpleITK)
# ----------------------------

def load_nifti_sitk(path: Path) -> sitk.Image:
    img = sitk.ReadImage(str(path))
    return img

def reorient_to_ras(img: sitk.Image) -> sitk.Image:
    """
    SimpleITK uses LPS by convention; many DL pipelines don’t strictly need RAS
    as long as you are consistent. We keep LPS but ensure direction matrix is valid.
    This function is kept minimal; full reorientation can be done via nibabel if needed.
    """
    # Ensure direction is set; if missing, keep as is.
    return img

def resample_isotropic(img: sitk.Image, out_spacing: Tuple[float, float, float]) -> sitk.Image:
    in_spacing = img.GetSpacing()
    in_size = img.GetSize()
    out_size = [
        int(round(in_size[i] * (in_spacing[i] / out_spacing[i])))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(out_spacing)
    resampler.SetSize(out_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    # T1: linear interpolation
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(img)

def n4_bias_correction(img: sitk.Image, shrink_factor: int = 2, n_iter: int = 50) -> sitk.Image:
    """
    N4 bias correction (optional but helpful for MRI).
    """
    img_float = sitk.Cast(img, sitk.sitkFloat32)
    mask = sitk.OtsuThreshold(img_float, 0, 1, 200)  # quick mask
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([n_iter])
    corrected = corrector.Execute(
        sitk.Shrink(img_float, [shrink_factor]*3),
        sitk.Shrink(mask, [shrink_factor]*3)
    )
    # Apply estimated bias field to full-res image
    log_bias_field = corrector.GetLogBiasFieldAsImage(img_float)
    corrected_full = img_float / sitk.Exp(log_bias_field)
    return sitk.Cast(corrected_full, sitk.sitkFloat32)

def sitk_to_numpy(img: sitk.Image) -> np.ndarray:
    arr = sitk.GetArrayFromImage(img)  # z,y,x
    # Convert to x,y,z if you prefer; many 3D CNN use z,y,x too.
    return arr

def intensity_normalize(arr: np.ndarray, method: str = "zscore") -> np.ndarray:
    """
    method:
      - zscore: (x-mean)/std within non-zero mask
      - pminmax: clip to [1,99] percentile then scale to [0,1]
    """
    a = arr.astype(np.float32)
    mask = a > 0  # crude mask; replace with brain mask if you have one
    if mask.sum() < 100:
        return a

    vals = a[mask]
    if method == "zscore":
        m = float(vals.mean())
        s = float(vals.std() + 1e-8)
        a[mask] = (a[mask] - m) / s
        a[~mask] = 0.0
        return a
    elif method == "pminmax":
        lo = np.percentile(vals, 1)
        hi = np.percentile(vals, 99)
        a = np.clip(a, lo, hi)
        a = (a - lo) / (hi - lo + 1e-8)
        a[~mask] = 0.0
        return a
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def center_crop_pad(arr: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    arr: (z,y,x)
    target_shape: (z,y,x)
    """
    tz, ty, tx = target_shape
    z, y, x = arr.shape

    out = np.zeros((tz, ty, tx), dtype=arr.dtype)

    def _compute(src_len, dst_len):
        if src_len >= dst_len:
            src_start = (src_len - dst_len) // 2
            src_end = src_start + dst_len
            dst_start = 0
            dst_end = dst_len
        else:
            src_start = 0
            src_end = src_len
            dst_start = (dst_len - src_len) // 2
            dst_end = dst_start + src_len
        return src_start, src_end, dst_start, dst_end

    zs, ze, zd_s, zd_e = _compute(z, tz)
    ys, ye, yd_s, yd_e = _compute(y, ty)
    xs, xe, xd_s, xd_e = _compute(x, tx)

    out[zd_s:zd_e, yd_s:yd_e, xd_s:xd_e] = arr[zs:ze, ys:ye, xs:xe]
    return out

def save_nifti_from_numpy(arr_zyx: np.ndarray, ref_img: sitk.Image, out_path: Path) -> None:
    """
    Save numpy array back to NIfTI using reference spacing/origin/direction.
    """
    out_img = sitk.GetImageFromArray(arr_zyx)  # z,y,x
    out_img.SetSpacing(ref_img.GetSpacing())
    out_img.SetOrigin(ref_img.GetOrigin())
    out_img.SetDirection(ref_img.GetDirection())
    sitk.WriteImage(out_img, str(out_path))


# ----------------------------
# DICOM -> NIfTI conversion
# ----------------------------

def convert_dicom_series_to_nifti(imageset_dir: Path, tmp_dir: Path) -> Optional[Path]:
    """
    Convert one imageset directory to NIfTI.
    Returns path to converted NIfTI if success.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # dicom2nifti expects a directory that contains a single series or multiple;
    # It will output NIfTI files into tmp_dir.
    try:
        dicom2nifti.convert_directory(
            str(imageset_dir),
            str(tmp_dir),
            compression=True,
            reorient=False  # keep consistent; we handle later
        )
    except Exception:
        return None

    nii_files = sorted(tmp_dir.glob("*.nii.gz"))
    if not nii_files:
        nii_files = sorted(tmp_dir.glob("*.nii"))
    if not nii_files:
        return None

    # If multiple outputs, pick largest by file size (usually the full 3D)
    nii_files.sort(key=lambda p: p.stat().st_size, reverse=True)
    return nii_files[0]


# ----------------------------
# Main pipeline
# ----------------------------

def process_one_subject(best: Dict, out_root: Path,
                        target_spacing: Tuple[float, float, float],
                        target_shape: Tuple[int, int, int],
                        do_n4: bool,
                        norm_method: str,
                        save_npy: bool) -> Dict:
    subject_id = best["subject_id"]
    seq_name = best["seq_name"]
    session_name = best["session_name"]
    imageset_id = best["imageset_id"]
    imageset_dir = best["imageset_dir"]

    out_dir = out_root / subject_id / normalize_seq_name(seq_name) / session_name / imageset_id
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = out_dir / "_tmp_convert"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    nii_path = convert_dicom_series_to_nifti(imageset_dir, tmp_dir)
    if nii_path is None:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return {"subject_id": subject_id, "status": "fail_convert"}

    # Load and preprocess
    img = load_nifti_sitk(nii_path)
    img = reorient_to_ras(img)
    img = resample_isotropic(img, target_spacing)

    if do_n4:
        img = n4_bias_correction(img)

    arr = sitk_to_numpy(img)  # z,y,x
    arr = intensity_normalize(arr, method=norm_method)
    arr = center_crop_pad(arr, target_shape)

    # Save outputs
    out_nii = out_dir / "t1_preproc.nii.gz"
    save_nifti_from_numpy(arr, img, out_nii)

    if save_npy:
        out_npy = out_dir / "t1_preproc.npy"
        np.save(out_npy, arr)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "subject_id": subject_id,
        "status": "ok",
        "seq_name": seq_name,
        "session": session_name,
        "imageset_id": imageset_id,
        "out": str(out_nii),
        "score": best["score"],
        "n_files": best["n_files"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=str, help="ADNI root folder, e.g. /data/ADNI1")
    ap.add_argument("--out", required=True, type=str, help="Output root folder")
    ap.add_argument("--target_spacing", nargs=3, type=float, default=[1.0, 1.0, 1.0])
    ap.add_argument("--target_shape", nargs=3, type=int, default=[160, 192, 160], help="(z y x)")
    ap.add_argument("--do_n4", type=int, default=1, help="1 to enable N4 bias correction")
    ap.add_argument("--norm", type=str, default="zscore", choices=["zscore", "pminmax"])
    ap.add_argument("--save_npy", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    target_spacing = (float(args.target_spacing[0]), float(args.target_spacing[1]), float(args.target_spacing[2]))
    target_shape = (int(args.target_shape[0]), int(args.target_shape[1]), int(args.target_shape[2]))
    do_n4 = bool(args.do_n4)
    save_npy = bool(args.save_npy)

    cands = collect_candidate_series(root)

    # Group by subject -> pick best candidate series per subject
    by_subj: Dict[str, List[Dict]] = {}
    for c in cands:
        by_subj.setdefault(c["subject_id"], []).append(c)

    results = []
    for sid, lst in tqdm(sorted(by_subj.items(), key=lambda x: x[0]), desc="Subjects"):
        best = pick_best_candidate(lst)
        if best is None or best["score"] == 0:
            results.append({"subject_id": sid, "status": "skip_no_mprage"})
            continue
        res = process_one_subject(
            best=best,
            out_root=out,
            target_spacing=target_spacing,
            target_shape=target_shape,
            do_n4=do_n4,
            norm_method=args.norm,
            save_npy=save_npy
        )
        results.append(res)

    # Save a small manifest
    import csv
    manifest = out / "manifest.csv"
    keys = sorted({k for r in results for k in r.keys()})
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow(r)

    print(f"Done. Manifest: {manifest}")


if __name__ == "__main__":
    main()
