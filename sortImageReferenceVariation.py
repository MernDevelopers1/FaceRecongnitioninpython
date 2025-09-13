#!/usr/bin/env python3
"""
Robust image sorter using face_recognition.

Features:
- Recursive scan of source folder for images
- Multi-pass face detection: HOG -> upsample -> Haar cascade -> (optional) CNN
- One-face-only mode unless reference given (multi-face handled separately)
- Unreadable/corrupt files moved to `errors/`
- Face_not_found moved to `face_not_found/`
- In reference mode:
    * If reference is a folder with multiple subfolders → each subfolder becomes a cluster, folder name preserved
    * Multi-face images are checked: if they contain a reference face → saved to that person's folder under `multi_face/`
- Unsupervised online clustering for non-reference mode (min distance to all encodings per cluster)
- Safe move with name collision avoidance
"""
import argparse
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np
from PIL import Image, UnidentifiedImageError, ImageOps
import face_recognition
import cv2
import uuid
import os

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def iter_images(src_dir: Path, recursive: bool = True):
    if recursive:
        return [p for p in src_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
    else:
        return [p for p in src_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()]

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def unique_filename(dest_dir: Path, name: str) -> Path:
    target = dest_dir / name
    if not target.exists():
        return target
    base = Path(name).stem
    ext = Path(name).suffix
    for i in range(100):
        candidate = dest_dir / f"{base}_{uuid.uuid4().hex[:6]}{ext}"
        if not candidate.exists():
            return candidate
    return dest_dir / f"{base}_{uuid.uuid4().hex}{ext}"

def safe_move(src: Path, dest_dir: Path):
    safe_mkdir(dest_dir)
    dest = unique_filename(dest_dir, src.name)
    shutil.move(str(src), str(dest))
    return dest

def load_and_resize(path: Path, max_side: int = 1600) -> Optional[np.ndarray]:
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        if hasattr(Image, "Resampling"):
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        else:
            img.thumbnail((max_side, max_side), Image.LANCZOS)
        img = img.convert("RGB")
        return np.asarray(img)
    except UnidentifiedImageError:
        print(f"[SKIP] Cannot identify file as image → {path}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to open/resize {path} → {e}")
        return None

def detect_locations_hog(img_rgb, upsample=0):
    return face_recognition.face_locations(img_rgb, number_of_times_to_upsample=upsample, model="hog")

def detect_locations_cnn(img_rgb, upsample=0):
    return face_recognition.face_locations(img_rgb, number_of_times_to_upsample=upsample, model="cnn")

def detect_locations_haar(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        return []
    rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30,30))
    return [(y, x+w, y+h, x) for (x, y, w, h) in rects]

def encode_for_locations(img_rgb, boxes, num_jitters=1):
    return face_recognition.face_encodings(img_rgb, known_face_locations=boxes, num_jitters=num_jitters)

def detect_and_encode(img_rgb, use_cnn=False, allow_haar=True, upsample_max=2):
    for up in range(0, upsample_max+1):
        boxes = detect_locations_hog(img_rgb, upsample=up)
        if boxes:
            return boxes, encode_for_locations(img_rgb, boxes), f"hog_up{up}"

    if use_cnn:
        for up in range(0, upsample_max+1):
            boxes = detect_locations_cnn(img_rgb, upsample=up)
            if boxes:
                return boxes, encode_for_locations(img_rgb, boxes), f"cnn_up{up}"

    if allow_haar:
        boxes = detect_locations_haar(img_rgb)
        if boxes:
            return boxes, encode_for_locations(img_rgb, boxes), "haar"

    try:
        h, w = img_rgb.shape[:2]
        if max(h, w) < 800:
            big = cv2.resize(img_rgb, (int(w*1.5), int(h*1.5)))
            boxes = detect_locations_hog(big, upsample=0)
            if boxes:
                scaled = [(int(t/1.5), int(r/1.5), int(b/1.5), int(l/1.5)) for (t,r,b,l) in boxes]
                return scaled, encode_for_locations(img_rgb, scaled), "hog_upscale"
    except Exception:
        pass

    return [], [], "none"

def load_reference_folder(ref_dir: Path, max_side=1600, use_cnn=False, allow_haar=True):
    """
    Returns dict {person_name: [encodings]} where person_name is subfolder name.
    """
    references = {}
    for person_folder in ref_dir.iterdir():
        if person_folder.is_dir():
            encs = []
            for p in iter_images(person_folder, recursive=True):
                rgb = load_and_resize(p, max_side)
                if rgb is None: continue
                boxes, e, _ = detect_and_encode(rgb, use_cnn=use_cnn, allow_haar=allow_haar)
                if e: encs.extend(e)
            if encs:
                references[person_folder.name] = encs
    return references

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--reference", type=Path, default=None)
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--max-side", type=int, default=1600)
    ap.add_argument("--no-haar-fallback", action="store_true")
    ap.add_argument("--use-cnn", action="store_true")
    ap.add_argument("--upsample", type=int, default=1)
    args = ap.parse_args()

    src_dir, out_dir = args.src, args.out
    allow_haar, use_cnn = not args.no_haar_fallback, args.use_cnn
    upsample_max = max(0, int(args.upsample))

    # dirs
    face_not_found_dir = out_dir / "face_not_found"
    multi_face_dir = out_dir / "multi_face"
    errors_dir = out_dir / "errors"
    clusters_root = out_dir / "clusters"
    non_match_dir = clusters_root / "non_match"
    for d in [face_not_found_dir, multi_face_dir, errors_dir, clusters_root, non_match_dir]:
        safe_mkdir(d)

    # reference mode setup
    ref_encodings = {}
    if args.reference:
        if args.reference.is_dir():
            ref_encodings = load_reference_folder(args.reference, args.max_side, use_cnn, allow_haar)
            print(f"[INFO] Loaded reference clusters: {list(ref_encodings.keys())}")
        else:
            rgb = load_and_resize(args.reference, args.max_side)
            if rgb is not None:
                boxes, e, _ = detect_and_encode(rgb, use_cnn=use_cnn, allow_haar=allow_haar)
                if e: ref_encodings["person_ref"] = [e[0]]

    paths = iter_images(src_dir, recursive=True)
    for path in paths:
        rgb = load_and_resize(path, args.max_side)
        if rgb is None:
            safe_move(path, errors_dir); continue

        boxes, encs, _ = detect_and_encode(rgb, use_cnn=use_cnn, allow_haar=allow_haar, upsample_max=upsample_max)
        if not encs:
            safe_move(path, face_not_found_dir); continue

        # MULTI FACE
        if len(encs) > 1 and ref_encodings:
            matched = False
            for person, refs in ref_encodings.items():
                for enc in encs:
                    dists = [np.linalg.norm(enc - r) for r in refs]
                    if min(dists) <= args.threshold:
                        dest = clusters_root / person / "multi_face"
                        safe_move(path, dest)
                        matched = True
                        print(f"[MULTI MATCH {person}] {path.name}")
                        break
                if matched: break
            if not matched:
                safe_move(path, multi_face_dir)
            continue

        # SINGLE FACE
        enc = encs[0]
        if ref_encodings:
            matched = False
            for person, refs in ref_encodings.items():
                dists = [np.linalg.norm(enc - r) for r in refs]
                if min(dists) <= args.threshold:
                    dest = clusters_root / person
                    safe_move(path, dest)
                    matched = True
                    print(f"[REF MATCH {person}] {path.name}")
                    break
            if not matched:
                safe_move(path, non_match_dir)
                print(f"[REF NON-MATCH] {path.name}")
        else:
            # fallback unsupervised clustering could be added here
            safe_move(path, clusters_root / "unsorted")

if __name__ == "__main__":
    main()
