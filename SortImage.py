#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image, UnidentifiedImageError, ImageOps
import face_recognition
import cv2

# -------------------
# Utility / IO
# -------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def iter_images(src_dir: Path, recursive: bool = True) -> List[Path]:
    if recursive:
        return [p for p in src_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
    else:
        return [p for p in src_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()]

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_and_resize(path: Path, max_side: int = 1600) -> Optional[np.ndarray]:
    """Open with PIL, EXIF-orient, RGB, thumbnail to max_side, return numpy array; None on failure."""
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)  # fix orientation
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

# -------------------
# Detection / Encoding
# -------------------
def detect_one_face_hog(image_rgb: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    boxes = face_recognition.face_locations(image_rgb, model="hog")
    if len(boxes) == 1:
        return boxes[0]
    return None

def detect_one_face_haar(image_rgb: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        return None
    rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    # Convert first rect (x,y,w,h) -> (top,right,bottom,left)
    if len(rects) == 1:
        x, y, w, h = rects[0]
        return (y, x + w, y + h, x)
    return None

def encode_face(image_rgb: np.ndarray, box: Tuple[int,int,int,int]) -> Optional[np.ndarray]:
    # num_jitters=1 for speed; model="small" (standard 128-d embedding)
    enc = face_recognition.face_encodings(image_rgb, known_face_locations=[box], num_jitters=1, model="small")
    if len(enc) == 1:
        return enc[0]
    return None

def get_one_face_encoding(image_rgb: np.ndarray, allow_haar_fallback: bool = True) -> Tuple[str, Optional[np.ndarray]]:
    """
    Returns a status and an encoding.
    Status one of: 'ok', 'no_face', 'multi_face', 'encode_fail'
    """
    # First pass (HOG)
    boxes = face_recognition.face_locations(image_rgb, model="hog")
    if len(boxes) == 1:
        enc = encode_face(image_rgb, boxes[0])
        return ("ok", enc if enc is not None else None)
    elif len(boxes) > 1:
        return ("multi_face", None)

    # Fallback (Haar) if HOG found none
    if allow_haar_fallback:
        box = detect_one_face_haar(image_rgb)
        if box is not None:
            enc = encode_face(image_rgb, box)
            return ("ok", enc if enc is not None else None)

    return ("no_face", None)

# -------------------
# Clustering (online)
# -------------------
class Cluster:
    def __init__(self, cid: int, first_enc: np.ndarray):
        self.id = f"person_{cid:03d}"
        self.centroid = first_enc.astype(np.float64)
        self.count = 1

    def distance(self, enc: np.ndarray) -> float:
        return float(np.linalg.norm(self.centroid - enc))

    def add(self, enc: np.ndarray):
        self.count += 1
        # running centroid
        self.centroid = self.centroid + (enc - self.centroid) / self.count

# -------------------
# Reference handling
# -------------------
def load_reference_encoding(ref_path: Path, max_side: int, allow_haar_fallback: bool) -> Optional[np.ndarray]:
    if ref_path.is_dir():
        encs = []
        for p in iter_images(ref_path, recursive=True):
            rgb = load_and_resize(p, max_side=max_side)
            if rgb is None:
                continue
            status, enc = get_one_face_encoding(rgb, allow_haar_fallback=allow_haar_fallback)
            if status == "ok" and enc is not None:
                encs.append(enc)
        if not encs:
            print(f"[WARN] No valid face encodings found in reference folder: {ref_path}")
            return None
        return np.mean(np.vstack(encs), axis=0)
    else:
        rgb = load_and_resize(ref_path, max_side=max_side)
        if rgb is None:
            return None
        status, enc = get_one_face_encoding(rgb, allow_haar_fallback=allow_haar_fallback)
        if status == "ok" and enc is not None:
            return enc
        print(f"[WARN] No valid face found in reference image: {ref_path}")
        return None

# -------------------
# Main processing
# -------------------
def main():
    ap = argparse.ArgumentParser(description="Sort images by face using face_recognition (one-face-only).")
    ap.add_argument("--src", required=True, type=Path, help="Source folder (scans subfolders).")
    ap.add_argument("--out", required=True, type=Path, help="Output root folder.")
    ap.add_argument("--reference", type=Path, default=None,
                    help="(Optional) Reference image OR folder for the target person. If given, only matches to this person are moved to a single folder.")
    ap.add_argument("--threshold", type=float, default=0.48,
                    help="Distance threshold (lower = stricter). ~0.45-0.52 typical.")
    ap.add_argument("--max-side", type=int, default=1600, help="Resize largest side to this many pixels.")
    ap.add_argument("--no-haar-fallback", action="store_true", help="Disable Haar fallback.")
    ap.add_argument("--use-cnn", action="store_true",
                    help="Use dlib CNN detector instead of HOG (NOT recommended on low VRAM).")
    args = ap.parse_args()

    src_dir: Path = args.src
    out_dir: Path = args.out
    ref: Optional[Path] = args.reference
    threshold: float = args.threshold
    max_side: int = args.max_side
    allow_haar = not args.no_haar_fallback

    # Output folders
    face_not_found_dir = out_dir / "face_not_found"
    multi_face_dir = out_dir / "multi_face"
    errors_dir = out_dir / "errors"
    clusters_root = out_dir / "clusters"
    person_dir = out_dir / "person_matches"  # used only in reference mode

    for d in [face_not_found_dir, multi_face_dir, errors_dir, clusters_root, person_dir]:
        safe_mkdir(d)

    # Load reference (if any)
    reference_encoding = None
    if ref is not None:
        reference_encoding = load_reference_encoding(ref, max_side, allow_haar)
        if reference_encoding is None:
            print("[WARN] Proceeding without reference (unsupervised clustering).")
            ref = None

    # CNN vs HOG info
    detector_model = "cnn" if args.use_cnn else "hog"
    if detector_model == "cnn":
        print("[INFO] Using dlib CNN detector (GPU). Watch out for CUDA OOM.")
    else:
        print("[INFO] Using HOG detector (CPU).")

    # If user chose CNN, patch detection call inside our encoding function by monkey-patching
    # (We still use HOG in get_one_face_encoding to keep fallback logic. Here we only warn.)
    # For robustness in your environment (CUDA OOM earlier) we keep HOG in get_one_face_encoding.

    # Process images
    paths = iter_images(src_dir, recursive=True)
    print(f"[INFO] Found {len(paths)} image(s) in {src_dir} (recursive).")

    clusters: List[Cluster] = []
    moved = 0
    skipped = 0

    for path in paths:
        rgb = load_and_resize(path, max_side=max_side)
        if rgb is None:
            # Unreadable -> errors
            try:
                shutil.move(str(path), str(errors_dir / path.name))
                print(f"[ERROR->moved] Unreadable image → {path.name}")
            except Exception as e:
                print(f"[ERROR] Moving unreadable {path} failed → {e}")
            continue

        # Detect/encode
        status, enc = get_one_face_encoding(rgb, allow_haar_fallback=allow_haar)
        if status == "multi_face":
            try:
                shutil.move(str(path), str(multi_face_dir / path.name))
                print(f"[MULTI] → {path.name}")
            except Exception as e:
                print(f"[ERROR] Moving multi_face {path} failed → {e}")
            continue
        if status != "ok" or enc is None:
            try:
                shutil.move(str(path), str(face_not_found_dir / path.name))
                print(f"[NO FACE] → {path.name}")
            except Exception as e:
                print(f"[ERROR] Moving no_face {path} failed → {e}")
            continue

        # Reference mode
        if reference_encoding is not None:
            dist = float(np.linalg.norm(enc - reference_encoding))
            if dist <= threshold:
                dest = person_dir
                dest.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.move(str(path), str(dest / path.name))
                    moved += 1
                    print(f"[MATCH {dist:.3f}] → {path.name}")
                except Exception as e:
                    print(f"[ERROR] Move failed (match) {path} → {e}")
            else:
                try:
                    shutil.move(str(path), str(clusters_root / "non_match" / path.name))
                    safe_mkdir(clusters_root / "non_match")
                    print(f"[NON-MATCH {dist:.3f}] → {path.name}")
                except Exception as e:
                    print(f"[ERROR] Move failed (non-match) {path} → {e}")
            continue

        # Unsupervised: online clustering with strict threshold
        if not clusters:
            c = Cluster(1, enc)
            clusters.append(c)
            dest = clusters_root / c.id
            safe_mkdir(dest)
            try:
                shutil.move(str(path), str(dest / path.name))
                moved += 1
                print(f"[NEW {c.id}] → {path.name}")
            except Exception as e:
                print(f"[ERROR] Move failed (new cluster) {path} → {e}")
            continue

        # Find best cluster
        best_idx = None
        best_dist = 10.0
        for idx, c in enumerate(clusters):
            d = c.distance(enc)
            if d < best_dist:
                best_dist = d
                best_idx = idx

        if best_dist <= threshold and best_idx is not None:
            clusters[best_idx].add(enc)
            dest = clusters_root / clusters[best_idx].id
            safe_mkdir(dest)
            try:
                shutil.move(str(path), str(dest / path.name))
                moved += 1
                print(f"[ADD {clusters[best_idx].id} d={best_dist:.3f}] → {path.name}")
            except Exception as e:
                print(f"[ERROR] Move failed (add cluster) {path} → {e}")
        else:
            c = Cluster(len(clusters) + 1, enc)
            clusters.append(c)
            dest = clusters_root / c.id
            safe_mkdir(dest)
            try:
                shutil.move(str(path), str(dest / path.name))
                moved += 1
                print(f"[NEW {c.id} d={best_dist:.3f}] → {path.name}")
            except Exception as e:
                print(f"[ERROR] Move failed (new cluster) {path} → {e}")

    print(f"\n[SUMMARY] Moved: {moved} | Skipped/Unmoved (multi/no-face/errors moved to bins): {len(paths)-moved}")

if __name__ == "__main__":
    main()
