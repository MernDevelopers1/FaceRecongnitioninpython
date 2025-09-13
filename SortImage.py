#!/usr/bin/env python3
"""
Robust image sorter using face_recognition.

Features:
- Recursive scan of source folder for images
- Multi-pass face detection: HOG -> upsample -> Haar cascade -> (optional) CNN
- One-face-only mode: multi-face images are moved to `multi_face/`
- Unreadable/corrupt files moved to `errors/`
- face_not_found moved to `face_not_found/`
- Unsupervised online clustering that compares new encoding to ALL encodings in each cluster (min distance)
- Optional reference mode (moves matches to a single `person_matches/` folder)
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

# allowed image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ---------------------------
# Utilities
# ---------------------------
def iter_images(src_dir: Path, recursive: bool = True):
    if recursive:
        return [p for p in src_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
    else:
        return [p for p in src_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()]

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def unique_filename(dest_dir: Path, name: str) -> Path:
    """
    If dest_dir/name exists, append a short uuid before extension to avoid collisions.
    """
    target = dest_dir / name
    if not target.exists():
        return target
    base = Path(name).stem
    ext = Path(name).suffix
    for i in range(100):
        candidate = dest_dir / f"{base}_{uuid.uuid4().hex[:6]}{ext}"
        if not candidate.exists():
            return candidate
    # fallback (shouldn't happen)
    return dest_dir / f"{base}_{uuid.uuid4().hex}{ext}"

def safe_move(src: Path, dest_dir: Path):
    safe_mkdir(dest_dir)
    dest = unique_filename(dest_dir, src.name)
    shutil.move(str(src), str(dest))
    return dest

# ---------------------------
# Image loading & resizing
# ---------------------------
def load_and_resize(path: Path, max_side: int = 1600) -> Optional[np.ndarray]:
    """
    Load image with PIL, fix EXIF orientation, resize so largest side <= max_side,
    convert to RGB numpy array. Returns None for unreadable files.
    """
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        # use high-quality resampling
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

# ---------------------------
# Detection & encoding
# ---------------------------
def detect_locations_hog(img_rgb: np.ndarray, upsample: int = 0) -> List[Tuple[int,int,int,int]]:
    return face_recognition.face_locations(img_rgb, number_of_times_to_upsample=upsample, model="hog")

def detect_locations_cnn(img_rgb: np.ndarray, upsample: int = 0) -> List[Tuple[int,int,int,int]]:
    return face_recognition.face_locations(img_rgb, number_of_times_to_upsample=upsample, model="cnn")

def detect_locations_haar(img_rgb: np.ndarray) -> List[Tuple[int,int,int,int]]:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        return []
    rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30,30))
    boxes = []
    for (x, y, w, h) in rects:
        # convert to (top, right, bottom, left)
        boxes.append((y, x + w, y + h, x))
    return boxes

def encode_for_locations(img_rgb: np.ndarray, boxes: List[Tuple[int,int,int,int]], num_jitters:int = 1) -> List[np.ndarray]:
    encs = face_recognition.face_encodings(img_rgb, known_face_locations=boxes, num_jitters=num_jitters)
    return encs

def detect_and_encode(img_rgb: np.ndarray, use_cnn=False, allow_haar=True, upsample_max=2) -> Tuple[List[Tuple[int,int,int,int]], List[np.ndarray], str]:
    """
    Try multiple detection approaches and return:
      (locations_list, encodings_list, method_string)
    where method_string describes which method found faces.
    """
    # 1) HOG with upsample 0..upsample_max
    for up in range(0, upsample_max+1):
        boxes = detect_locations_hog(img_rgb, upsample=up)
        if boxes:
            encs = encode_for_locations(img_rgb, boxes)
            return boxes, encs, f"hog_up{up}"

    # 2) optional CNN (if user requested/useable)
    if use_cnn:
        for up in range(0, upsample_max+1):
            boxes = detect_locations_cnn(img_rgb, upsample=up)
            if boxes:
                encs = encode_for_locations(img_rgb, boxes)
                return boxes, encs, f"cnn_up{up}"

    # 3) Haar fallback
    if allow_haar:
        boxes = detect_locations_haar(img_rgb)
        if boxes:
            encs = encode_for_locations(img_rgb, boxes)
            return boxes, encs, "haar"

    # 4) try simple upscale (useful for very small heads) - upscale x1.5 and try HOG once
    try:
        h, w = img_rgb.shape[:2]
        scale = 1.5
        if max(h, w) < 800:
            big = cv2.resize(img_rgb, (int(w*scale), int(h*scale)))
            boxes = detect_locations_hog(big, upsample=0)
            if boxes:
                # convert boxes back to original scale
                scaled_boxes = []
                for (top, right, bottom, left) in boxes:
                    scaled_boxes.append((int(top/scale), int(right/scale), int(bottom/scale), int(left/scale)))
                encs = encode_for_locations(img_rgb, scaled_boxes)
                return scaled_boxes, encs, "hog_upscale"
    except Exception:
        pass

    return [], [], "none"

# ---------------------------
# Clustering
# ---------------------------
class Cluster:
    def __init__(self, cid: int, first_encoding: np.ndarray):
        self.id = f"person_{cid:03d}"
        self.encodings = [first_encoding]  # store list of examples
    def min_distance(self, enc: np.ndarray) -> float:
        dists = [np.linalg.norm(e - enc) for e in self.encodings]
        return float(min(dists))
    def add_encoding(self, enc: np.ndarray):
        self.encodings.append(enc)

# ---------------------------
# Reference loader
# ---------------------------
def load_reference_encoding(ref: Path, max_side:int=1600, use_cnn=False, allow_haar=True) -> Optional[np.ndarray]:
    """
    If ref is a folder, average all valid encodings found there. If file, return single encoding.
    """
    if not ref.exists():
        print(f"[WARN] Reference path not found: {ref}")
        return None
    if ref.is_dir():
        encs = []
        for p in iter_images(ref, recursive=True):
            rgb = load_and_resize(p, max_side=max_side)
            if rgb is None:
                continue
            boxes, encodings, _ = detect_and_encode(rgb, use_cnn=use_cnn, allow_haar=allow_haar)
            if encodings:
                encs.extend(encodings)
        if not encs:
            print("[WARN] No valid encodings in reference folder.")
            return None
        return np.mean(np.vstack(encs), axis=0)
    else:
        rgb = load_and_resize(ref, max_side=max_side)
        if rgb is None:
            return None
        boxes, encs, _ = detect_and_encode(rgb, use_cnn=use_cnn, allow_haar=allow_haar)
        if not encs:
            print("[WARN] No face encoding in reference image.")
            return None
        return encs[0]

# ---------------------------
# Main
# ---------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, type=Path, help="Source root folder (will scan subfolders).")
    p.add_argument("--out", required=True, type=Path, help="Output root folder.")
    p.add_argument("--reference", type=Path, default=None, help="Optional reference image or folder.")
    p.add_argument("--threshold", type=float, default=0.45, help="Distance threshold (lower = stricter).")
    p.add_argument("--max-side", type=int, default=1600, help="Max image side to resize (keeps memory low).")
    p.add_argument("--no-haar-fallback", action="store_true", help="Disable Haar fallback.")
    p.add_argument("--use-cnn", action="store_true", help="Use dlib CNN detector (requires dlib built with CUDA; may OOM).")
    p.add_argument("--upsample", type=int, default=1, help="Max hog upsample attempts (0..n).")
    args = p.parse_args()

    src_dir = args.src
    out_dir = args.out
    ref = args.reference
    threshold = args.threshold
    max_side = args.max_side
    allow_haar = not args.no_haar_fallback
    use_cnn = args.use_cnn
    upsample_max = max(0, int(args.upsample))

    # make bins
    face_not_found_dir = out_dir / "face_not_found"
    multi_face_dir = out_dir / "multi_face"
    errors_dir = out_dir / "errors"
    clusters_root = out_dir / "clusters"
    person_matches_dir = out_dir / "person_matches"
    for d in [face_not_found_dir, multi_face_dir, errors_dir, clusters_root, person_matches_dir]:
        safe_mkdir(d)

    # load reference if provided
    reference_encoding = None
    if ref is not None:
        print("[INFO] Loading reference encoding...")
        reference_encoding = load_reference_encoding(ref, max_side=max_side, use_cnn=use_cnn, allow_haar=allow_haar)
        if reference_encoding is None:
            print("[WARN] Reference provided but no encodings found. Falling back to unsupervised clustering.")
            ref = None

    detector_type = "cnn" if use_cnn else "hog"
    print(f"[INFO] Detector: {detector_type}  |  Max upsample: {upsample_max}  |  Haar fallback: {allow_haar}")

    paths = iter_images(src_dir, recursive=True)
    print(f"[INFO] Found {len(paths)} images (recursive).")

    clusters: List[Cluster] = []
    moved = 0
    stats = {"no_face":0, "multi":0, "errors":0, "matched_ref":0, "cluster_new":0, "cluster_add":0}

    for path in paths:
        rgb = load_and_resize(path, max_side=max_side)
        if rgb is None:
            # move to errors
            try:
                dest = safe_move(path, errors_dir)
                stats["errors"] += 1
                print(f"[ERROR->moved] Unreadable image → {path.name}")
            except Exception as e:
                print(f"[ERROR] Moving unreadable {path.name} failed -> {e}")
            continue

        try:
            boxes, encs, method = detect_and_encode(rgb, use_cnn=use_cnn, allow_haar=allow_haar, upsample_max=upsample_max)
        except Exception as e:
            print(f"[ERROR] Detection/encoding failed for {path.name} -> {e}")
            safe_move(path, errors_dir)
            stats["errors"] += 1
            continue

        if len(encs) == 0:
            # no face detected
            safe_move(path, face_not_found_dir)
            stats["no_face"] += 1
            print(f"[NO FACE] → {path.name}")
            continue

        if len(encs) > 1:
            # multi-face image
            safe_move(path, multi_face_dir)
            stats["multi"] += 1
            print(f"[MULTI] → {path.name} (faces={len(encs)})")
            continue

        # now we have exactly 1 encoding
        enc = encs[0]

        # reference mode
        if reference_encoding is not None:
            dist = float(np.linalg.norm(enc - reference_encoding))
            if dist <= threshold:
                safe_move(path, person_matches_dir)
                stats["matched_ref"] += 1
                print(f"[REF MATCH d={dist:.3f}] → {path.name}")
            else:
                safe_move(path, clusters_root / "non_match")
                print(f"[REF NON-MATCH d={dist:.3f}] → {path.name}")
            continue

        # unsupervised clustering (compare to all encodings in each cluster by min distance)
        best_cluster_idx = None
        best_cluster_dist = float("inf")
        for idx, c in enumerate(clusters):
            d = c.min_distance(enc)
            if d < best_cluster_dist:
                best_cluster_dist = d
                best_cluster_idx = idx

        if best_cluster_idx is not None and best_cluster_dist <= threshold:
            clusters[best_cluster_idx].add_encoding(enc)
            dest = clusters_root / clusters[best_cluster_idx].id
            safe_move(path, dest)
            stats["cluster_add"] += 1
            print(f"[ADD {clusters[best_cluster_idx].id} d={best_cluster_dist:.3f}] → {path.name}")
        else:
            # new cluster
            cid = len(clusters) + 1
            c = Cluster(cid, enc)
            clusters.append(c)
            dest = clusters_root / c.id
            safe_move(path, dest)
            stats["cluster_new"] += 1
            print(f"[NEW {c.id} d={best_cluster_dist:.3f}] → {path.name}")

    total_moved = sum(stats.values())
    print("\n[SUMMARY]")
    print(f"Total images scanned: {len(paths)}")
    print(f"Moved to clusters: {stats['cluster_new'] + stats['cluster_add']}")
    print(f"Moved to ref matches: {stats['matched_ref']}")
    print(f"Multi-face: {stats['multi']}  |  No face: {stats['no_face']}  |  Errors: {stats['errors']}")
    print(f"Clusters created: {len(clusters)}")
    print("Done.")

if __name__ == "__main__":
    main()
