import os
import shutil
import numpy as np
from deepface import DeepFace
from sklearn.cluster import DBSCAN

# Paths
SOURCE_DIR = "/mnt/d/Sorted/clusters/person_001"
OUTPUT_DIR = "/mnt/d/SortedImages"

os.makedirs(OUTPUT_DIR, exist_ok=True)
NO_FACE_DIR = os.path.join(OUTPUT_DIR, "no_face")
ERROR_DIR = os.path.join(OUTPUT_DIR, "errors")
os.makedirs(NO_FACE_DIR, exist_ok=True)
os.makedirs(ERROR_DIR, exist_ok=True)

# Step 1: Build embeddings for all images
embeddings = []
filenames = []

for root, _, files in os.walk(SOURCE_DIR):
    for file in files:
        path = os.path.join(root, file)
        try:
            reps = DeepFace.represent(
                img_path=path,
                detector_backend="retinaface",
                model_name="ArcFace",
                enforce_detection=True
            )
            
            for rep in reps:  # in case of multiple faces
                embeddings.append(rep["embedding"])
                filenames.append(path)
        
        except Exception as e:
            if "Face could not be detected" in str(e):
                shutil.move(path, NO_FACE_DIR)
            else:
                shutil.move(path, ERROR_DIR)

# Step 2: Cluster embeddings
if embeddings:
    embeddings = np.array(embeddings)
    cluster = DBSCAN(eps=0.6, min_samples=2, metric="cosine").fit(embeddings)
    labels = cluster.labels_

    # Step 3: Move images into folders
    for label, file in zip(labels, filenames):
        if label == -1:
            shutil.move(file, NO_FACE_DIR)
        else:
            person_dir = os.path.join(OUTPUT_DIR, f"person_{str(label).zfill(3)}")
            os.makedirs(person_dir, exist_ok=True)
            shutil.move(file, person_dir)

    print(f"[DONE] Moved {len(filenames)} images into {len(set(labels))} clusters.")
else:
    print("No embeddings found.")
