import cv2
import numpy as np
import face_recognition
from deepface import DeepFace
from flask import Blueprint, request, jsonify

compare_faces_bp7 = Blueprint("compare_faces_all", __name__)

# DeepFace supported models
DEEPFACE_MODELS = [
    # "VGG-Face",
     "Facenet", 
    #  "Facenet512",
    # "OpenFace", 
    # "DeepFace",
    #  "DeepID",
    # "Dlib", "ArcFace", "SFace"
]

# ======================
# Face detection utility
# ======================
def detect_faces(image_bytes):
    """Return face locations and cropped face images for comparison."""
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb)  # list of (top, right, bottom, left)
    faces = []

    for i, (top, right, bottom, left) in enumerate(face_locations):
        cropped = img[top:bottom, left:right]
        faces.append({
            "id": f"person_{i+1}",
            "location": {"top": top, "right": right, "bottom": bottom, "left": left},
            "image": cropped
        })

    return faces


# ======================
# Preprocessing variations
# ======================
def preprocess_variations(img):
    """Return multiple preprocessing variations of the same face image."""
    variations = {}

    # Original
    variations["original"] = img.copy()

    # Grayscale (convert back to 3 channels)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variations["grayscale"] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Histogram Equalization (per channel)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    variations["hist_equalized"] = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # Gaussian Blur Reduced (sharpen by subtracting blur)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    sharpened = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    variations["sharpened"] = sharpened

    # Resized (224x224)
    resized = cv2.resize(img, (224, 224))
    variations["resized"] = resized

    return variations


# ======================
# Comparisons
# ======================
def compare_with_deepface(face1, face2):
    results = []

    # Only original faces (no variations)
    try:
        for model in DEEPFACE_MODELS:
            result = DeepFace.verify(
                face1, face2,
                model_name=model,
                enforce_detection=False
            )
            distance = result["distance"]
            face_match_percentage = (1 - distance) * 100
            match = distance < 0.6

            record = {
                "library": "DeepFace",
                "model": model,
                "variation": "original",   # always original
                "match": str(match),
                "distance": distance,
                "face_match": face_match_percentage,
                "msg": "Face Matched" if match else "Face does not match"
            }

            print(f"[DeepFace][{model}][original] → {record}")
            results.append(record)

    except Exception as e:
        print(f"[DeepFace][ERROR] → {e}")
        results.append({
            "library": "DeepFace",
            "model": model,
            "variation": "original",
            "error": str(e)
        })

    return results


def compare_with_face_recognition(face1, face2):
    results = []
    img1_variations = preprocess_variations(face1)
    img2_variations = preprocess_variations(face2)

    for var_name, img1 in img1_variations.items():
        img2 = img2_variations[var_name]
        try:
            enc1 = face_recognition.face_encodings(img1)
            enc2 = face_recognition.face_encodings(img2)

            if len(enc1) > 0 and len(enc2) > 0:
                distance = face_recognition.face_distance([enc1[0]], enc2[0])[0]
                face_match_percentage = (1 - distance) * 100
                match = distance < 0.6

                record = {
                    "library": "face_recognition",
                    "model": "dlib",
                    "variation": var_name,
                    "match": str(match),
                    "distance": float(distance),
                    "face_match": face_match_percentage,
                    "msg": "Face Matched" if match else "Face does not match"
                }

                print(f"[face_recognition][dlib][{var_name}] → {record}")
                results.append(record)
            else:
                results.append({
                    "library": "face_recognition",
                    "model": "dlib",
                    "variation": var_name,
                    "match": "False",
                    "distance": 0,
                    "face_match": 0,
                    "msg": "No face detected"
                })
        except Exception as e:
            print(f"[face_recognition][dlib][{var_name}] ERROR → {e}")
            results.append({
                "library": "face_recognition",
                "model": "dlib",
                "variation": var_name,
                "error": str(e)
            })
    return results


# ======================
# Flask Route
# ======================
@compare_faces_bp7.route("/compare_faces_all", methods=["POST"])
def compare_faces_all():
    # return {"message": "CORS fixed!"}
    if "image1" not in request.files or "image2" not in request.files:
        return jsonify({"error": "Missing required fields: image1 and image2"}), 400

    try:
        image1_bytes = request.files["image1"].read()
        image2_bytes = request.files["image2"].read()

        # Detect faces
        faces1 = detect_faces(image1_bytes)
        faces2 = detect_faces(image2_bytes)

        all_comparisons = []

        for f1 in faces1:
            for f2 in faces2:
                deepface_results = compare_with_deepface(f1["image"], f2["image"])
                fr_results = compare_with_face_recognition(f1["image"], f2["image"])
                combined = deepface_results + fr_results

                # Sort by best match
                sorted_results = sorted(combined, key=lambda x: x.get("face_match", 0), reverse=True)

                all_comparisons.append({
                    "face1_id": f1["id"],
                    "face1_location": f1["location"],
                    "face2_id": f2["id"],
                    "face2_location": f2["location"],
                    "best_result": sorted_results[0] if sorted_results else {},
                    "all_results": sorted_results
                })

        return jsonify({
            "faces_in_image1": len(faces1),
            "faces_in_image2": len(faces2),
            "comparisons": all_comparisons
        }), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Error occurred during face comparison!"}), 500
