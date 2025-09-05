import cv2
import numpy as np
import face_recognition
from deepface import DeepFace
from flask import Blueprint, request, jsonify

compare_faces_bp7 = Blueprint("compare_faces_all", __name__)

# DeepFace supported models
DEEPFACE_MODELS = [
    "VGG-Face", "Facenet", "Facenet512",
    "OpenFace", "DeepFace", "DeepID",
    "Dlib", "ArcFace", "SFace"
]

# Preprocessing variations
def preprocess_variations(image_bytes):
    """Return multiple preprocessing variations of the same image."""
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
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


def compare_with_deepface(image1_bytes, image2_bytes):
    """Compare using all DeepFace models and all preprocessing variations."""
    results = []
    img1_variations = preprocess_variations(image1_bytes)
    img2_variations = preprocess_variations(image2_bytes)

    for model in DEEPFACE_MODELS:
        for var_name, img1 in img1_variations.items():
            img2 = img2_variations[var_name]  # same variation for both
            try:
                result = DeepFace.verify(
                    img1, img2,
                    model_name=model,
                    enforce_detection=False
                )
                distance = result["distance"]
                face_match_percentage = (1 - distance) * 100
                match = distance < 0.6

                results.append({
                    "library": "DeepFace",
                    "model": model,
                    "variation": var_name,
                    "match": str(match),
                    "distance": distance,
                    "face_match": face_match_percentage,
                    "msg": "Face Matched" if match else "Face does not match"
                })

            except Exception as e:
                results.append({
                    "library": "DeepFace",
                    "model": model,
                    "variation": var_name,
                    "error": str(e)
                })
    return results


def compare_with_face_recognition(image1_bytes, image2_bytes):
    """Compare using face_recognition with preprocessing variations."""
    results = []
    img1_variations = preprocess_variations(image1_bytes)
    img2_variations = preprocess_variations(image2_bytes)

    for var_name, img1 in img1_variations.items():
        img2 = img2_variations[var_name]
        try:
            enc1 = face_recognition.face_encodings(img1)
            enc2 = face_recognition.face_encodings(img2)

            if len(enc1) > 0 and len(enc2) > 0:
                distance = face_recognition.face_distance([enc1[0]], enc2[0])[0]
                face_match_percentage = (1 - distance) * 100
                match = distance < 0.6

                results.append({
                    "library": "face_recognition",
                    "model": "dlib",
                    "variation": var_name,
                    "match": str(match),
                    "distance": float(distance),
                    "face_match": face_match_percentage,
                    "msg": "Face Matched" if match else "Face does not match"
                })
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
            results.append({
                "library": "face_recognition",
                "model": "dlib",
                "variation": var_name,
                "error": str(e)
            })
    return results


@compare_faces_bp7.route("/compare_faces_all", methods=["POST"])
def compare_faces_all():
    if "image1" not in request.files or "image2" not in request.files:
        return jsonify({"error": "Missing required fields: image1 and image2"}), 400

    try:
        image1_bytes = request.files["image1"].read()
        image2_bytes = request.files["image2"].read()

        # Run both DeepFace and face_recognition
        deepface_results = compare_with_deepface(image1_bytes, image2_bytes)
        fr_results = compare_with_face_recognition(image1_bytes, image2_bytes)

        all_results = deepface_results + fr_results

        # Sort by best face match %
        sorted_results = sorted(
            all_results,
            key=lambda x: x.get("face_match", 0),
            reverse=True
        )

        return jsonify({
            "num_faces_detected": 2,
            "best_result": sorted_results[0] if sorted_results else {},
            "all_results": sorted_results
        }), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Error occurred during face comparison!"}), 500
