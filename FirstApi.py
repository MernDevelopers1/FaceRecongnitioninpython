import cv2
import face_recognition
from flask import request, jsonify, Blueprint
import numpy as np
from deepface import DeepFace
from concurrent.futures import ThreadPoolExecutor


def fallback_to_deepface(image1_bytes, image2_bytes):
    image1 = cv2.imdecode(np.frombuffer(image1_bytes, np.uint8), cv2.IMREAD_COLOR)
    image2 = cv2.imdecode(np.frombuffer(image2_bytes, np.uint8), cv2.IMREAD_COLOR)

    try:
        result = DeepFace.verify(image1, image2, enforce_detection=False)
        if (result['verified'] and result['distance'] > 0.6):
            return {
                    "match": str(False),
                    "distance": result['distance'],
                    "face_match": (1 - result['distance']) * 100,
                    "msg": "Face does not match"
                }

        return {
            "match": str(result['verified']),
            "distance": result['distance'],
            "face_match": (1 - result['distance']) * 100,
            "msg": "Face Matched" if result['verified'] else "Face does not match"
        }
    except Exception as e:
        print(f"DeepFace Error: {e}")
        return None


def find_face_encodings(image_bytes, model='large', grayscale=False):
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Convert the image to grayscale if the option is enabled
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Convert grayscale back to BGR by duplicating the grayscale channel
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Detect face locations
    face_locations = face_recognition.face_locations(image, model=model)
    if len(face_locations) == 0:
        print("No face found in the image.")
        return None

    # Get face encodings
    face_enc = face_recognition.face_encodings(image, face_locations)
    return face_enc if len(face_enc) > 0 else None


compare_faces_bp = Blueprint('compare_faces', __name__)

@compare_faces_bp.route('/compare_faces', methods=['POST'])
def compare_faces():
    print("Recognition Start")
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Missing required fields: image1 and image2'}), 400

    try:
        print("Images Found!")
        image1_bytes = request.files['image1'].read()
        image2_bytes = request.files['image2'].read()

        # Convert images to grayscale before processing
        face_enc1 = find_face_encodings(image1_bytes, model='large', grayscale=True)
        face_enc2 = find_face_encodings(image2_bytes, model='large', grayscale=True)

        print("Converted to Gray Scale!")
        # Fall back to DeepFace if no face is detected
        if face_enc1 is None or face_enc2 is None:
            print("Face Not Found!")
            fallback_result = fallback_to_deepface(image1_bytes, image2_bytes)
            if fallback_result:
                print("Fallback Result:", fallback_result)
                return jsonify({
                    "num_faces_detected": 2,
                    "comparison_results": [fallback_result]
                }), 200

            return jsonify({
                "num_faces_detected": 0,
                "comparison_results": [
                    {
                        "match": "False",
                        "distance": 0,
                        "face_match": 0,
                        "msg": "Face not found!"
                    }
                ]
            }), 400

        # Batch processing face comparisons
        face_enc1_np = np.array(face_enc1)
        face_enc2_np = np.array(face_enc2)

        distances = face_recognition.face_distance(face_enc1_np, face_enc2_np)
        matches = distances <= 0.6

        results = [
            {
                "match": str(matches[i]),
                "distance": distances[i],
                "face_match": (1 - distances[i]) * 100,
                "msg": "Face Matched" if matches[i] else "Face does not match"
            }
            for i in range(len(distances))
        ]

        print("Recognition Completed")
        return jsonify({
            "num_faces_detected": len(face_enc1),
            "comparison_results": results
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': "Error occurred during face comparison!"}), 500
