import cv2
import face_recognition
from flask import request, jsonify, Blueprint
import numpy as np
from deepface import DeepFace

def find_face_encodings(image_bytes, model='large', grayscale=False):
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Convert the image to grayscale if the option is enabled
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Convert grayscale back to BGR by duplicating the grayscale channel
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    face_enc = face_recognition.face_encodings(image, model=model)
    return face_enc if len(face_enc) > 0 else None

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

compare_faces_bp = Blueprint('compare_faces', __name__)

@compare_faces_bp.route('/compare_faces', methods=['POST'])
def compare_faces():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Missing required fields: image1 and image2'}), 400

    try:
        image1_bytes = request.files['image1'].read()
        image2_bytes = request.files['image2'].read()

        # Convert images to grayscale before processing
        face_enc1 = find_face_encodings(image1_bytes, model='large', grayscale=True)
        face_enc2 = find_face_encodings(image2_bytes, model='large', grayscale=True)

        # Fall back to DeepFace if no face is detected
        if face_enc1 is None or face_enc2 is None:
            fallback_result = fallback_to_deepface(image1_bytes, image2_bytes)
            if fallback_result:
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

        results = []
        for face_encoding1 in face_enc1:
            for face_encoding2 in face_enc2:
                match = face_recognition.compare_faces([face_encoding2], face_encoding1, tolerance=0.6)[0]
                distance = face_recognition.face_distance([face_encoding2], face_encoding1)[0]

                result = {
                    "match": str(match),
                    "distance": distance,
                    "face_match": (1 - distance) * 100,
                    "msg": "Face Matched" if match else "Face does not match"
                }
                results.append(result)

        return jsonify({
            "num_faces_detected": len(face_enc1),
            "comparison_results": results
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': "Error occurred during face comparison!"}), 500
