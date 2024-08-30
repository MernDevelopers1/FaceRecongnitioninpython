import cv2
import face_recognition
from flask import request, jsonify, Blueprint
import numpy as np

def find_face_encodings(image_bytes, model='large'):
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    # Optionally preprocess image, e.g., grayscale or normalization can be done here
    face_enc = face_recognition.face_encodings(image, model=model)
    return face_enc if len(face_enc) > 0 else None

compare_faces1_bp = Blueprint('compare_faces1', __name__)

@compare_faces1_bp.route('/compare_faces1', methods=['POST'])
def compare_faces():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Missing required fields: image1 and image2'}), 400

    try:
        image1_bytes = request.files['image1'].read()
        image2_bytes = request.files['image2'].read()

        # Using the larger model for better accuracy
        face_enc1 = find_face_encodings(image1_bytes, model='large')
        face_enc2 = find_face_encodings(image2_bytes, model='large')

        if face_enc1 is None or face_enc2 is None:
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
                match = face_recognition.compare_faces([face_encoding2], face_encoding1, tolerance=0.50)[0]
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
