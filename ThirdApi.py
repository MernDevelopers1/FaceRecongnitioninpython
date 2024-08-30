import cv2
import face_recognition
from flask import request, jsonify, Blueprint
import numpy as np

def find_face_encodings(image_bytes, model='large', grayscale=False):
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    
    # Convert the image to grayscale if the option is enabled
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Convert grayscale back to BGR by duplicating the grayscale channel
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    face_enc = face_recognition.face_encodings(image, model=model)
    return face_enc if len(face_enc) > 0 else None

compare_faces2_bp = Blueprint('compare_faces2', __name__)

@compare_faces2_bp.route('/compare_faces2', methods=['POST'])
def compare_faces():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Missing required fields: image1 and image2'}), 400

    try:
        image1_bytes = request.files['image1'].read()
        image2_bytes = request.files['image2'].read()

        # Convert images to grayscale before processing
        face_enc1 = find_face_encodings(image1_bytes, model='large', grayscale=True)
        face_enc2 = find_face_encodings(image2_bytes, model='large', grayscale=True)

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
                match = face_recognition.compare_faces([face_encoding2], face_encoding1, tolerance=0.45)[0]
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
