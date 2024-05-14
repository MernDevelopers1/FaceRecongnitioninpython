import cv2
import face_recognition
import base64
from flask import request, jsonify, Blueprint
import numpy as np
def find_face_encodings(image_bytes):
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    face_enc = face_recognition.face_encodings(image)
    return face_enc if len(face_enc) > 0 else None

compare_faces_bp = Blueprint('compare_faces', __name__)  
@compare_faces_bp.route('/compare_faces', methods=['POST'])
def compare_faces():


    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Missing required fields: image1 and image2'}), 400

    try:
        image1_bytes = request.files['image1'].read()
        image2_bytes = request.files['image2'].read()
        face_enc1 = find_face_encodings(image1_bytes)
        face_enc2 = find_face_encodings(image2_bytes)
        if None in (face_enc1[0].any(), face_enc2[0].any()):
            return jsonify({'error': 'No faces detected in one or both images'}), 400
        results = []
        for face_encoding in face_enc1:
            match = face_recognition.compare_faces([ face_enc2[0]], face_encoding)[0]
            distance = face_recognition.face_distance([ face_enc2[0]], face_encoding)[0]
            results.append({
                "match": str(match),
                "distance": distance
            })
        return jsonify({
            "num_faces_detected": len(face_enc1),
            "comparison_results": results
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred during comparison'}), 500


