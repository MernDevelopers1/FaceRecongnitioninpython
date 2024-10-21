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
        
        result = DeepFace.verify(image1, image2, model_name="ArcFace", enforce_detection=False)
        distance = result['distance']
        face_match_percentage = (1 - distance) * 100
        match = distance < 0.6  
        results = {
            "match": str(match),  
            "distance": distance,
            "face_match": face_match_percentage,
            "msg": "Face Matched" if match else "Face does not match"
        }
        return results
    except Exception as e:
        print(f"DeepFace Error: {e}")
        return None

compare_faces5_bp = Blueprint('compare_faces5', __name__)

@compare_faces5_bp.route('/compare_faces5', methods=['POST'])
def compare_faces():
    print("Recognition Start")
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Missing required fields: image1 and image2'}), 400
    try:
        print("Images Found!")
        image1_bytes = request.files['image1'].read()
        image2_bytes = request.files['image2'].read()
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
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': "Error occurred during face comparison!"}), 500
