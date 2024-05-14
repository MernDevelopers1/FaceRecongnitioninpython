import cv2
from flask import request, jsonify, Blueprint
import numpy as np
from deepface import DeepFace

def find_face_encodings(image_bytes):
    
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    return image

compare_faces1_bp = Blueprint('compare_faces1', __name__)  # Replace 'compare_faces' with your desired blueprint name

# Define your routes (functions) within the blueprint object
@compare_faces1_bp.route('/compare_faces1', methods=['POST'])
def compare_faces():
    
    print("api CAlled!! 1")

    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Missing required fields: image1 and image2'}), 400

    try:
        image1_bytes = request.files['image1'].read()
        image2_bytes = request.files['image2'].read()

    
        image1 = find_face_encodings(image1_bytes)
        image2 = find_face_encodings(image2_bytes)


        if None in (image1.any(), image2.any()):
            return jsonify({'error': 'No faces detected in one or both images'}), 400

        result = DeepFace.verify(image1, image2)
        return jsonify({"issame face": result['verified']})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred during comparison'}), 500
