import cv2
from flask import request, jsonify, Blueprint
import numpy as np
from deepface import DeepFace

def find_face_encodings(image_bytes, grayscale=False):
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Convert the image to grayscale if the option is enabled
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Convert grayscale back to BGR by duplicating the grayscale channel
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image

compare_faces3_bp = Blueprint('compare_faces3', __name__)

@compare_faces3_bp.route('/compare_faces3', methods=['POST'])
def compare_faces():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Missing required fields: image1 and image2'}), 400

    try:
        image1_bytes = request.files['image1'].read()
        image2_bytes = request.files['image2'].read()

        # Convert images to grayscale before processing if needed
        image1 = find_face_encodings(image1_bytes, grayscale=True)
        image2 = find_face_encodings(image2_bytes, grayscale=True)

        # Use DeepFace to verify faces
        try:
            result = DeepFace.verify(img1_path=image1, img2_path=image2, enforce_detection=False)
            distance = result['distance']
            match = result['verified']

            response = {
                "match": str(match),
                "distance": distance,
                "face_match": (1 - distance) * 100 if match else 0,
                "msg": "Face Matched" if match else "Face does not match"
            }

            return jsonify({
                "num_faces_detected": 2,
                "comparison_results": [response]
            })

        except ValueError as ve:
            # Handle cases where face is not detected in one or both images
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

# Register the blueprint with the Flask app (not included in the snippet)
