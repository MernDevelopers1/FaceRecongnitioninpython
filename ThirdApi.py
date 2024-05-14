import cv2
import face_recognition
from flask import request, jsonify, Blueprint, make_response
import numpy as np

def find_face_encodings(image_bytes):
    try:
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"Error decoding image bytes: {e}")
        return None
compare_faces2_bp = Blueprint('compare_faces2', __name__) 
@compare_faces2_bp.route('/compare_faces2', methods=['POST'])
def compare_faces_with_multiple():
    try:

        image1_bytes = request.files['image1'].read()
        image2_bytes = request.files['image2'].read()
        
        image1 = find_face_encodings(image1_bytes)
        image2 = find_face_encodings(image2_bytes)
        print("encoding compeleted")
        if image1 is None or image2 is None:
            raise Exception("Failed to decode one or both images")
        face_locations = face_recognition.face_locations(image1)
        face_encodings = []
        print("Face Locations", face_locations)
        if not face_locations:
            return jsonify({
                "num_faces_detected": 0,
                "comparison_results": [],
                "error": "No faces detected in image1"
            })
        for face_location in face_locations:
            print("done 1",face_location)
            top, right, bottom, left = face_location
            print("done 2",top,right,bottom,left)
            face_image = image1[top:bottom, left:right]
            print("done 3", face_image)
            face_encoding = face_recognition.face_encodings(face_image)[0]
            print("done 4")
            face_encodings.append(face_encoding)
            print("done 5")
        known_encoding = face_recognition.face_encodings(image2)[0]
        results = []
        for face_encoding in face_encodings:
            match = face_recognition.compare_faces([known_encoding], face_encoding)[0]
            distance = face_recognition.face_distance([known_encoding], face_encoding)[0]
            results.append({
                "match": match,
                "distance": distance
            })
        return jsonify({
            "num_faces_detected": len(face_locations),
            "comparison_results": results
        })
    except Exception as e:
        print(f"Error during face comparison: {e}")
        error_message = {"error": str(e)}
        return make_response(jsonify(error_message), 400) 

