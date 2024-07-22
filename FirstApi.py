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
    print("Called!!")


    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Missing required fields: image1 and image2'}), 400

    try:
        image1_bytes = request.files['image1'].read()
        image2_bytes = request.files['image2'].read()
        face_enc1 = find_face_encodings(image1_bytes)
        face_enc2 = find_face_encodings(image2_bytes)
        # print(face_enc1,face_enc2)
        print("not error occure")
        if  face_enc1 is None or face_enc2 is None:
            print("No FaceDetect")
            return jsonify({
            "num_faces_detected": 0,
            "comparison_results": [
                {
                    "match": "False",
                    "distance": 0,
                    "face_match": 0,
                    "msg":"Face not found!"


                }

            ]
        }),400
            
        print('Face Detected')
        results = []
        for face_encoding in face_enc1:
            match = face_recognition.compare_faces([ face_enc2[0]], face_encoding)[0]
            distance = face_recognition.face_distance([ face_enc2[0]], face_encoding)[0]
            if match and distance <= 0.5:
                results.append({
                    "match": str(match),
                    "distance": distance,
                    "face_match": ( 1 - distance )*100,
                    "msg":"Face Matched"
                })
            else: 
                results.append({
                    "match": "False",
                    "distance": distance,
                    "face_match": ( 1 - distance )*100,
                    "msg":"Face does not match"
                })
            print(results)
        return jsonify({
            "num_faces_detected": len(face_enc1),
            "comparison_results": results
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': "Error Occure during faceComparison!"}), 500


