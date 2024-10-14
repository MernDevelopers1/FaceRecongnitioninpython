import cv2
import numpy as np
import insightface

# Initialize the InsightFace model
app = insightface.app.FaceAnalysis()
app.prepare(ctx_id=-1)  # Set to 0 if you want to use GPU

def get_face_embedding(image_path):
    # Load an image and detect faces
    img = cv2.imread(image_path)
    faces = app.get(img)

    if not faces:
        print("No face detected in the image.")
        return None
    
    # Get the normed embedding for the first detected face
    return faces[0].normed_embedding

def compare_faces(image_path1, image_path2, threshold=0.4):
    # Get embeddings for both images
    embedding1 = get_face_embedding(image_path1)
    embedding2 = get_face_embedding(image_path2)

    if embedding1 is None or embedding2 is None:
        return False

    # Calculate the Euclidean distance between the two embeddings
    distance = np.linalg.norm(embedding1 - embedding2)
    print(f"Distance between faces: {distance}")

    # Compare distance with threshold
    return distance < threshold

# Example usage
image_path1 = "./images/1.jpg"
image_path2 = "./images/1.png"
are_faces_same = compare_faces(image_path1, image_path2)

if are_faces_same:
    print("The faces are similar.")
else:
    print("The faces are not similar.")
