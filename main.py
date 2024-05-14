import cv2
from deepface import DeepFace
import os
# import matplotlib.pyplot as plt

# DeepFace.build_model("Facenet")
# DeepFace.build_model("OpenFace")
# DeepFace.build_model("DeepFace")
# DeepFace.build_model("FbDeepFace")

def preprocess_image(img_path):
    # Resize image to a smaller dimension (e.g., 160x160)
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
    else:
        print(f"Error: Image {img_path} not found!")
        return None  # Or handle the error differently

    # Convert image to grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize the grayscale image
    # resized = cv2.resize(img, (650, 650))

    return img
img1 = 'public/DJI_0617.JPG'

img2 = 'public/DJI_0611.JPG'

img1_processed = preprocess_image(img1)
img2_processed = preprocess_image(img2)
if img1_processed is not None and img2_processed is not None:
    result = DeepFace.verify(img1_processed, img2_processed)
    print("is same face:", result['verified'])
else:
    print("Error: Face could not be detected in one of the images.")


