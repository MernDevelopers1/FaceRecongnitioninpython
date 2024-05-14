import dlib
import numpy as np

# Load the pre-trained face detector
detector = dlib.get_frontal_face_detector()

# Load the pre-trained facial landmark predictor
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load the pre-trained face recognition model
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# Function to extract features from an image
def extract_features(image):
    dets = detector(image, 1)
    shape = predictor(image, dets[0])
    face_descriptor = facerec.compute_face_descriptor(image, shape)
    return np.array(face_descriptor)

# Example usage:
image = cv2.imread('face_image.jpg')
features = extract_features(image)
