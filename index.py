import cv2
import dlib
import numpy as np

# Load the pre-trained face detector with improved accuracy
detector = dlib.get_frontal_face_detector()

# Load the pre-trained facial landmark predictor
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load the pre-trained face recognition model
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
# Preprocessing function for images
def preprocess_image(image):
    if image is not None:
    # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply histogram equalization
        equalized = cv2.equalizeHist(gray)
    # Apply Gaussian blur for noise reduction
        blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
        return blurred
    else:
        print("Error: Input image is None.")

# Function to extract facial features from an image
def extract_features(image,shape):
   
    try:
        face_descriptor = facerec.compute_face_descriptor(image, shape)
        return np.array(face_descriptor)
    except Exception as e:
        print("An error occurred during face descriptor computation:", e)
        return None

def detect_Face(image):
    dets = detector(image, 1)
    if len(dets) == 0:
        return None
    
    shape = predictor(image, dets[0])
    return shape
# Function to compare two images and determine if they are of the same person
def compare_images(image1_path, image2_path, threshold=0.6):
    # Load the images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # Preprocess the images (optional)
    # image1 = preprocess_image(image1)
    # image2 = preprocess_image(image2)
    
    shape1 = detect_Face(image1)
    shape2 = detect_Face(image2)
    print(shape1)
    print(shape2)
    # Extract facial features from both images
    features1 = extract_features(image1, shape1)
    features2 = extract_features(image2, shape2)

    # Check if facial features were successfully extracted
    if features1 is None or features2 is None:
        return False

    # Calculate the Euclidean distance between the facial feature vectors
    distance = np.linalg.norm(features1 - features2)

    # Compare the distance with a threshold to determine if the images are of the same person
    if distance < threshold:
        return True
    else:
        return False


# Paths to the two images to be compared
image1_path = 'public/croped (1).JPG'
image2_path = 'public/croped (2).JPG'

# Compare the images
result = compare_images(image1_path, image2_path)

# Print the result
if result:
    print("The images are of the same person.")
else:
    print("The images are of different persons.")

