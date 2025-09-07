import os
import shutil
import face_recognition

# Input and output folders
INPUT_FOLDER = "/mnt/d/TestImages"
OUTPUT_FOLDER = "/mnt/d/TestOutput"

# Create output folders
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
FACE_NOT_FOUND_FOLDER = os.path.join(OUTPUT_FOLDER, "face_not_found")
os.makedirs(FACE_NOT_FOUND_FOLDER, exist_ok=True)

# Parameters
TOLERANCE = 0.6  # smaller = stricter match

# Keep known encodings
known_faces = []  # list of (encoding, person_id)
person_count = 0

def get_face_encoding(img_path):
    """Return first face encoding in the image (if any)."""
    img = face_recognition.load_image_file(img_path)
    encodings = face_recognition.face_encodings(img)
    if len(encodings) > 0:
        return encodings[0]
    return None

# Loop through all images
for filename in os.listdir(INPUT_FOLDER):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(INPUT_FOLDER, filename)
    encoding = get_face_encoding(img_path)

    if encoding is None:
        # No face found → move to "face_not_found"
        dest_path = os.path.join(FACE_NOT_FOUND_FOLDER, filename)
        shutil.move(img_path, dest_path)  # use shutil.move if you want to move instead of copy
        print(f"{filename} → face_not_found")
        continue

    matched_person_id = None

    # Compare with known faces
    for known_encoding, person_id in known_faces:
        distance = face_recognition.face_distance([known_encoding], encoding)[0]
        if distance < TOLERANCE:
            matched_person_id = person_id
            break

    # If no match, create new person folder
    if matched_person_id is None:
        person_count += 1
        matched_person_id = f"person_{person_count}"
        known_faces.append((encoding, matched_person_id))

        # Create folder for new person
        os.makedirs(os.path.join(OUTPUT_FOLDER, matched_person_id), exist_ok=True)

    # Move image into the correct folder
    dest_path = os.path.join(OUTPUT_FOLDER, matched_person_id, filename)
    shutil.move(img_path, dest_path)  # use shutil.move if you want to move instead of copy

    print(f"{filename} → {matched_person_id}")
