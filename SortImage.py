import os
import shutil
import face_recognition
from PIL import UnidentifiedImageError

# Input + output folders
INPUT_FOLDER = "/mnt/d/TestImages"
OUTPUT_FOLDER = "/mnt/d/TestOutput"
NOT_FOUND_FOLDER = os.path.join(OUTPUT_FOLDER, "face_not_found")

# Create folders if not exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(NOT_FOUND_FOLDER, exist_ok=True)

# Store encodings + person folder mapping
known_faces = []
known_folders = []


def get_face_encoding(img_path):
    """Return first face encoding in the image (if any), with fallback detection."""
    try:
        img = face_recognition.load_image_file(img_path)

        # Try CNN first (more accurate)
        locations = face_recognition.face_locations(img, model="cnn")

        if len(locations) == 0:
            # Fallback to HOG
            locations = face_recognition.face_locations(img, model="hog")

        if len(locations) > 0:
            encodings = face_recognition.face_encodings(img, known_face_locations=locations)
            if len(encodings) > 0:
                return encodings[0]

    except UnidentifiedImageError:
        print(f"[ERROR] Cannot identify file as image → {img_path}")
    except Exception as e:
        print(f"[ERROR] Failed to process {img_path} → {e}")

    return None


def sort_images():
    person_count = 0

    for filename in os.listdir(INPUT_FOLDER):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(INPUT_FOLDER, filename)
        encoding = get_face_encoding(img_path)

        if encoding is None:
            # No face or unreadable → move to face_not_found
            shutil.move(img_path, os.path.join(NOT_FOUND_FOLDER, filename))
            print(f"[NO FACE] → {filename}")
            continue

        # Compare with known faces
        matches = face_recognition.compare_faces(known_faces, encoding, tolerance=0.5)

        if True in matches:
            match_index = matches.index(True)
            person_folder = known_folders[match_index]
        else:
            # New person → create folder
            person_count += 1
            person_folder = os.path.join(OUTPUT_FOLDER, f"person_{person_count}")
            os.makedirs(person_folder, exist_ok=True)
            known_faces.append(encoding)
            known_folders.append(person_folder)

        # Move file to that person's folder
        shutil.move(img_path, os.path.join(person_folder, filename))
        print(f"[SORTED] {filename} → {person_folder}")


if __name__ == "__main__":
    sort_images()
