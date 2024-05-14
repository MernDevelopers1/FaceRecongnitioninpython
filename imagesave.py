import os
import base64
from flask import Flask, request, jsonify

app = Flask(__name__)

# Configure upload directory (adjust the path as needed)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/save_image', methods=['POST'])
def save_image():
    """
    API endpoint to receive a Base64-encoded image and save it to a file.

    Returns:
        JSON: A response containing a success message or error details.
    """

    try:
        # Check for required field
        if 'image' not in request.files:
            return jsonify({'error': 'Missing required field: image'}), 400

        # Extract Base64-encoded image data
        image_data = request.files['image'].read()

        # Decode and validate Base64 data (optional)
        try:
            decoded_data = base64.b64decode(image_data)
        except Exception as e:
            return jsonify({'error': f'Invalid Base64 data: {e}'}), 400

        # Generate a unique filename
        filename = f'{os.urandom(16).hex()}.jpg'  # Use .jpg for demonstration, adjust for actual format

        # Create upload directory if it doesn't exist
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        # Save image to file
        with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'wb') as f:
            f.write(image_data)

        return jsonify({'message': f'Image saved successfully with filename: {filename}'})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred while saving the image'}), 500

if __name__ == '__main__':
    app.run(debug=True)
