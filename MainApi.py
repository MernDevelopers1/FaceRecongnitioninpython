from flask import Flask, jsonify

# Assuming FirstApi.py is in the same directory
# from FirstApi import compare_faces_bp
from FirstApi import compare_faces_bp
from SecondApi import compare_faces1_bp
from ThirdApi import compare_faces2_bp
from FourthApi import compare_faces3_bp
from FifthApi import compare_faces5_bp
from OpenFaceModel import compare_faces6_bp

app = Flask(__name__)

# app.register_blueprint(compare_faces_bp, url_prefix='/', methods=['POST'])
app.register_blueprint(compare_faces6_bp, url_prefix='/', methods=['POST'])
app.register_blueprint(compare_faces_bp, url_prefix='/', methods=['POST'])
app.register_blueprint(compare_faces1_bp, url_prefix='/', methods=['POST'])
app.register_blueprint(compare_faces2_bp, url_prefix='/', methods=['POST'])
app.register_blueprint(compare_faces3_bp, url_prefix='/', methods=['POST'])
app.register_blueprint(compare_faces5_bp, url_prefix='/', methods=['POST'])
@app.route('/hello', methods=['GET'])
def hello_world():
    return jsonify(message="Hello, World!")

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5000)
