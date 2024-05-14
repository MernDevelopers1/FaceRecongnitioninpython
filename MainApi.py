from flask import Flask

# Assuming FirstApi.py is in the same directory
from FirstApi import compare_faces_bp
from SecondApi import compare_faces1_bp
from ThirdApi import compare_faces2_bp

app = Flask(__name__)

app.register_blueprint(compare_faces_bp, url_prefix='/', methods=['POST'])
app.register_blueprint(compare_faces1_bp, url_prefix='/', methods=['POST'])
app.register_blueprint(compare_faces2_bp, url_prefix='/', methods=['POST'])

if __name__ == '__main__':
    app.run(debug=True)
