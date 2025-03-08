from flask import Flask
from flask_cors import CORS
from database.db import mongo
from routes.auth_routes import auth_bp
from routes.detection_routes import detection_bp
from routes.report_routes import report_bp

app = Flask(__name__)
CORS(app)

# Load Configurations
app.config.from_pyfile('config.py')

# Initialize MongoDB
mongo.init_app(app)

# Register Blueprints
app.register_blueprint(auth_bp, url_prefix="/auth")
app.register_blueprint(detection_bp, url_prefix="/detection")
app.register_blueprint(report_bp, url_prefix="/report")

if __name__ == '__main__':
    app.run(debug=True)
