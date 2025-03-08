import os
## type down the mongo key url here 
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/deepfake_db")
SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key_here")
UPLOAD_FOLDER = "uploads/"
REPORT_FOLDER = "reports/"
