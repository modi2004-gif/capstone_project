import os
import sys
import uuid
import traceback
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# ============================================================
# SET CORRECT PATH TO ml_model/predict
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # /backend
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))  # project root
PREDICT_DIR = os.path.join(PROJECT_ROOT, "ml_model", "predict")

print("\n======================================")
print(" EXPECTED PREDICT FOLDER:")
print(" ", PREDICT_DIR)
print("======================================\n")

# Remove backend folder from sys.path to avoid wrong imports
if BASE_DIR in sys.path:
    sys.path.remove(BASE_DIR)

# Add ml_model/predict FIRST so Python always loads correct module
if PREDICT_DIR not in sys.path:
    sys.path.insert(0, PREDICT_DIR)

# Import predict_helper from correct folder
try:
    import predict_helper
    print("predict_helper loaded from:", predict_helper.__file__)
    from predict_helper import predict_from_path
except Exception as e:
    print("❌ IMPORT ERROR:", e)
    raise SystemExit(1)

# ============================================================
# UPLOAD SETTINGS
# ============================================================

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXT = {"png", "jpg", "jpeg"}

app = Flask(__name__)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


# ============================================================
# /predict ENDPOINT
# ============================================================

@app.route("/predict", methods=["POST"])
def predict_endpoint():

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "Filename is empty"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Only JPG, JPEG, PNG files allowed"}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        file_path = os.path.join(UPLOAD_DIR, unique_name)
        file.save(file_path)

        # Run prediction
        results = predict_from_path(file_path)

        return jsonify({
            "success": True,
            "predictions": results
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    print("🚀 AI Crop Disease Prediction Backend Started")
    print("➡ Endpoint: http://127.0.0.1:5000/predict")
    app.run(host="0.0.0.0", port=5000, debug=True)
