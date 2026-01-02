# app.py
from flask import Flask, request, jsonify
from model_pipeline import predict_from_raw
import hashlib
from scipy.io import loadmat

# Gemini integration
import openai

openai.api_key = "YOUR_GEMINI_API_KEY"  # <--- Keep this here

app = Flask(__name__)

@app.route("/predict_verified", methods=["POST"])
def predict_verified():
    if 'eeg_file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['eeg_file']

    # Save uploaded file
    saved_path = "/tmp/uploaded_eeg.mat"
    file.save(saved_path)

    # Compute MD5 hash
    import hashlib
    with open(saved_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()

    # Load EEG like Colab
    from scipy.io import loadmat
    mat_data = loadmat(saved_path)
    eeg = mat_data['Data']

    # Predict
    from model_pipeline import predict_from_raw
    prediction = predict_from_raw(eeg)

    # Return both prediction and hash
    return jsonify({
        "prediction": int(prediction),
        "file_hash": file_hash
    })
