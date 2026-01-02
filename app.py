# app.py
from flask import Flask, request, jsonify
from model_pipeline import predict_from_raw

# Gemini integration
import openai

openai.api_key = "YOUR_GEMINI_API_KEY"  # <--- Keep this here

app = Flask(__name__)

@app.route("/predict_verified", methods=["POST"])
def predict_verified():
    if 'eeg_file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['eeg_file']

    # Save uploaded file temporarily
    saved_path = "/tmp/uploaded_eeg.mat"
    file.save(saved_path)

    # Compute MD5 hash to check
    with open(saved_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    print("Uploaded file hash:", file_hash)

    # Load EEG like Colab
    mat_data = loadmat(saved_path)
    eeg = mat_data['Data']

    prediction = predict_from_raw(eeg)

    return jsonify({
        "prediction": int(prediction),
        "file_hash": file_hash
    })
