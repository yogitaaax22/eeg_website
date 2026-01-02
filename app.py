# app.py
from flask import Flask, request, jsonify
import numpy as np
import os
from model_pipeline import predict_from_raw

# Gemini integration
import openai

openai.api_key = "YOUR_GEMINI_API_KEY"  # <--- Keep this here

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if 'eeg_file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['eeg_file']

    # Save temporarily
    saved_path = "/tmp/uploaded_eeg.npz"
    file.save(saved_path)

    # Load EEG from .npz (stable format)
    try:
        eeg_data = np.load(saved_path)['data']
    except Exception as e:
        return jsonify({"error": f"Failed to load EEG: {str(e)}"}), 400

    # Predict using model pipeline
    try:
        prediction = predict_from_raw(eeg_data)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
