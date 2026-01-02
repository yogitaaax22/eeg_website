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
    # Get uploaded files from HTML input (matches your index.html)
    if 'mat_files' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist('mat_files')  # supports multiple uploads
    results = []

    for file in files:
        try:
            # Save temporarily
            saved_path = f"/tmp/{file.filename}"
            file.save(saved_path)

            # Load EEG from .npz (matches Colab format)
            eeg_data = np.load(saved_path)['data']

            # Predict using model pipeline
            prediction = predict_from_raw(eeg_data)
            results.append((file.filename, int(prediction)))

        except Exception as e:
            results.append((file.filename, f"Error: {str(e)}"))

    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(debug=True)
