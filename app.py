from flask import Flask, request, jsonify
import numpy as np
import os
from model_pipeline import predict_from_raw

# --- Gemini API setup (same place as before) ---
import openai  # or whatever library you were using for Gemini
openai.api_key = os.environ.get("OPENAI_API_KEY")
# keep any other Gemini setup here exactly as you had earlier

# --- Flask App ---
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    # Check file upload (matches index.html input name)
    if 'mat_files' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist('mat_files')
    results = []

    for file in files:
        try:
            # Save file temporarily (Render allows /tmp)
            temp_path = f"/tmp/{file.filename}"
            file.save(temp_path)

            # ⚠️ This assumes predict_from_raw handles loading .mat files internally
            prediction = predict_from_raw(temp_path)

            results.append({
                "file": file.filename,
                "prediction": prediction
            })

        except Exception as e:
            results.append({
                "file": file.filename,
                "error": str(e)
            })

    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(debug=True)
