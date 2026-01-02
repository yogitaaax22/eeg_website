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

    # 1️⃣ Save uploaded file exactly as-is
    saved_path = "/tmp/uploaded_eeg.mat"
    file.save(saved_path)

    # 2️⃣ Compute hash for verification
    with open(saved_path, "rb") as f:
        file_bytes = f.read()
        file_hash = hashlib.md5(file_bytes).hexdigest()

    print("Uploaded file MD5 hash:", file_hash)

    # 3️⃣ Load EEG exactly like Colab
    mat_data = loadmat(saved_path)
    if 'Data' not in mat_data:
        return jsonify({"error": "Invalid .mat file"}), 400

    eeg = mat_data['Data']

    # 4️⃣ Preprocess and predict exactly like Colab
    features = preprocess_eeg(eeg)
    prediction = predict_from_raw(eeg)

    print("Prediction (0=Relax,1=Stress):", prediction)

    return jsonify({
        "prediction": int(prediction),
        "file_hash": file_hash
    })

if __name__ == "__main__":
    app.run(debug=True)

