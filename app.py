# app.py
from flask import Flask, request, jsonify
from model_pipeline import predict_from_raw

# Gemini integration
import openai

openai.api_key = "YOUR_GEMINI_API_KEY"  # <--- Keep this here

app = Flask(__name__)

@app.route("/predict_mat", methods=["POST"])
def predict_mat():
    # 1️⃣ Load the uploaded .mat file
    file = request.files['eeg_file']  # This must match your form input name
    mat_data = loadmat(file)
    eeg = mat_data['Data']

    # 2️⃣ Debug prints (temporary)
    print("Raw EEG shape:", eeg.shape)
    print("Raw EEG dtype:", eeg.dtype)
    print("Raw EEG min/max:", eeg.min(), eeg.max())

    # 3️⃣ Preprocess and extract features exactly as in Colab
    features = preprocess_eeg(eeg)
    print("Features shape:", features.shape)
    print("Features min/max:", features.min(), features.max())

    # 4️⃣ Predict using the exact same model
    pred = predict_from_raw(eeg)
    print("Prediction:", pred)

    return jsonify({"prediction": int(pred)})
