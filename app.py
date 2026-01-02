# app.py
from flask import Flask, request, jsonify
from model_pipeline import predict_from_raw

# Gemini integration
import openai

openai.api_key = "YOUR_GEMINI_API_KEY"  # <--- Keep this here

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    eeg = request.json["eeg"]
    result = predict_from_raw(eeg)   # model logic in separate file
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
