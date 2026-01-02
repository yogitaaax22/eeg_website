import os
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg') # Necessary for cloud servers without a monitor
import matplotlib.pyplot as plt

from flask import Flask, render_template, request
from scipy.io import loadmat
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from scipy.stats.mstats import winsorize
from collections import Counter
import google.generativeai as genai

app = Flask(__name__)

# --- 1. LOAD YOUR 82% FILES ---
# This looks for the model in the same folder as app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'stress_model.pkl')

stress_model = None

try:
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            stress_model = pickle.load(f)
        print("✅ SUCCESS: 82% Stress Model Loaded.")
    else:
        print(f"❌ ERROR: Model file not found at {model_path}")
except Exception as e:
    print(f"❌ ERROR LOADING MODEL: {e}")

# Gemini API Configuration
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

# --- 2. THE 82% FEATURE LOGIC ---
def extract_features(eeg):
    # Ensure data is oriented correctly
    if eeg.shape[0] < eeg.shape[1]: eeg = eeg.T
    
    # Preprocessing: Winsorize & Normalize
    eeg = winsorize(eeg, limits=[0.05, 0.05], axis=0)
    eeg = (eeg - np.mean(eeg, axis=0)) / (np.std(eeg, axis=0) + 1e-6)

    def get_bands(sig):
        freqs, psd = welch(sig, fs=128, nperseg=256)
        low = np.mean(psd[(freqs >= 0.5) & (freqs <= 8)])
        alpha = np.mean(psd[(freqs >= 8) & (freqs <= 13)])
        beta = np.mean(psd[(freqs >= 13) & (freqs <= 30)])
        total = low + alpha + beta + 1e-6
        return [low/total, alpha/total, beta/total, skew(sig), kurtosis(sig)]

    all_feats = np.array([get_bands(eeg[:, i]) for i in range(eeg.shape[1])])
    
    # 82% Strategy: Spatial Averaging (Front 16 vs Back 16)
    front = all_feats[:16].mean(axis=0)
    back = all_feats[16:].mean(axis=0)
    return np.concatenate([front, back]).reshape(1, -1)

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    clinical_note = ""
    
    if request.method == "POST":
        uploaded_files = request.files.getlist("mat_files")
        if uploaded_files:
            # Create static folder for plots if it doesn't exist
            static_dir = os.path.join(BASE_DIR, "static")
            if not os.path.exists(static_dir): 
                os.makedirs(static_dir)
            
            predicted_levels, filenames = [], []

            for file in uploaded_files:
                file_path = os.path.join(static_dir, file.filename)
                file.save(file_path)
                filenames.append(file.filename)
                
                try:
                    data = loadmat(file_path)
                    eeg = data.get("Data")
                    
                    if stress_model is None:
                        raise ValueError("Model not loaded.")

                    # Predict
                    feats = extract_features(eeg)
                    prob = stress_model.predict_proba(feats)[:, 1]
                    state = "Stress" if prob > 0.60 else "Relax"
                    predicted_levels.append(state)
                    
                except Exception as e:
                    print(f"Error processing {file.filename}: {e}")

            if predicted_levels:
                counts = Counter(predicted_levels)
                main_state = counts.most_common(1)[0][0]

                # --- GEMINI AI REPORT ---
                try:
                    model_gen = genai.GenerativeModel('gemini-2.0-flash-lite')
                    prompt = f"EEG Results: {dict(counts)}. Majority state: {main_state}. Provide a short report."
                    response = model_gen.generate_content(prompt)
                    clinical_note = response.text.strip().replace("\n\n", "<br><br>")
                except:
                    clinical_note = f"Analysis complete. Most trials showed: {main_state}."

                # --- GENERATE PLOT ---
                plt.figure(figsize=(6, 4))
                plt.bar(counts.keys(), counts.values(), color=['#2e7d32', '#ef5350'])
                plt.title("Session Stress Analysis")
                plt.ylabel("Trial Count")
                plot_path = os.path.join(static_dir, "plot.png")
                plt.savefig(plot_path)
                plt.close()

                results = list(zip(filenames, predicted_levels))
            
    return render_template("index.html", results=results, clinical_note=clinical_note)

if __name__ == "__main__":
    # Local run (for cloud, gunicorn will use app:app)
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))
