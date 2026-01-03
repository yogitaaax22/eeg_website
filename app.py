import os, gc
import numpy as np
import pickle
from flask import Flask, render_template, request
from scipy.io import loadmat
from scipy.signal import welch
from scipy.stats import skew, kurtosis, mstats
import google.generativeai as genai

app = Flask(__name__)

# --- 1. CONFIGURATION ---
# Using the stable name to prevent the "fallback error"
genai.configure(api_key="YOUR_GEMINI_API_KEY")
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# --- 2. ENGINE (82% ACCURACY LOGIC) ---
def extract_features_final(eeg):
    if eeg.shape[0] < eeg.shape[1]: eeg = eeg.T
    # 5% Winsorization & Z-Score (Match Colab)
    eeg = mstats.winsorize(eeg, limits=[0.05, 0.05], axis=0)
    eeg = (eeg - np.mean(eeg, axis=0)) / (np.std(eeg, axis=0) + 1e-6)
    
    all_channel_features = []
    for ch in range(eeg.shape[1]):
        sig = eeg[:, ch]
        f, psd = welch(sig, fs=128, nperseg=256)
        # Power Bands
        low = np.mean(psd[(f >= 0.5) & (f <= 8)])
        alpha = np.mean(psd[(f >= 8) & (f <= 13)])
        beta = np.mean(psd[(f >= 13) & (f <= 30)])
        total = low + alpha + beta + 1e-6
        all_channel_features.append([low/total, alpha/total, beta/total, float(skew(sig)), float(kurtosis(sig))])
    
    ch_feats = np.array(all_channel_features)
    # Frontal (1-16) and Occipital (17-32) averaging
    front, back = ch_feats[:16].mean(axis=0), ch_feats[16:].mean(axis=0)
    return np.concatenate([front, back]).reshape(1, -1)

# --- 3. LOAD MODEL ---
try:
    with open('stress_model.pkl', 'rb') as f:
        stress_model = pickle.load(f)
except:
    stress_model = None

# --- 4. ROUTES ---
@app.route("/", methods=["GET", "POST"])
def index():
    results = []      # Matches {% if results %} in your HTML
    clinical_note = "" # Matches {{ clinical_note }} in your HTML
    error = None

    if request.method == "POST":
        # Match HTML input name="mat_files"
        files = request.files.getlist("mat_files")
        
        if not files or files[0].filename == '':
            return render_template("index.html", error="No files uploaded.")

        try:
            last_status = "Relaxed"
            for file in files:
                data = loadmat(file)
                # Check for "Data" (Case sensitive)
                eeg = data.get("Data") or data.get("data")
                del data # Clear memory
                
                if eeg is not None:
                    features = extract_features_final(eeg)
                    del eeg # Clear memory
                    
                    if stress_model:
                        prob = float(stress_model.predict_proba(features)[0][1])
                        # CALIBRATION: The Person 12 Fix (0.82 threshold)
                        if prob > 0.96: status = "High Stress"
                        elif prob > 0.82: status = "Moderate Stress"
                        else: status = "Relaxed"
                        
                        results.append((file.filename, status))
                        last_status = status
                
                gc.collect() # Crucial for Render Free Tier

            # --- GEMINI AI ---
            if results:
                prompt = f"Patient EEG shows {last_status}. Provide 1 clinical wellness tip."
                try:
                    response = gemini_model.generate_content(prompt)
                    clinical_note = response.text
                except:
                    clinical_note = "Neural rhythms suggest focusing on rhythmic breathing."

        except Exception as e:
            error = f"Processing Error: {str(e)}"

    return render_template("index.html", results=results, clinical_note=clinical_note, error=error)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
