import os, gc
import numpy as np
import pickle
from flask import Flask, render_template, request
from scipy.io import loadmat
from scipy.signal import welch
from scipy.stats import skew, kurtosis, mstats
import google.generativeai as genai

app = Flask(__name__)

# --- 1. CONFIG ---
genai.configure(api_key="YOUR_GEMINI_API_KEY")
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# --- 2. THE ENGINE ---
def extract_features_final(eeg):
    if eeg.shape[0] < eeg.shape[1]: eeg = eeg.T
    eeg = mstats.winsorize(eeg, limits=[0.05, 0.05], axis=0)
    eeg = (eeg - np.mean(eeg, axis=0)) / (np.std(eeg, axis=0) + 1e-6)
    
    all_channel_features = []
    for ch in range(eeg.shape[1]):
        sig = eeg[:, ch]
        # Welch Power Spectral Density
        f, psd = welch(sig, fs=128, nperseg=256)
        low = np.mean(psd[(f >= 0.5) & (f <= 8)])
        alpha = np.mean(psd[(f >= 8) & (f <= 13)])
        beta = np.mean(psd[(f >= 13) & (f <= 30)])
        total = low + alpha + beta + 1e-6
        all_channel_features.append([low/total, alpha/total, beta/total, float(skew(sig)), float(kurtosis(sig))])
    
    ch_feats = np.array(all_channel_features)
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
    results = []
    ai_note = ""
    
    if request.method == "POST":
        # MATCHING YOUR HTML NAME: "mat_files"
        uploaded_files = request.files.getlist("mat_files")
        
        if not uploaded_files or uploaded_files[0].filename == '':
            return render_template("index.html")

        last_label = "Relaxed"
        for file in uploaded_files:
            try:
                data = loadmat(file)
                eeg = data.get("Data") or data.get("data")
                del data # Save memory
                
                if eeg is not None:
                    model_in = extract_features_final(eeg)
                    del eeg
                    
                    if stress_model:
                        prob = float(stress_model.predict_proba(model_in)[0][1])
                        # Person 12 Calibration
                        if prob > 0.96: label = "High Stress"
                        elif prob > 0.82: label = "Moderate Stress"
                        else: label = "Relaxed"
                        
                        results.append((file.filename, label))
                        last_label = label
                
                gc.collect() # Force cleanup
            except:
                continue

        if results:
            try:
                prompt = f"The EEG analysis shows {last_label}. Provide one clinical wellness tip."
                ai_note = gemini_model.generate_content(prompt).text
            except:
                ai_note = "Maintain deep diaphragmatic breathing to stabilize neural rhythms."

    return render_template("index.html", results=results, clinical_note=ai_note)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
