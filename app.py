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

# --- 2. THE 82% ACCURACY ENGINE ---
def extract_features_final(eeg):
    try:
        if eeg.shape[0] < eeg.shape[1]: eeg = eeg.T
        eeg = mstats.winsorize(eeg, limits=[0.05, 0.05], axis=0)
        eeg = (eeg - np.mean(eeg, axis=0)) / (np.std(eeg, axis=0) + 1e-6)
        
        all_channel_features = []
        for ch in range(eeg.shape[1]):
            sig = eeg[:, ch]
            f, psd = welch(sig, fs=128, nperseg=256)
            low = np.mean(psd[(f >= 0.5) & (f <= 8)])
            alpha = np.mean(psd[(f >= 8) & (f <= 13)])
            beta = np.mean(psd[(f >= 13) & (f <= 30)])
            total = low + alpha + beta + 1e-6
            all_channel_features.append([low/total, alpha/total, beta/total, float(skew(sig)), float(kurtosis(sig))])
        
        ch_feats = np.array(all_channel_features)
        front, back = ch_feats[:16].mean(axis=0), ch_feats[16:].mean(axis=0)
        return np.concatenate([front, back]).reshape(1, -1)
    except Exception as e:
        print(f"Feature Error: {e}")
        return None

# --- 3. LOAD MODEL ---
try:
    with open('stress_model.pkl', 'rb') as f:
        stress_model = pickle.load(f)
except:
    stress_model = None

# --- 4. THE ROUTE ---
@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    clinical_note = ""
    error = None

    if request.method == "POST":
        files = request.files.getlist("mat_files")
        
        if not files or files[0].filename == '':
            return render_template("index.html", error="No file selected.")

        try:
            for file in files:
                data_dict = loadmat(file)
                # This line finds out what the file actually contains
                all_keys = [k for k in data_dict.keys() if not k.startswith('_')]
                
                # Try every common key name
                eeg = data_dict.get("Data") or data_dict.get("data") or data_dict.get("val") or data_dict.get("EEG")
                
                if eeg is not None:
                    model_in = extract_features_final(eeg)
                    if stress_model:
                        prob = float(stress_model.predict_proba(model_in)[0][1])
                        # Use your 82% Accuracy Calibration
                        status = "High Stress" if prob > 0.96 else "Moderate Stress" if prob > 0.82 else "Relaxed"
                        results.append((file.filename, status))
                    else:
                        error = "Model file (stress_model.pkl) is missing from the server."
                else:
                    # THIS IS THE FIX: It tells you what is actually inside your file
                    error = f"Could not find EEG data. Your file has these keys: {all_keys}. Update app.py to match."
                
                gc.collect()

            if results:
                # Gemini Tip
                try:
                    res = gemini_model.generate_content(f"User is {results[0][1]}. 1-sentence health tip.")
                    clinical_note = res.text
                except:
                    clinical_note = "Focus on deep breathing."

        except Exception as e:
            error = f"System Crash: {str(e)}"

    return render_template("index.html", results=results, clinical_note=clinical_note, error=error)
