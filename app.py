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
# Using Gemini 2.0 Flash Lite as requested
genai.configure(api_key="YOUR_GEMINI_API_KEY")
gemini_model = genai.GenerativeModel('gemini-2.0-flash-lite-preview-02-05')

# --- 2. PRE-PROCESSING ENGINE (82% Accuracy Logic) ---
def extract_features_final(eeg):
    # Orientation check
    if eeg.shape[0] < eeg.shape[1]: eeg = eeg.T
    
    # 5% Winsorization & Z-Score Normalization
    eeg = mstats.winsorize(eeg, limits=[0.05, 0.05], axis=0)
    eeg = (eeg - np.mean(eeg, axis=0)) / (np.std(eeg, axis=0) + 1e-6)
    
    all_channel_features = []
    for ch in range(eeg.shape[1]):
        sig = eeg[:, ch]
        # Welch PSD for Alpha, Beta, Low bands
        f, psd = welch(sig, fs=128, nperseg=256)
        low = np.mean(psd[(f >= 0.5) & (f <= 8)])
        alpha = np.mean(psd[(f >= 8) & (f <= 13)])
        beta = np.mean(psd[(f >= 13) & (f <= 30)])
        total = low + alpha + beta + 1e-6
        all_channel_features.append([low/total, alpha/total, beta/total, float(skew(sig)), float(kurtosis(sig))])
    
    ch_feats = np.array(all_channel_features)
    # Mean of first 16 (Frontal) and last 16 (Occipital)
    front, back = ch_feats[:16].mean(axis=0), ch_feats[16:].mean(axis=0)
    return np.concatenate([front, back]).reshape(1, -1)

# --- 3. LOAD YOUR PICKLE MODEL ---
try:
    with open('stress_model.pkl', 'rb') as f:
        stress_model = pickle.load(f)
except Exception as e:
    stress_model = None
    print(f"Model Load Error: {e}")

# --- 4. FLASK ROUTES ---
@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    clinical_note = ""
    error = None

    if request.method == "POST":
        uploaded_files = request.files.getlist("mat_files")
        
        if not uploaded_files or uploaded_files[0].filename == '':
            return render_template("index.html", error="No file selected.")

        try:
            last_status = "Relaxed"
            for file in uploaded_files:
                # Direct load from file stream to save Render RAM
                data_dict = loadmat(file)
                eeg = data_dict.get("Data") or data_dict.get("data")
                del data_dict # Immediate cleanup
                
                if eeg is not None:
                    model_in = extract_features_final(eeg)
                    del eeg # Immediate cleanup
                    
                    if stress_model:
                        # Probability from your XGBoost model
                        prob = float(stress_model.predict_proba(model_in)[0][1])
                        
                        # Thresholds for Person 12 (0.82 calibration)
                        if prob > 0.96: status = "High Stress"
                        elif prob > 0.82: status = "Moderate Stress"
                        else: status = "Relaxed"
                        
                        results.append((file.filename, status))
                        last_status = status
                
                gc.collect() # Garbage collection for Render stability

            # --- GEMINI 2.0 FLASH LITE INSIGHT ---
            if results:
                prompt = f"Patient EEG results show {last_status}. Provide a 1-sentence medical wellness tip."
                try:
                    response = gemini_model.generate_content(prompt)
                    clinical_note = response.text
                except:
                    clinical_note = "Focus on deep, rhythmic breathing to lower neural stress markers."

        except Exception as e:
            error = f"Processing Error: {str(e)}"

    return render_template("index.html", results=results, clinical_note=clinical_note, error=error)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
