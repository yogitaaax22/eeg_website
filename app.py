import os
import numpy as np
import pickle
from flask import Flask, render_template, request
from scipy.io import loadmat
from scipy.signal import welch
from scipy.stats import skew, kurtosis, mstats
import google.generativeai as genai

app = Flask(__name__)

# --- 1. GEMINI CONFIGURATION (UPDATED TO 2.5 FLASH-LITE) ---
# Replace with your actual API Key
genai.configure(api_key="YOUR_GEMINI_API_KEY")
# Updated model name to gemini-2.5-flash-lite
gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite')

# --- 2. THE PROCESSING ENGINE (82% ACCURACY LOGIC) ---
def extract_band_power(signal, fs=128):
    freqs, psd = welch(signal, fs=fs, nperseg=fs*2)
    bands = {'low': (0.5, 8), 'alpha': (8, 13), 'beta': (13, 30)}
    powers = []
    for band in bands.values():
        idx = (freqs >= band[0]) & (freqs <= band[1])
        powers.append(np.mean(psd[idx]))
    total = sum(powers) + 1e-6
    return [p / total for p in powers]

def extract_features_final(eeg):
    if eeg.shape[0] < eeg.shape[1]:
        eeg = eeg.T
    
    # 5% Winsorization for artifact removal
    eeg = mstats.winsorize(eeg, limits=[0.05, 0.05], axis=0)
    
    # Z-Score Normalization
    eeg = (eeg - np.mean(eeg, axis=0)) / (np.std(eeg, axis=0) + 1e-6)

    all_channel_features = []
    for ch in range(eeg.shape[1]):
        sig = eeg[:, ch]
        rel_p = extract_band_power(sig)
        all_channel_features.append(rel_p + [float(skew(sig)), float(kurtosis(sig))])

    ch_feats = np.array(all_channel_features)
    
    # Regional Mean (Frontal vs Occipital)
    front = ch_feats[:16].mean(axis=0)
    back = ch_feats[16:].mean(axis=0)
    
    # 10-feature vector for XGBoost
    model_in = np.concatenate([front, back]).reshape(1, -1)
    
    # UI Stats (Alpha, Beta, Gamma/Low bands)
    visual_out = [float(front[0]), float(front[1]), float(front[2])] 
    return model_in, visual_out

# --- 3. LOADING THE MODEL ---
try:
    with open('stress_model.pkl', 'rb') as f:
        stress_model = pickle.load(f)
except Exception as e:
    stress_model = None
    print(f"Model Load Error: {e}")

# --- 4. CALIBRATED THRESHOLDS (PERSON 12 FIX) ---
def get_label(prob):
    # Adjusted threshold to 0.82 to handle high neural baselines
    if prob > 0.96:    
        return "🔴 High Stress"
    elif prob > 0.82:  
        return "🟠 Moderate Stress"
    else:              
        return "🟢 Relaxed"

# --- 5. UPDATED FLASK ROUTE (MULTIPLE FILE SUPPORT) ---
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # MATCHED TO HTML: Changed 'file' to 'mat_files'
        uploaded_files = request.files.getlist("mat_files")
        
        if not uploaded_files or uploaded_files[0].filename == '':
            return render_template("index.html", error="Please upload at least one .mat file")

        results = []
        last_status = "Unknown"
        
        for file in uploaded_files:
            file_path = "temp.mat"
            file.save(file_path)
            
            try:
                data_dict = loadmat(file_path)
                eeg = data_dict.get("Data") # Ensure your .mat file has this key
                
                if eeg is not None:
                    model_in, vis_stats = extract_features_final(eeg)
                    
                    if stress_model:
                        prob = float(stress_model.predict_proba(model_in)[0][1])
                        last_status = get_label(prob)
                        results.append((file.filename, last_status))
                    else:
                        return render_template("index.html", error="Model not loaded on server.")
                else:
                    results.append((file.filename, "Error: No 'Data' key found"))
            except Exception as e:
                results.append((file.filename, f"Error: {str(e)}"))

        # Gemini AI Analysis (Updated to 2.5 Flash-Lite)
        prompt = (f"The user's brain activity shows a current state of {last_status}. "
                  "Give a very brief, 1-sentence supportive health tip.")
        try:
            ai_response = gemini_model.generate_content(prompt).text
        except:
            ai_response = "Keep focusing on mindful breathing to stabilize your neural baseline."

        return render_template("index.html", 
                               results=results, 
                               clinical_note=ai_response)
            
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
