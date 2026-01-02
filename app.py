import os
import numpy as np
import pickle
from flask import Flask, render_template, request
from scipy.io import loadmat
from scipy.signal import welch
from scipy.stats import skew, kurtosis, mstats
import google.generativeai as genai

app = Flask(__name__)

# --- 1. GEMINI CONFIGURATION ---
# Keep your API key here as it was before
genai.configure(api_key="YOUR_GEMINI_API_KEY")
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

#logic from google colab
def extract_band_power(signal, fs=128):
    # Mirroring your exact welch parameters
    freqs, psd = welch(signal, fs=fs, nperseg=fs*2)
    bands = {'low': (0.5, 8), 'alpha': (8, 13), 'beta': (13, 30)}
    powers = []
    for band in bands.values():
        idx = (freqs >= band[0]) & (freqs <= band[1])
        powers.append(np.mean(psd[idx]))
    total = sum(powers) + 1e-6
    return [p / total for p in powers]

def extract_features_final(eeg):
    # Standardize orientation exactly like your Colab
    if eeg.shape[0] < eeg.shape[1]:
        eeg = eeg.T
    
    # EXACT 5% WINSORIZATION
    eeg = mstats.winsorize(eeg, limits=[0.05, 0.05], axis=0)
    
    # EXACT Z-SCORE NORMALIZATION
    eeg = (eeg - np.mean(eeg, axis=0)) / (np.std(eeg, axis=0) + 1e-6)

    all_channel_features = []
    for ch in range(eeg.shape[1]):
        sig = eeg[:, ch]
        rel_p = extract_band_power(sig)
        # 5 features: 3 bands + skew + kurtosis
        all_channel_features.append(rel_p + [float(skew(sig)), float(kurtosis(sig))])

    ch_feats = np.array(all_channel_features)
    
    # EXACT FRONT/BACK SPLIT
    front = ch_feats[:16].mean(axis=0)
    back = ch_feats[16:].mean(axis=0)
    
    # This creates the 10-feature vector for XGBoost
    model_in = np.concatenate([front, back]).reshape(1, -1)
    
    # Returning features for the graph (Low, Alpha, Beta)
    visual_out = [float(front[0]), float(front[1]), float(front[2])] 
    return model_in, visual_out

# --- 3. LOADING THE MODEL ---
try:
    with open('stress_model.pkl', 'rb') as f:
        stress_model = pickle.load(f)
except:
    stress_model = None

# --- 4. THE TRUTH-BASED THRESHOLDS ---
def get_label(prob):
    # Adjusted for Person 12: Relax (~0.63) and Stress (~0.90+)
    if prob > 0.92:    return "🔴 High Stress"
    elif prob > 0.82:  return "🟠 Moderate Stress"
    elif prob > 0.60:  return "🟡 Low Stress"
    else:              return "🟢 Relaxed"

# --- 5. THE WEBSITE ROUTES ---
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file: return render_template("index.html", error="No file")
        
        file_path = "temp.mat"
        file.save(file_path)
        
        try:
            data_dict = loadmat(file_path)
            eeg = data_dict.get("Data") # The exact key from your Colab
            
            if eeg is None: return render_template("index.html", error="No 'Data' key")

            # Run your Colab Engine
            model_in, vis_stats = extract_features_final(eeg)
            
            # Get Probability from XGBoost
            prob = float(stress_model.predict_proba(model_in)[0][1])
            status = get_label(prob)

            # GEMINI PART: AI explanation
            prompt = f"The EEG analysis shows a stress probability of {prob:.2f} ({status}). Provide a 2-sentence health tip."
            ai_response = gemini_model.generate_content(prompt).text

            return render_template("index.html", 
                                 prediction=status, 
                                 score=round(prob, 4),
                                 stats=vis_stats,
                                 ai_report=ai_response)
        except Exception as e:
            return render_template("index.html", error=f"Math Error: {str(e)}")
            
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
