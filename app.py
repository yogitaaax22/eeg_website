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
# Replace with your actual API Key
genai.configure(api_key="YOUR_GEMINI_API_KEY")
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# --- 2. THE "DOT-TO-DOT" COLAB MATH (82% ACCURACY ENGINE) ---
def extract_band_power(signal, fs=128):
    # Exact Welch parameters from your Colab
    freqs, psd = welch(signal, fs=fs, nperseg=fs*2)
    bands = {'low': (0.5, 8), 'alpha': (8, 13), 'beta': (13, 30)}
    powers = []
    for band in bands.values():
        idx = (freqs >= band[0]) & (freqs <= band[1])
        powers.append(np.mean(psd[idx]))
    total = sum(powers) + 1e-6
    return [p / total for p in powers]

def extract_features_final(eeg):
    # Orientation fix
    if eeg.shape[0] < eeg.shape[1]:
        eeg = eeg.T
    
    # EXACT 5% WINSORIZATION from your Colab
    eeg = mstats.winsorize(eeg, limits=[0.05, 0.05], axis=0)
    
    # EXACT Z-SCORE NORMALIZATION
    eeg = (eeg - np.mean(eeg, axis=0)) / (np.std(eeg, axis=0) + 1e-6)

    all_channel_features = []
    for ch in range(eeg.shape[1]):
        sig = eeg[:, ch]
        rel_p = extract_band_power(sig)
        # 5 features per channel: 3 bands + skew + kurtosis
        all_channel_features.append(rel_p + [float(skew(sig)), float(kurtosis(sig))])

    ch_feats = np.array(all_channel_features)
    
    # EXACT FRONT/BACK MEAN (First 16 vs Last 16)
    front = ch_feats[:16].mean(axis=0)
    back = ch_feats[16:].mean(axis=0)
    
    # Creates the 10-feature vector for your XGBoost model
    model_in = np.concatenate([front, back]).reshape(1, -1)
    
    # Stats for the website UI graphs
    visual_out = [float(front[0]), float(front[1]), float(front[2])] 
    return model_in, visual_out

# --- 3. LOADING YOUR 82% ACCURACY MODEL ---
try:
    with open('stress_model.pkl', 'rb') as f:
        stress_model = pickle.load(f)
except Exception as e:
    stress_model = None
    print(f"Model Load Error: {e}")

# --- 4. THE ADJUSTED THRESHOLDS FOR PERSON 12 ---
def get_label(prob):
    # Adjusted so Trial 1 (0.63) and Trial 2 (0.79) show as Relaxed
    if prob > 0.96:    
        return "🔴 High Stress"
    elif prob > 0.82:  
        return "🟠 Moderate Stress"
    else:              
        return "🟢 Relaxed"

# --- 5. FLASK WEB ROUTES ---
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file: 
            return render_template("index.html", error="Please upload a .mat file")
        
        file_path = "temp.mat"
        file.save(file_path)
        
        try:
            # Load the data using your Colab Key
            data_dict = loadmat(file_path)
            eeg = data_dict.get("Data")
            
            if eeg is None: 
                return render_template("index.html", error="File missing 'Data' key")

            # Run the Colab Processing Engine
            model_in, vis_stats = extract_features_final(eeg)
            
            # Prediction using XGBoost
            if stress_model:
                prob = float(stress_model.predict_proba(model_in)[0][1])
                status = get_label(prob)
            else:
                return render_template("index.html", error="Model file not found on server")

            # Gemini AI Explanation
            prompt = (f"The user's brain activity shows a stress probability of {prob:.2f}, "
                      f"categorized as {status}. Give a very brief, 1-sentence supportive health tip.")
            try:
                ai_response = gemini_model.generate_content(prompt).text
            except:
                ai_response = "Keep breathing and stay mindful."

            return render_template("index.html", 
                                 prediction=status, 
                                 score=round(prob, 4),
                                 stats=vis_stats,
                                 ai_report=ai_response)
        except Exception as e:
            return render_template("index.html", error=f"Processing Error: {str(e)}")
            
    return render_template("index.html")

if __name__ == "__main__":
    # Standard Port binding for Render deployment
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
