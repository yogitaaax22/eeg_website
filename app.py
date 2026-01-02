import os
import numpy as np
import pickle # Added for model loading
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from flask import Flask, render_template, request
from scipy.io import loadmat
from scipy.signal import welch # Added for 82% math
from scipy.stats import skew, kurtosis # Added for 82% math
from scipy.stats.mstats import winsorize # Added for 82% math
from collections import Counter
import google.generativeai as genai

app = Flask(__name__)

# --- 82% MODEL LOADING ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'stress_model.pkl')
stress_model = None

try:
    with open(model_path, 'rb') as f:
        stress_model = pickle.load(f)
except Exception as e:
    print(f"Model Load Error: {e}")

API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    print("Warning: GOOGLE_API_KEY not found in environment variables.")

# --- 82% ACCURACY MATH (From your Colab) ---
def extract_features(eeg):
    if eeg.shape[0] < eeg.shape[1]: eeg = eeg.T
    
    # 1. Winsorize & Normalize (Your exact Colab steps)
    eeg = winsorize(eeg, limits=[0.05, 0.05], axis=0)
    eeg = (eeg - np.mean(eeg, axis=0)) / (np.std(eeg, axis=0) + 1e-6)

    def get_band_powers(sig):
        freqs, psd = welch(sig, fs=128, nperseg=256)
        # Your Colab ranges: low (0.5-8), alpha (8-13), beta (13-30)
        low = np.mean(psd[(freqs >= 0.5) & (freqs <= 8)])
        alpha = np.mean(psd[(freqs >= 8) & (freqs <= 13)])
        beta = np.mean(psd[(freqs >= 13) & (freqs <= 30)])
        total = low + alpha + beta + 1e-6
        # Returns: low_rel, alpha_rel, beta_rel, skew, kurtosis
        return [low/total, alpha/total, beta/total, skew(sig), kurtosis(sig)]

    all_channel_feats = []
    for ch in range(eeg.shape[1]):
        all_channel_feats.append(get_band_powers(eeg[:, ch]))
    
    ch_feats = np.array(all_channel_feats)
    # Front (0-15) and Back (16-31)
    front = ch_feats[:16].mean(axis=0)
    back = ch_feats[16:].mean(axis=0)
    
    # Final 10 features for XGBoost
    return np.concatenate([front, back]).reshape(1, -1)

def compute_stress_level(feats):
    """Uses the 82% XGBoost model with the 0.6 Golden Threshold."""
    if stress_model is None: return "Model Error"
    
    # Get probability of class 1 (Stress)
    prob = stress_model.predict_proba(feats)[:, 1][0]
    
    # Apply your 82% accuracy threshold
    if prob > 0.6:
        return "High Stress"
    elif prob > 0.4: # Optional: adding a middle ground for your visuals
        return "Moderate Stress"
    elif prob > 0.2:
        return "Low Stress"
    else:
        return "Relax"

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    clinical_note = ""
    
    if request.method == "POST":
        uploaded_files = request.files.getlist("mat_files")
        if uploaded_files:
            if not os.path.exists("static"): os.makedirs("static")
            
            predicted_levels, filenames, all_powers = [], [], []

            for file in uploaded_files:
                file_path = os.path.join("static", file.filename)
                file.save(file_path)
                filenames.append(file.filename)
                
                try:
                    data = loadmat(file_path)
                    eeg = data.get("Data")
                    
                    # Run the 82% extraction
                    feats = extract_features(eeg)
                    
                    # Save "avg" for your existing chart logic (Alpha is index 1, Beta is index 2 in our 10-feat vector)
                    # We take the front-back average to keep your chart working
                    all_powers.append(feats[0]) 
                    
                    predicted_levels.append(compute_stress_level(feats))
                except Exception as e:
                    print(f"Error reading {file.filename}: {e}")

            if predicted_levels:
                counts = Counter(predicted_levels)
                summary_stats = ", ".join([f"{k}: {v}" for k, v in counts.items()])
                avg = np.mean(all_powers, axis=0) # Indexing here to match your Delta/Theta/Alpha/Beta chart
                main_state = counts.most_common(1)[0][0]

                # --- YOUR ORIGINAL GEMINI 2.5 LITE LOGIC ---
                try:
                    model_gen = genai.GenerativeModel('gemini-2.0-flash-lite')
                    prompt = (
                        "Context: Senior Data Analyst report for an EEG session. "
                        f"DATA: {summary_stats}. Alpha: {avg[1]:.4f}, Beta: {avg[2]:.4f}. "
                        "TASK: Write exactly 2 paragraphs (5 lines each). "
                        "Para 1 (Session Overview): Start by confirming the primary classification is "
                        f"'{main_state}'. Group all sessional data here: mention the {summary_stats} "
                        f"distribution and the specific Alpha ({avg[1]:.4f}) and Beta ({avg[2]:.4f}) "
                        "amplitudes as the direct evidence for this session's outcome. "
                        "Para 2 (Neural Analysis): Focus entirely on the 'observed oscillations'. "
                        "Explain the relationship between the calmness of Alpha and the engagement "
                        f"of Beta. Describe why this specific balance creates the '{main_state}' "
                        "effect on the user's focus and clarity. Keep all technical theory here. "
                        "STYLE: Highly organized, clean, and professional. No bolding. One <br> for spacing."
                    )
                    response = model_gen.generate_content(prompt)
                    clinical_note = response.text.strip().replace("\n\n", "<br><br>")
                except Exception as e:
                    print(f"Gemini fallback active: {e}")
                    clinical_note = f"Analysis complete. Detected {main_state} state."

                # --- YOUR ORIGINAL PLOTTING ---
                plt.figure(figsize=(7, 5))
                labels = ["Relax", "Low Stress", "Moderate Stress", "High Stress"]
                pcts = [(counts.get(l, 0) / len(predicted_levels)) * 100 for l in labels]
                plt.bar(labels, pcts, color=["#2e7d32","#9ccc65","#ffa726","#ef5350"])
                plt.title("Stress Level Distribution (%)")
                plt.savefig("static/plot.png")
                plt.close()

                plt.figure(figsize=(5, 4))
                # Using the relative powers from your feature extraction for the chart
                plt.bar(["Low-Freq", "Alpha", "Beta", "Skew"], [avg[0], avg[1], avg[2], avg[3]], color=['#808080', '#D4AC0D', '#2E86C1', '#C0392B'])
                plt.title("Average Band Power")
                plt.savefig("static/clinical_plot.png")
                plt.close()

                results = list(zip(filenames, predicted_levels))
            
    return render_template("index.html", results=results, clinical_note=clinical_note)

if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))
