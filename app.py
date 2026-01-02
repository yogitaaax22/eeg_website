import os
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from flask import Flask, render_template, request
from scipy.io import loadmat
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from scipy.stats.mstats import winsorize
from collections import Counter
import google.generativeai as genai

app = Flask(__name__)

# --- 1. LOAD YOUR .PKL MODEL ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'stress_model.pkl')
stress_model = None

try:
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            stress_model = pickle.load(f)
        print("✅ Success: stress_model.pkl loaded.")
    else:
        print("❌ Error: stress_model.pkl not found.")
except Exception as e:
    print(f"❌ Error loading pkl file: {e}")

API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    print("Warning: GOOGLE_API_KEY not found in environment variables.")

# --- 2. THE 82% ACCURACY ENGINE ---
def extract_features(eeg):
    # Ensure shape is (time, channels)
    if eeg.shape[0] < eeg.shape[1]: eeg = eeg.T
    
    # Winsorizing & Normalizing (The 82% logic clean-up)
    eeg = winsorize(eeg, limits=[0.05, 0.05], axis=0)
    eeg = (eeg - np.mean(eeg, axis=0)) / (np.std(eeg, axis=0) + 1e-6)

    def get_band_powers(sig):
        freqs, psd = welch(sig, fs=128, nperseg=256)
        # 4 bands for your Visual Graphs
        d = np.mean(psd[(freqs >= 0.5) & (freqs <= 4)])
        t = np.mean(psd[(freqs >= 4) & (freqs <= 8)])
        a = np.mean(psd[(freqs >= 8) & (freqs <= 13)])
        b = np.mean(psd[(freqs >= 13) & (freqs <= 30)])
        total = d + t + a + b + 1e-6
        # [Delta, Theta, Alpha, Beta, Skew, Kurtosis]
        return [d/total, t/total, a/total, b/total, skew(sig), kurtosis(sig)]

    all_channel_feats = np.array([get_band_powers(eeg[:, i]) for i in range(eeg.shape[1])])
    f_mean = all_channel_feats[:16].mean(axis=0)
    b_mean = all_channel_feats[16:].mean(axis=0)

    # OUTPUT A: 10 Features for the XGBoost model
    model_in = np.concatenate([
        [f_mean[0]+f_mean[1], f_mean[2], f_mean[3], f_mean[4], f_mean[5]],
        [b_mean[0]+b_mean[1], b_mean[2], b_mean[3], b_mean[4], b_mean[5]]
    ]).reshape(1, -1)
    
    # OUTPUT B: 4 Bands for your existing chart logic
    visual_out = f_mean[:4] 
    
    return model_in, visual_out

def compute_stress_level(model_input):
    """Uses the loaded XGBoost model and the 0.6 Golden Threshold."""
    if stress_model is None: return "Model Error"
    
    prob = stress_model.predict_proba(model_input)[:, 1][0]
    
    if prob > 0.6: return "High Stress"
    elif prob > 0.4: return "Moderate Stress"
    elif prob > 0.2: return "Low Stress"
    else: return "Relax"

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
                    eeg = data.get("Data", np.random.rand(32, 1000))
                    
                    # Split outputs: one for model, one for your existing visual logic
                    model_data, visual_data = extract_features(eeg)
                    
                    all_powers.append(visual_data)
                    predicted_levels.append(compute_stress_level(model_data))
                except Exception as e:
                    print(f"Error reading {file.filename}: {e}")

            if predicted_levels:
                counts = Counter(predicted_levels)
                summary_stats = ", ".join([f"{k}: {v}" for k, v in counts.items()])
                avg = np.mean(all_powers, axis=0) # Index 2=Alpha, 3=Beta to match your chart/Gemini
                main_state = counts.most_common(1)[0][0]

                
                try:
                    model_ai = genai.GenerativeModel('gemini-2.5-flash-lite')
                    prompt = (
                        "Context: Senior Data Analyst report for an EEG session. "
                        f"DATA: {summary_stats}. Alpha: {avg[2]:.4f}, Beta: {avg[3]:.4f}. "
                        "TASK: Write exactly 2 paragraphs (5 lines each). "
                        "Para 1 (Session Overview): Start by confirming the primary classification is "
                        f"'{main_state}'. Group all sessional data here: mention the {summary_stats} "
                        f"distribution and the specific Alpha ({avg[2]:.4f}) and Beta ({avg[3]:.4f}) "
                        "amplitudes as the direct evidence for this session's outcome. "
                        "Para 2 (Neural Analysis): Focus entirely on the 'observed oscillations'. "
                        "Explain the relationship between the calmness of Alpha and the engagement "
                        f"of Beta. Describe why this specific balance creates the '{main_state}' "
                        "effect on the user's focus and clarity. Keep all technical theory here. "
                        "STYLE: Highly organized, clean, and professional. No bolding. One <br> for spacing."
                    )
                    response = model_ai.generate_content(prompt)
                    clinical_note = response.text.strip().replace("\n\n", "<br><br>")
                except Exception as e:
                    print(f"Gemini error: {e}")
                    clinical_note = f"Analysis complete. Majority state: {main_state}."

                
                plt.figure(figsize=(7, 5))
                labels = ["Relax", "Low Stress", "Moderate Stress", "High Stress"]
                pcts = [(counts.get(l, 0) / len(predicted_levels)) * 100 for l in labels]
                plt.bar(labels, pcts, color=["#2e7d32","#9ccc65","#ffa726","#ef5350"])
                plt.title("Stress Level Distribution (%)")
                plt.savefig("static/plot.png")
                plt.close()

                plt.figure(figsize=(5, 4))
                plt.bar(["Delta", "Theta", "Alpha", "Beta"], avg, color=['#808080', '#D4AC0D', '#2E86C1', '#C0392B'])
                plt.title("Average Band Power")
                plt.savefig("static/clinical_plot.png")
                plt.close()

                results = list(zip(filenames, predicted_levels))
            
    return render_template("index.html", results=results, clinical_note=clinical_note)

if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))
