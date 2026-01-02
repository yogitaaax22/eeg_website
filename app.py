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

# --- MODEL LOADING ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'stress_model.pkl')
stress_model = None

try:
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            stress_model = pickle.load(f)
except Exception as e:
    print(f"Model Load Error: {e}")

API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    print("Warning: GOOGLE_API_KEY not found in environment variables.")

# --- THE 82% LOGIC FUNCTIONS ---
def extract_features(eeg):
    if eeg.shape[0] < eeg.shape[1]: eeg = eeg.T
    
    # Noise cleaning & Normalization
    eeg = winsorize(eeg, limits=[0.05, 0.05], axis=0)
    eeg = (eeg - np.mean(eeg, axis=0)) / (np.std(eeg, axis=0) + 1e-6)

    def get_powers(sig):
        freqs, psd = welch(sig, fs=128, nperseg=256)
        # Power for the 4 VISUAL BANDS
        d = np.mean(psd[(freqs >= 0.5) & (freqs <= 4)])
        t = np.mean(psd[(freqs >= 4) & (freqs <= 8)])
        a = np.mean(psd[(freqs >= 8) & (freqs <= 13)])
        b = np.mean(psd[(freqs >= 13) & (freqs <= 30)])
        total = d + t + a + b + 1e-6
        return [d/total, t/total, a/total, b/total, skew(sig), kurtosis(sig)]

    all_channel_feats = np.array([get_powers(eeg[:, i]) for i in range(eeg.shape[1])])
    f_mean = all_channel_feats[:16].mean(axis=0)
    b_mean = all_channel_feats[16:].mean(axis=0)

    # For XGBoost (10 features: [Low, Alpha, Beta, Skew, Kurt] x 2)
    model_in = np.concatenate([
        [f_mean[0]+f_mean[1], f_mean[2], f_mean[3], f_mean[4], f_mean[5]],
        [b_mean[0]+b_mean[1], b_mean[2], b_mean[3], b_mean[4], b_mean[5]]
    ]).reshape(1, -1)
    
    # For your Bar Chart (4 features)
    vis_out = f_mean[:4] 
    
    return model_in, vis_out

def compute_stress_level(model_in):
    if stress_model is None: return "Model Error"
    prob = stress_model.predict_proba(model_in)[:, 1][0]
    # YOUR GOLDEN 0.6 THRESHOLD
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
                    
                    # Call the accuracy engine
                    m_in, v_out = extract_features(eeg)
                    
                    all_powers.append(v_out)
                    predicted_levels.append(compute_stress_level(m_in))
                except Exception as e:
                    print(f"Error reading {file.filename}: {e}")

            if predicted_levels:
                counts = Counter(predicted_levels)
                summary_stats = ", ".join([f"{k}: {v}" for k, v in counts.items()])
                avg = np.mean(all_powers, axis=0)
                main_state = counts.most_common(1)[0][0]

                # --- CALL GEMINI 2.5 LITE (EXACTLY AS YOU HAD IT) ---
                try:
                    model = genai.GenerativeModel('gemini-2.5-flash-lite')
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
                    response = model.generate_content(prompt)
                    clinical_note = response.text.strip().replace("\n\n", "<br><br>")
                except Exception as e:
                    print(f"Gemini fallback active: {e}")
                    clinical_note = (
                        f"<b>System Analysis:</b> A dominant state of {main_state} was detected. "
                        f"Distribution: {summary_stats}. <br><br><b>Neural Pattern:</b> "
                        f"Alpha power ({avg[2]:.4f}) vs Beta ({avg[3]:.4f}) supports the classification."
                    )

                # --- PLOTTING ---
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
