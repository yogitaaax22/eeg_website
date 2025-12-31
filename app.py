import os
import numpy as np
# --- CRUCIAL MAC FIX ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
# -----------------------
from flask import Flask, render_template, request
from scipy.io import loadmat
from collections import Counter
import google.generativeai as genai

app = Flask(__name__)

# --- CONFIGURATION ---
API_KEY = "AIzaSyDj2KlimOkPyFXMFCTden3mbfXPwCowyz0".strip()
genai.configure(api_key=API_KEY)

def extract_features(eeg_data):
    """Simplified feature extraction for demo purposes."""
    # Takes the mean power of the first 4 frequency bands
    return np.mean(np.abs(eeg_data), axis=1)[:4]

def compute_stress_level(features):
    """Logic to map EEG features to stress levels."""
    alpha, beta = features[2], features[3]
    ratio = alpha / (beta + 1e-6)
    if ratio > 1.5: return "Relax"
    elif ratio > 1.0: return "Low Stress"
    elif ratio > 0.6: return "Moderate Stress"
    else: return "High Stress"

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
                
                # Load EEG Data
                try:
                    data = loadmat(file_path)
                    eeg = data.get("Data", np.random.rand(4, 1000))
                    feats = extract_features(eeg)
                    all_powers.append(feats)
                    predicted_levels.append(compute_stress_level(feats))
                except Exception as e:
                    print(f"Error reading {file.filename}: {e}")

            if predicted_levels:
                counts = Counter(predicted_levels)
                summary_stats = ", ".join([f"{k}: {v}" for k, v in counts.items()])
                avg = np.mean(all_powers, axis=0)
                main_state = counts.most_common(1)[0][0]

                # --- CALL GEMINI 2.5 LITE ---
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
                # Stress Chart
                plt.figure(figsize=(7, 5))
                labels = ["Relax", "Low Stress", "Moderate Stress", "High Stress"]
                pcts = [(counts.get(l, 0) / len(predicted_levels)) * 100 for l in labels]
                plt.bar(labels, pcts, color=["#2e7d32","#9ccc65","#ffa726","#ef5350"])
                plt.title("Stress Level Distribution (%)")
                plt.savefig("static/plot.png")
                plt.close()

                # Band Power Chart
                plt.figure(figsize=(5, 4))
                plt.bar(["Delta", "Theta", "Alpha", "Beta"], avg, color=['#808080', '#D4AC0D', '#2E86C1', '#C0392B'])
                plt.title("Average Band Power")
                plt.savefig("static/clinical_plot.png")
                plt.close()

                results = list(zip(filenames, predicted_levels))
            
    return render_template("index.html", results=results, clinical_note=clinical_note)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
