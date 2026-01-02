import os
import numpy as np
import pickle
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from scipy.io import loadmat
from scipy.signal import welch
from scipy.stats import skew, kurtosis, mstats
import google.generativeai as genai

# Use 'Agg' backend to prevent Matplotlib from crashing on Render
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# --- 1. GEMINI CONFIGURATION ---
genai.configure(api_key="YOUR_GEMINI_API_KEY")
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# --- 2. THE PROCESSING ENGINE ---
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
    if eeg.shape[0] < eeg.shape[1]: eeg = eeg.T
    eeg = mstats.winsorize(eeg, limits=[0.05, 0.05], axis=0)
    eeg = (eeg - np.mean(eeg, axis=0)) / (np.std(eeg, axis=0) + 1e-6)
    all_channel_features = []
    for ch in range(eeg.shape[1]):
        sig = eeg[:, ch]
        rel_p = extract_band_power(sig)
        all_channel_features.append(rel_p + [float(skew(sig)), float(kurtosis(sig))])
    ch_feats = np.array(all_channel_features)
    front, back = ch_feats[:16].mean(axis=0), ch_feats[16:].mean(axis=0)
    return np.concatenate([front, back]).reshape(1, -1), eeg

# --- 3. GRAPH GENERATOR ---
def generate_graph(eeg_data):
    plt.figure(figsize=(10, 4))
    plt.plot(eeg_data[:, 0], color='#00d2ff', linewidth=0.5) # Plot first channel
    plt.title("Neural Activity Pattern")
    plt.axis('off')
    
    buf = BytesIO()
    plt.savefig(buf, format="png", transparent=True)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# --- 4. THE ROUTES ---
try:
    with open('stress_model.pkl', 'rb') as f: stress_model = pickle.load(f)
except: stress_model = None

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    ai_response = ""
    if request.method == "POST":
        files = request.files.getlist("mat_files")
        for file in files:
            try:
                data = loadmat(file)
                eeg = data.get('Data') or data.get('data')
                if eeg is not None:
                    features, clean_eeg = extract_features_final(eeg)
                    prob = float(stress_model.predict_proba(features)[0][1]) if stress_model else 0.5
                    
                    # Calibration Threshold (0.82)
                    status = "High Stress" if prob > 0.96 else "Moderate Stress" if prob > 0.82 else "Relaxed"
                    
                    # Generate the visual graph
                    graph_url = generate_graph(clean_eeg)
                    results.append({'filename': file.filename, 'status': status, 'graph': graph_url})
            except: continue

        if results:
            try:
                # Use standard model name to avoid fallback
                prompt = f"EEG analysis result: {results[0]['status']}. Give a 1-sentence supportive health tip."
                ai_response = gemini_model.generate_content(prompt).text
            except:
                ai_response = "Take a deep breath and center your neural activity."

    return render_template("index.html", results=results, clinical_note=ai_response)

if __name__ == "__main__":
    app.run(debug=True)
