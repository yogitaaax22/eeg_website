# model_pipeline.py
import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis, mstats
import pickle

# -------------------------
# Load trained model & baseline
# -------------------------
with open("stress_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("baseline.pkl", "rb") as f:
    baseline = pickle.load(f)

# -------------------------
# Feature extraction
# -------------------------
def extract_band_power(signal, fs=128):
    freqs, psd = welch(signal, fs=fs, nperseg=fs*2)
    bands = {'low': (0.5, 8), 'alpha': (8, 13), 'beta': (13, 30)}
    powers = []
    for band in bands.values():
        idx = (freqs >= band[0]) & (freqs <= band[1])
        powers.append(np.mean(psd[idx]))
    total = sum(powers) + 1e-6
    return [p / total for p in powers]

# -------------------------
# Preprocessing function
# -------------------------
def preprocess_eeg(eeg):
    eeg = np.array(eeg)
    if eeg.shape[0] < eeg.shape[1]:
        eeg = eeg.T

    eeg = mstats.winsorize(eeg, limits=[0.05, 0.05], axis=0)
    eeg = (eeg - np.mean(eeg, axis=0)) / (np.std(eeg, axis=0) + 1e-6)

    all_channel_features = []
    for ch in range(eeg.shape[1]):
        sig = eeg[:, ch]
        rel_p = extract_band_power(sig)
        all_channel_features.append(rel_p + [skew(sig), kurtosis(sig)])

    ch_feats = np.array(all_channel_features)
    front = ch_feats[:16].mean(axis=0)
    back = ch_feats[16:].mean(axis=0)
    features = np.concatenate([front, back])
    return features

# -------------------------
# Prediction functions
# -------------------------
def predict_from_raw(eeg):
    features = preprocess_eeg(eeg)
    prob = model.predict_proba([features])[0][1]
    threshold = 0.6
    return int(prob > threshold)

def predict_prob_from_raw(eeg):
    features = preprocess_eeg(eeg)
    prob = model.predict_proba([features])[0][1]
    return prob
