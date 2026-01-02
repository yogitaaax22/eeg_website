# model_pipeline.py
import numpy as np
import pickle
from scipy.signal import welch
from scipy.stats import skew, kurtosis, mstats

# Load trained model and baseline
with open("stress_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("baseline.pkl", "rb") as f:
    baseline = pickle.load(f)

# -----------------------
# EXACT FEATURE FUNCTIONS
# -----------------------

def extract_band_power(signal, fs=128):
    """
    Extract relative band powers for low, alpha, beta bands
    """
    freqs, psd = welch(signal, fs=fs, nperseg=fs*2)
    bands = {'low': (0.5, 8), 'alpha': (8, 13), 'beta': (13, 30)}

    powers = []
    for band in bands.values():
        idx = (freqs >= band[0]) & (freqs <= band[1])
        powers.append(np.mean(psd[idx]))

    total = sum(powers) + 1e-6
    return [p / total for p in powers]

def preprocess_eeg(eeg):
    """
    Preprocess raw EEG exactly as in Colab:
    - Transpose if needed
    - Winsorize 5%
    - Normalize per channel
    - Extract band powers + skew/kurtosis
    - Average front and back channels
    """
    eeg = np.array(eeg)
    if eeg.shape[0] < eeg.shape[1]:
        eeg = eeg.T  # same as Colab

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

# -----------------------
# PREDICTION FUNCTION
# -----------------------

def predict_from_raw(eeg):
    """
    Predict stress (1) or relax (0) using exact Colab pipeline
    Returns 0/1
    """
    features = preprocess_eeg(eeg)
    prob = model.predict_proba([features])[0][1]
    threshold = 0.6  # exact same as Colab
    return int(prob > threshold)

def predict_prob_from_raw(eeg):
    """
    Return raw probability (optional)
    """
    features = preprocess_eeg(eeg)
    prob = model.predict_proba([features])[0][1]
    return prob
