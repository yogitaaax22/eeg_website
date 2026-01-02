import pickle
import numpy as np
from scipy.io import loadmat
from scipy.signal import welch
from scipy.stats import skew, kurtosis, mstats


# ---------------- LOAD TRAINED OBJECTS ----------------
with open("stress_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("baseline.pkl", "rb") as f:
    baseline = pickle.load(f)


# ---------------- FEATURE FUNCTIONS ----------------
def extract_band_power(signal, fs=128):
    freqs, psd = welch(signal, fs=fs, nperseg=fs*2)
    bands = [(0.5, 8), (8, 13), (13, 30)]

    powers = []
    for low, high in bands:
        idx = (freqs >= low) & (freqs <= high)
        powers.append(np.mean(psd[idx]))

    total = sum(powers) + 1e-6
    return [p / total for p in powers]


def preprocess_eeg(eeg):
    eeg = np.array(eeg)

    # Ensure (time, channels)
    if eeg.shape[0] < eeg.shape[1]:
        eeg = eeg.T

    # Winsorize (EXACT as training)
    eeg = mstats.winsorize(eeg, limits=[0.05, 0.05], axis=0)

    # Per-channel normalization
    eeg = (eeg - np.mean(eeg, axis=0)) / (np.std(eeg, axis=0) + 1e-6)

    all_channel_features = []

    for ch in range(eeg.shape[1]):
        sig = eeg[:, ch]
        rel_p = extract_band_power(sig)
        all_channel_features.append(
            rel_p + [skew(sig), kurtosis(sig)]
        )

    ch_feats = np.array(all_channel_features)

    # Frontal + back averaging (EXACT)
    front = ch_feats[:16].mean(axis=0)
    back = ch_feats[16:].mean(axis=0)

    features = np.concatenate([front, back])  # 10 features
    return features


def predict_from_raw(raw_eeg):
    features = preprocess_eeg(raw_eeg)

    # Baseline correction (EXACT)
    features = features - baseline

    X = features.reshape(1, -1)

    # Probability-based thresholding
    prob = model.predict_proba(X)[0, 1]
    return int(prob > 0.6)
