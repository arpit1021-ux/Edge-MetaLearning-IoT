import numpy as np

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)

def create_windows(data, labels, window=128, stride=64):
    X, y = [], []
    for i in range(0, len(data)-window, stride):
        X.append(data[i:i+window])
        y.append(labels[i+window])
    return np.array(X), np.array(y)
