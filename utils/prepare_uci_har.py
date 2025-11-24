import numpy as np
import os
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import normalize, create_windows


def load_signals(folder):
    """
    Loads UCI HAR signals from Inertial Signals folder
    Returns array of shape (N, 128, 6)
    """

    # Sensor signal file names
    signal_files = [
        "body_acc_x",
        "body_acc_y",
        "body_acc_z",
        "body_gyro_x",
        "body_gyro_y",
        "body_gyro_z"
    ]

    signals = []

    for signal in signal_files:
        # Example: body_acc_x_train.txt
        file_path_train = os.path.join(folder, f"{signal}_train.txt")
        file_path_test  = os.path.join(folder, f"{signal}_test.txt")

        train_data = np.loadtxt(file_path_train)
        test_data  = np.loadtxt(file_path_test)

        signals.append(np.vstack([train_data, test_data]))  # combine train+test

    # Stack into shape (6, N, 128) then transpose to (N, 128, 6)
    stacked = np.stack(signals, axis=2)
    return stacked  # shape: (N, 128, 6)

def load_labels(folder):
    train_labels = np.loadtxt(os.path.join(folder, "y_train.txt")).astype(int)
    test_labels  = np.loadtxt(os.path.join(folder, "y_test.txt")).astype(int)
    return np.concatenate([train_labels, test_labels], axis=0)


def main():
    print("Loading UCI HAR dataset...")

    DATA_ROOT = "data/UCIHAR"  # adjust if your folder name is different

    # Load raw signals
    signals = load_signals(os.path.join(DATA_ROOT, "Inertial Signals"))
    labels  = load_labels(DATA_ROOT)

    print("Raw Signals shape:", signals.shape)  # should be (10299,128,6)
    print("Labels shape:", labels.shape)

    # Normalize
    for i in range(6):
        signals[:,:,i] = normalize(signals[:,:,i])

    # Save final numpy arrays
    np.save("data/features.npy", signals)
    np.save("data/labels.npy", labels)

    print("Saved files:")
    print(" → data/features.npy")
    print(" → data/labels.npy")

if __name__ == "__main__":
    main()
