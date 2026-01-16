# compare_models.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import LSTMClassifier
from utils import evaluate, device

RESULTS_DIR = "results"
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")
CLIENT_DIR = "data/windows/clients"

def load_test_data(client_id=1):
    X_test = np.load(f"{CLIENT_DIR}/client{client_id}_X_test.npy")
    y_test = np.load(f"{CLIENT_DIR}/client{client_id}_y_test.npy")
    return X_test, y_test

def main():
    X_test, y_test = load_test_data(1)
    n_features = X_test.shape[2]

    # Load centralized model
    centralized_model = LSTMClassifier(n_features).to(device)
    centralized_model.load_state_dict(
        torch.load(os.path.join(RESULTS_DIR, "centralized_model.pt"), map_location=device)
    )

    # Load federated model
    federated_model = LSTMClassifier(n_features).to(device)
    federated_model.load_state_dict(
        torch.load(os.path.join(RESULTS_DIR, "federated_model.pt"), map_location=device)
    )

    # Evaluate
    cent_metrics = evaluate(centralized_model, X_test, y_test)
    fed_metrics = evaluate(federated_model, X_test, y_test)

    print("\n📊 Model Comparison")
    print("---------------------")
    print("Centralized →", cent_metrics)
    print("Federated   →", fed_metrics)

    # Plot comparison
    labels = ["Accuracy", "F1-score"]
    cent_values = [cent_metrics["accuracy"], cent_metrics["f1"]]
    fed_values = [fed_metrics["accuracy"], fed_metrics["f1"]]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(6,4))
    plt.bar(x - width/2, cent_values, width, label="Centralized")
    plt.bar(x + width/2, fed_values, width, label="Federated")

    plt.xticks(x, labels)
    plt.ylabel("Score")
    plt.title("Centralized vs Federated Model Performance")
    plt.legend()
    plt.grid(True, axis="y")

    out_path = os.path.join(PLOT_DIR, "centralized_vs_federated.png")
    plt.savefig(out_path, dpi=200)
    plt.close()

    print("\n✅ Comparison plot saved →", out_path)

if __name__ == "__main__":
    main()
