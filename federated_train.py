# Client 1 (Fab A) ─┐
# Client 2 (Fab B) ─┼──> FedAvg Aggregation ──> Global Model
# Client 3 (Fab C) ─┘


# federated_train.py
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import LSTMClassifier
from utils import device, train_one_epoch, evaluate

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
CLIENT_DIR = "data/windows/clients"
RESULTS_DIR = "results"
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

ROUNDS = 5
LOCAL_EPOCHS = 1
LR = 0.001

CLIENT_IDS = [1, 2, 3]

# -------------------------------------------------
# UTILS
# -------------------------------------------------
def load_client_data(cid):
    X_train = np.load(f"{CLIENT_DIR}/client{cid}_X_train.npy")
    y_train = np.load(f"{CLIENT_DIR}/client{cid}_y_train.npy")
    X_test = np.load(f"{CLIENT_DIR}/client{cid}_X_test.npy")
    y_test = np.load(f"{CLIENT_DIR}/client{cid}_y_test.npy")
    return X_train, y_train, X_test, y_test

def average_weights(state_dicts):
    avg_dict = {}
    for key in state_dicts[0]:
        avg_dict[key] = sum(sd[key] for sd in state_dicts) / len(state_dicts)
    return avg_dict

# -------------------------------------------------
# FEDERATED TRAINING
# -------------------------------------------------
def federated_training():
    # Initialize global model
    sample_X = np.load(f"{CLIENT_DIR}/client1_X_train.npy")
    n_features = sample_X.shape[2]
    global_model = LSTMClassifier(n_features).to(device)

    loss_fn = torch.nn.BCELoss()

    global_acc = []

    print("\n🚀 Starting Federated Learning\n")

    for rnd in range(1, ROUNDS + 1):
        print(f"\n🌐 Round {rnd}")
        local_states = []

        for cid in CLIENT_IDS:
            X_train, y_train, X_test, y_test = load_client_data(cid)

            # Local model
            local_model = LSTMClassifier(n_features).to(device)
            local_model.load_state_dict(global_model.state_dict())

            optimizer = torch.optim.Adam(local_model.parameters(), lr=LR)

            for _ in range(LOCAL_EPOCHS):
                train_one_epoch(
                    local_model, optimizer, loss_fn,
                    X_train, y_train
                )

            local_states.append(
                {k: v.cpu() for k, v in local_model.state_dict().items()}
            )

        # FedAvg aggregation
        avg_state = average_weights(local_states)
        global_model.load_state_dict(avg_state)

        # Evaluate global model on client 1 test set
        _, _, X_test, y_test = load_client_data(1)
        metrics = evaluate(global_model, X_test, y_test)
        global_acc.append(metrics["accuracy"])

        print(
            f"Global Model → Accuracy: {metrics['accuracy']:.4f}, "
            f"F1: {metrics['f1']:.4f}"
        )

    # Save federated model
    torch.save(
        global_model.state_dict(),
        os.path.join(RESULTS_DIR, "federated_model.pt")
    )

    # Plot accuracy over rounds
    plt.figure()
    plt.plot(global_acc, marker="o")
    plt.title("Federated Model Accuracy over Rounds")
    plt.xlabel("Federated Round")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, "federated_accuracy.png"))
    plt.close()

    print("\n✅ Federated training complete")
    print("Model saved → results/federated_model.pt")
    print("Plot saved → results/plots/federated_accuracy.png")

# -------------------------------------------------
if __name__ == "__main__":
    federated_training()
