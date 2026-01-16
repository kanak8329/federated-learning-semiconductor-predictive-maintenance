# centralized_train.py
# import os
# # import glob
# # import numpy as np
# # import torch
# # import matplotlib.pyplot as plt
# #
# # from model import LSTMClassifier
# # from utils import device, train_one_epoch, evaluate
# #
# # RESULTS_DIR = "results"
# # PLOT_DIR = os.path.join(RESULTS_DIR, "plots")
# # os.makedirs(PLOT_DIR, exist_ok=True)


# centralized_train.py
import os
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import LSTMClassifier
from utils import device, train_one_epoch, evaluate

RESULTS_DIR = os.path.abspath("results")
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

def load_all_clients_train():
    Xs, ys = [], []
    for path in glob.glob("data/windows/clients/*_X_train.npy"):
        Xs.append(np.load(path))
        ys.append(np.load(path.replace("_X_train.npy", "_y_train.npy")))
    return np.concatenate(Xs), np.concatenate(ys)

def load_test_client(client_id=1):
    X = np.load(f"data/windows/clients/client{client_id}_X_test.npy")
    y = np.load(f"data/windows/clients/client{client_id}_y_test.npy")
    return X, y

def main():
    X_train, y_train = load_all_clients_train()
    X_test, y_test = load_test_client(1)

    model = LSTMClassifier(X_train.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.BCELoss()

    history_loss, history_acc = [], []

    for epoch in range(1, 16):
        loss = train_one_epoch(model, optimizer, loss_fn, X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        history_loss.append(loss)
        history_acc.append(metrics["accuracy"])
        print(f"Epoch {epoch} | Loss={loss:.4f} | Acc={metrics['accuracy']:.4f}")

    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "centralized_model.pt"))

    plt.plot(history_loss)
    plt.title("Training Loss")
    plt.savefig(os.path.join(PLOT_DIR, "loss.png"))
    plt.close()

    plt.plot(history_acc)
    plt.title("Test Accuracy")
    plt.savefig(os.path.join(PLOT_DIR, "accuracy.png"))
    plt.close()

    print("\n✅ Centralized training finished.")
    print("Model + plots saved in results/")

if __name__ == "__main__":
    main()
