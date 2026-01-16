# utils.py
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batch_iter(X, y, batch_size=64, shuffle=True):
    idx = np.arange(len(X))
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, len(X), batch_size):
        b = idx[i:i+batch_size]
        yield X[b], y[b]

def train_one_epoch(model, optimizer, loss_fn, X, y, batch_size=64):
    model.train()
    total_loss = 0.0
    for xb, yb in batch_iter(X, y, batch_size):
        xb_t = torch.tensor(xb, dtype=torch.float32, device=device)
        yb_t = torch.tensor(yb, dtype=torch.float32, device=device)
        optimizer.zero_grad()
        out = model(xb_t)
        loss = loss_fn(out, yb_t)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
    return total_loss / len(X)

def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32, device=device)
        preds = model(xb).cpu().numpy()
        preds_bin = (preds >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y, preds_bin)),
        "f1": float(f1_score(y, preds_bin, zero_division=0)),
        "precision": float(precision_score(y, preds_bin, zero_division=0)),
        "recall": float(recall_score(y, preds_bin, zero_division=0)),
    }
    # compute AUC only if positive class exists
    try:
        metrics["roc_auc"] = float(roc_auc_score(y, preds))
    except Exception:
        metrics["roc_auc"] = None
    return metrics
