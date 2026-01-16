# make_windows.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
WINDOW_DIR = os.path.join(DATA_DIR, "windows")
CLIENT_DIR = os.path.join(WINDOW_DIR, "clients")

os.makedirs(CLIENT_DIR, exist_ok=True)

def load_clean_data():
    path = os.path.join(DATA_DIR, "secom_clean.csv")
    df = pd.read_csv(path)
    print("Loaded secom_clean.csv with shape:", df.shape)
    return df

def create_windows(df, seq_len=10):
    X = df.drop("label", axis=1).values
    y = df["label"].values

    Xw, yw = [], []
    for i in range(len(X) - seq_len):
        Xw.append(X[i:i + seq_len])
        yw.append(y[i + seq_len - 1])

    Xw = np.array(Xw)
    yw = np.array(yw)

    print("Windowed data shape:", Xw.shape)
    print("Labels shape:", yw.shape)

    return Xw, yw

def split_into_clients(Xw, yw, n_clients=3, test_size=0.2):
    indices = np.random.permutation(len(Xw))
    shards = np.array_split(indices, n_clients)

    for i, shard in enumerate(shards, start=1):
        Xc = Xw[shard]
        yc = yw[shard]

        X_train, X_test, y_train, y_test = train_test_split(
            Xc, yc, test_size=test_size, random_state=42
        )

        np.save(os.path.join(CLIENT_DIR, f"client{i}_X_train.npy"), X_train)
        np.save(os.path.join(CLIENT_DIR, f"client{i}_y_train.npy"), y_train)
        np.save(os.path.join(CLIENT_DIR, f"client{i}_X_test.npy"), X_test)
        np.save(os.path.join(CLIENT_DIR, f"client{i}_y_test.npy"), y_test)

        print(
            f"Client {i} → "
            f"train {X_train.shape}, test {X_test.shape}"
        )

if __name__ == "__main__":
    df = load_clean_data()
    Xw, yw = create_windows(df, seq_len=10)
    split_into_clients(Xw, yw)

    print("\n✅ STEP 3 COMPLETE — windows & client datasets created")
