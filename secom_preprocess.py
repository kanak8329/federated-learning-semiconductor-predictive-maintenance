# secom_preprocess.py
import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def load_secom():
    print("Loading SECOM raw files...")

    # Load features (space-separated, no header)
    X = pd.read_csv(
        os.path.join(DATA_DIR, "secom.data"),
        sep=" ",
        header=None,
        engine="python"
    )

    # Load labels
    y = pd.read_csv(
        os.path.join(DATA_DIR, "secom_labels.data"),
        sep=" ",
        header=None,
        engine="python"
    )

    # y has two columns: label and timestamp → we need only label
    y = y.iloc[:, 0]   # label column

    print(f"Raw features shape: {X.shape}")
    print(f"Raw labels shape: {y.shape}")

    return X, y


def clean_and_merge(X, y):
    print("Handling missing values...")

    # Replace all "NaN" with real numpy nan
    X = X.replace("NaN", np.nan)
    
    # Convert everything to float
    X = X.astype(float)

    # Impute missing values using median
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    X_clean = pd.DataFrame(X_imputed, columns=[f"f{i}" for i in range(X.shape[1])])

    # Convert labels from {-1, 1} → {0, 1}
    y_clean = y.replace({-1: 0})

    df = X_clean.copy()
    df["label"] = y_clean

    print("Merged dataset shape:", df.shape)
    print("Missing values after cleaning:", df.isna().sum().sum())

    return df


def scale_features(df):
    print("Scaling features...")

    scaler = StandardScaler()

    X = df.drop("label", axis=1)
    y = df["label"]

    X_scaled = scaler.fit_transform(X)

    df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    df_scaled["label"] = y

    return df_scaled


if __name__ == "__main__":
    X, y = load_secom()
    df = clean_and_merge(X, y)
    df_scaled = scale_features(df)

    out_path = os.path.join(DATA_DIR, "secom_clean.csv")
    df_scaled.to_csv(out_path, index=False)

    print("\n✅ Preprocessing complete!")
    print(f"Saved cleaned dataset → {out_path}")
