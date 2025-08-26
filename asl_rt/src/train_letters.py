# src/train_letters.py
import os, pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from features import row_box_normalize  # <-- import here

RAW = "data/raw/hand_signals.csv"
OUT = "models/letters_sklearn.pkl"

def _select_feature_columns(df):
    want = []
    for i in range(21):
        x, y = f"{i}x", f"{i}y"
        if x in df.columns and y in df.columns:
            want.extend([x, y])
    if len(want) != 42:
        raise ValueError(f"Expected 42 coord features, found {len(want)}.")
    return want

def load_xy(csv_path):
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False)]
    if "letter" not in df.columns:
        raise ValueError("'letter' column not found.")
    feat_cols = _select_feature_columns(df)
    df = df.dropna(subset=feat_cols + ["letter"]).reset_index(drop=True)
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df["letter"].astype(str).to_numpy()
    return X, y, feat_cols

def main():
    os.makedirs("models", exist_ok=True)
    X, y, feat_cols = load_xy(RAW)
    print(f"Loaded {X.shape[0]} rows, {X.shape[1]} features.")
    print(f"First 6 feature cols: {feat_cols[:6]} ...")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = Pipeline([
        ("norm", FunctionTransformer(row_box_normalize, validate=False)),
        ("scaler", StandardScaler(with_mean=True)),
        ("clf", LogisticRegression(max_iter=600))
    ])
    model.fit(Xtr, ytr)

    yhat = model.predict(Xte)
    print("Accuracy:", accuracy_score(yte, yhat))
    print(classification_report(yte, yhat, zero_division=0))

    with open(OUT, "wb") as f:
        pickle.dump(model, f)
    print("Saved ->", OUT)

if __name__ == "__main__":
    main()
