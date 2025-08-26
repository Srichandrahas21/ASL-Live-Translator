# src/train_words_rf.py
import os, pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

RAW = "data/raw/sign_language.csv"
OUT = "models/words_rf.pkl"

def main():
    os.makedirs("models", exist_ok=True)
    df = pd.read_csv(RAW)
    feat_cols = [c for c in df.columns if c != "label" and pd.api.types.is_numeric_dtype(df[c])]
    if "label" not in df.columns or not feat_cols:
        raise ValueError("sign_language.csv must have 'label' and numeric feature columns.")
    X = df[feat_cols].values
    y = df["label"].astype(str).values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=42)
    clf.fit(Xtr, ytr)

    yhat = clf.predict(Xte)
    print("Accuracy:", accuracy_score(yte, yhat))
    print(classification_report(yte, yhat, zero_division=0))

    with open(OUT, "wb") as f:
        pickle.dump(clf, f)
    print("Saved ->", OUT)

if __name__ == "__main__":
    main()
