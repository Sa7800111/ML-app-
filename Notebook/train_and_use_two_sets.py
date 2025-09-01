#!/usr/bin/env python3
import argparse, json, joblib, pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

FEATURES = ["Pclass","Sex","Age","Fare","Embarked"]
ID_COL = "PassengerId"
TARGET = "Survived"

def build_pipeline():
    cat = ["Pclass","Sex","Embarked"]
    num = ["Age","Fare"]
    pre = ColumnTransformer([        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),                           ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),                           ("scaler", StandardScaler())]), num)    ])
    return Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=1000))])

def ensure_cols(df, needed, name):
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"{name} is missing columns: {missing}")

def main(train_csv, other_csv, out_dir, preds_out):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(train_csv)
    ensure_cols(train, FEATURES + [TARGET], "train_csv")
    for c in ["Pclass","Sex","Embarked"]: train[c] = train[c].astype("category")
    Xtr = train[FEATURES]; ytr = train[TARGET].astype(int)

    pipe = build_pipeline().fit(Xtr, ytr)
    joblib.dump(pipe, out / "model.joblib")

    other = pd.read_csv(other_csv)
    ensure_cols(other, FEATURES, "other_csv")
    for c in ["Pclass","Sex","Embarked"]: other[c] = other[c].astype("category")

    Xo = other[FEATURES]
    yhat = pipe.predict(Xo); ypr = pipe.predict_proba(Xo)[:, 1]

    results = {}
    if TARGET in other.columns:
        yo = other[TARGET].astype(int)
        results["mode"] = "evaluate"
        results["accuracy"] = float(accuracy_score(yo, yhat))
        try:
            results["roc_auc"] = float(roc_auc_score(yo, ypr))
        except Exception:
            results["roc_auc"] = None
        (out / "metrics_other.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print("✅ Evaluated on other_csv ->", out / "metrics_other.json")
    else:
        results["mode"] = "predict"
        pred_df = other.copy()
        pred_df["pred"] = yhat; pred_df["proba_1"] = ypr
        cols = ([ID_COL] if ID_COL in pred_df.columns else []) + FEATURES + ["pred","proba_1"]
        pred_df[cols].to_csv(preds_out, index=False)
        print("✅ Wrote predictions ->", preds_out)

    (out / "training_summary.json").write_text(json.dumps({"features": FEATURES, "n_train": int(len(Xtr))}, indent=2), encoding="utf-8")
    print("✅ Saved model ->", out / "model.joblib")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--other_csv", required=True)
    ap.add_argument("--out_dir", default="models")
    ap.add_argument("--preds_out", default="preds_other.csv")
    args = ap.parse_args()
    main(args.train_csv, args.other_csv, args.out_dir, args.preds_out)
