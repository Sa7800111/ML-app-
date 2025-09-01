import json, joblib, argparse, pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from lib.schema_lock import load_schema, validate_df

SCHEMA_PATH = "config/schema.json"

def main(train_csv, out_dir):
    schema = load_schema(SCHEMA_PATH)
    ID, Y = schema["id"], schema["target"]
    FEATS = schema["features"]
    CAT = list(schema["categorical"].keys())
    NUM = list(schema["numeric"].keys())

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(train_csv)
    validate_df(df, schema, for_training=True)

    for c in CAT: df[c] = df[c].astype("category")
    X = df[FEATS]; y = df[Y].astype(int)

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pre = ColumnTransformer([        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),                           ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT),        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),                           ("scaler", StandardScaler())]), NUM)    ])
    pipe = Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=1000))]).fit(Xtr, ytr)

    yhat = pipe.predict(Xva); ypr = pipe.predict_proba(Xva)[:,1]
    metrics = {"accuracy": float(accuracy_score(yva, yhat)),               "roc_auc": float(roc_auc_score(yva, ypr)),               "features": FEATS, "categorical": CAT, "numeric": NUM}

    joblib.dump(pipe, out / "model.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "schema_version.txt").write_text(schema["version"], encoding="utf-8")
    print("Saved:", out / "model.joblib", "|", metrics)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="data/training/train.csv")
    ap.add_argument("--out_dir", default="models")
    args = ap.parse_args()
    main(args.train_csv, args.out_dir)
