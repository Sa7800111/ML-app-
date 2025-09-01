#!/usr/bin/env python3
import argparse, glob, os, random
from pathlib import Path
import pandas as pd

def drop_required_col(df):
    return df.drop(columns=["Age"]) if "Age" in df.columns else df

def unknown_embarked(df):
    if "Embarked" in df.columns and len(df) > 0:
        idx = df.sample(max(1, len(df)//20)).index
        df.loc[idx, "Embarked"] = "X"
    return df

def age_as_text(df):
    if "Age" in df.columns and len(df) > 0:
        idx = df.sample(max(1, len(df)//30)).index
        df.loc[idx, "Age"] = "child"
    return df

def negative_age(df):
    if "Age" in df.columns and len(df) > 0:
        idx = df.sample(max(1, len(df)//25)).index
        df.loc[idx, "Age"] = -12
    return df

def missing_fare(df):
    if "Fare" in df.columns and len(df) > 0:
        idx = df.sample(max(1, len(df)//20)).index
        df.loc[idx, "Fare"] = pd.NA
    return df

def outlier_fare(df):
    if "Fare" in df.columns and len(df) > 0:
        idx = df.sample(max(1, len(df)//40)).index
        df.loc[idx, "Fare"] = 9999.99
    return df

CORRUPTIONS = [drop_required_col, unknown_embarked, age_as_text, negative_age, missing_fare, outlier_fare]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="data/raw-data")
    ap.add_argument("--ratio", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.indir, "*.csv")))
    if not files:
        raise SystemExit(f"No CSVs in {a.indir}. Run split_dataset.py first.")

    random.seed(a.seed)
    n_corrupt = max(1, int(len(files)*a.ratio))
    for path in random.sample(files, n_corrupt):
        df = pd.read_csv(path); bad = df.copy()
        import random as _r
        for f in _r.sample(CORRUPTIONS, k=_r.randint(2,4)):
            bad = f(bad)
        p = Path(path); out = p.with_name(p.stem + "_err" + p.suffix)
        bad.to_csv(out, index=False)
        print("Wrote:", out)
