#!/usr/bin/env python3
import argparse, math, os, uuid
from datetime import datetime
import pandas as pd

def main(input_csv, outdir, num_files, shuffle, seed):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(input_csv)
    if shuffle:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = max(1, num_files); chunk = math.ceil(len(df)/n)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    written = 0
    for i in range(n):
        s, e = i*chunk, min((i+1)*chunk, len(df))
        if s >= e: break
        part = df.iloc[s:e].copy()
        name = f"batch_{ts}_{i:03d}_{uuid.uuid4().hex[:8]}.csv"
        part.to_csv(os.path.join(outdir, name), index=False); written += 1
    print(f"Wrote {written} files to {os.path.abspath(outdir)} (~{chunk} rows each)")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/training/train.csv")
    ap.add_argument("--outdir", default="data/raw-data")
    ap.add_argument("--num-files", type=int, default=40)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(); main(a.input, a.outdir, a.num_files, a.shuffle, a.seed)
