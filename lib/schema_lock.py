import json
import pandas as pd

def load_schema(path="config/schema.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def validate_df(df: pd.DataFrame, schema: dict, *, for_training=False) -> None:
    feats = schema["features"]
    needed = feats + ([schema["target"]] if for_training else [])
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    for col, allowed in schema.get("categorical", {}).items():
        if col in df.columns:
            bad = df[col].dropna().map(lambda x: x not in allowed)
            if bad.any():
                bad_values = df.loc[bad, col].unique().tolist()
                raise ValueError(f"{col} contains invalid values: {bad_values}; allowed={allowed}")
    for col, rules in schema.get("numeric", {}).items():
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if not rules.get("allow_null", False) and s.isna().any():
                raise ValueError(f"{col} has nulls but allow_null=false")
            if "min" in rules and (s.dropna() < rules["min"]).any():
                raise ValueError(f"{col} has values < {rules['min']}")
            if "max" in rules and (s.dropna() > rules["max"]).any():
                raise ValueError(f"{col} has values > {rules['max']}")
