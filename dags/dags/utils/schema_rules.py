import pandas as pd

FEATURES = ["Pclass","Sex","Age","Fare","Embarked"]
CAT_SETS = {
    "Pclass": {1,2,3},
    "Sex": {"male","female"},
    "Embarked": {"C","Q","S"},
}

NUM_RULES = {
    "Age": {"min":0, "max":100, "allow_null": True},
    "Fare": {"min":0, "allow_null": True},
}

def row_valid(r: pd.Series) -> bool:
    for c, allowed in CAT_SETS.items():
        if c in r and pd.notna(r[c]) and r[c] not in allowed:
            return False
    for c, rules in NUM_RULES.items():
        if c not in r: return False
        v = r[c]
        if pd.isna(v):
            if not rules.get("allow_null", False):
                return False
            else:
                continue
        try:
            x = float(v)
        except Exception:
            return False
        if "min" in rules and x < rules["min"]:
            return False
        if "max" in rules and x > rules["max"]:
            return False
    return True

def split_good_bad(df: pd.DataFrame):
    mask = df.apply(row_valid, axis=1)
    return df[mask].copy(), df[~mask].copy()
