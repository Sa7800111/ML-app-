import pandas as pd
df = pd.read_csv("data/training/test.csv")
feats = ["Pclass","Sex","Age","Fare","Embarked"]
missing = [c for c in feats if c not in df.columns]
if missing: raise SystemExit(f"test.csv missing {missing}")
df[feats].to_csv("data/test_features.csv", index=False)
print("Wrote data/test_features.csv")
