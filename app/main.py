from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os, json, sqlite3, joblib, pandas as pd
from dotenv import load_dotenv

load_dotenv()  # read .env

FEATURES = ["Pclass","Sex","Age","Fare","Embarked"]
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
DB_PATH = os.getenv("DB_PATH", "predictions.sqlite")  # if using SQLite locally

app = FastAPI(title="Titanic Model API")
_model = None

class PredictRequest(BaseModel):
    records: List[Dict[str, Any]]
    source: str = "webapp"

class PredictResponse(BaseModel):
    predictions: List[int]
    probabilities: Optional[List[float]] = None

def get_db():
    # Local SQLite quick store; for Postgres move storage there if desired
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""        CREATE TABLE IF NOT EXISTS predictions (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts TEXT DEFAULT CURRENT_TIMESTAMP,
          source TEXT,
          features TEXT,
          prediction INTEGER,
          proba REAL
        )
    """)
    return conn

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Model file not found: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model

@app.get("/health")
def health():
    _ = get_model(); _ = get_db()
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    df = pd.DataFrame(req.records)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail={"error": f"Missing columns: {missing}"})
    X = df[FEATURES]
    model = get_model()
    preds = model.predict(X).tolist()
    probs = model.predict_proba(X)[:,1].tolist()

    conn = get_db(); cur = conn.cursor()
    for f, p, pr in zip(df.to_dict("records"), preds, probs):
        cur.execute("INSERT INTO predictions(source,features,prediction,proba) VALUES(?,?,?,?)",
                    (req.source, json.dumps(f), int(p), float(pr)))
    conn.commit(); conn.close()
    return {"predictions": preds, "probabilities": probs}

@app.get("/past-predictions")
def past_predictions(limit: int = 50, source: str = "all"):
    conn = get_db(); cur = conn.cursor()
    if source in ("webapp","scheduled"):
        cur.execute("SELECT ts, source, features, prediction, proba FROM predictions WHERE source=? ORDER BY id DESC LIMIT ?",
                    (source, limit))
    else:
        cur.execute("SELECT ts, source, features, prediction, proba FROM predictions ORDER BY id DESC LIMIT ?",
                    (limit,))
    rows = cur.fetchall(); conn.close()
    items = [{"ts":r[0], "source":r[1], "features":json.loads(r[2]), "prediction":r[3], "proba":r[4]} for r in rows]
    return {"items": items}
