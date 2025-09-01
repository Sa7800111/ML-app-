from __future__ import annotations
import os, glob
from datetime import datetime
import pandas as pd
from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException
from sqlalchemy import create_engine, text
import requests

DATA_ROOT = "/opt/airflow/data"
GOOD = f"{DATA_ROOT}/good-data"
DB_URL = os.getenv("APP_DB_URL") or os.getenv("DATABASE_URL")
MODEL_API = os.getenv("FASTAPI_PREDICT_URL")

@dag(start_date=datetime(2025,1,1), schedule_interval="*/2 * * * *", catchup=False, tags=["prediction"])
def prediction_titanic():
    @task
    def check_for_new() -> list[str]:
        eng = create_engine(DB_URL)
        files = sorted(glob.glob(f"{GOOD}/*.csv"))
        if not files:
            raise AirflowSkipException("No good files")
        with eng.begin() as c:
            done = {r[0] for r in c.execute(text("SELECT filename FROM prediction_file_log"))}
        pending = [f for f in files if os.path.basename(f) not in done]
        if not pending:
            raise AirflowSkipException("No new good files")
        return pending

    @task
    def make_predictions(paths: list[str]) -> int:
        total = 0
        for p in paths:
            df = pd.read_csv(p)
            if not len(df):
                continue
            payload = {"records": df.to_dict("records"), "source": "scheduled"}
            r = requests.post(MODEL_API, json=payload, timeout=60)
            r.raise_for_status()
            total += len(df)
        return total

    @task
    def log_done(paths: list[str]):
        eng = create_engine(DB_URL)
        with eng.begin() as c:
            for p in paths:
                c.execute(text("INSERT INTO prediction_file_log(filename) VALUES(:f) ON CONFLICT DO NOTHING"),
                          {"f": os.path.basename(p)})

    files = check_for_new()
    _ = make_predictions(files)
    log_done(files)

prediction_titanic()
