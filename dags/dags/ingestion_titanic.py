from __future__ import annotations
import os, glob, shutil, json, random
from datetime import datetime
import pandas as pd
from airflow.decorators import task, dag
from airflow.exceptions import AirflowSkipException
from sqlalchemy import create_engine, text
from utils.schema_rules import split_good_bad, FEATURES

DATA_ROOT = "/opt/airflow/data"
RAW = f"{DATA_ROOT}/raw-data"
GOOD = f"{DATA_ROOT}/good-data"
BAD  = f"{DATA_ROOT}/bad-data"
DB_URL = os.getenv("APP_DB_URL") or os.getenv("DATABASE_URL")
TEAMS_WEBHOOK = os.getenv("TEAMS_WEBHOOK_URL", "")

@dag(start_date=datetime(2025,1,1), schedule_interval="* * * * *", catchup=False, tags=["ingestion"])
def ingestion_titanic():
    @task
    def pick_file() -> str:
        files = sorted(glob.glob(f"{RAW}/*.csv"))
        if not files:
            raise AirflowSkipException("No raw files to ingest")
        return random.choice(files)

    @task
    def validate_and_split(path: str) -> dict:
        df = pd.read_csv(path)
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            good, bad = pd.DataFrame(columns=FEATURES), df
            issues = {"missing_columns": len(missing)}
            severity = "high"
        else:
            good, bad = split_good_bad(df[FEATURES].copy())
            issues = {"invalid_rows": int(len(bad)), "invalid_rate": float(len(bad)/max(1,len(df)))}
            severity = "low" if len(bad)==0 else ("medium" if issues["invalid_rate"]<0.5 else "high")
        base = os.path.basename(path)
        gpath = os.path.join(GOOD, base)
        bpath = os.path.join(BAD, base)
        if len(good) and len(bad):
            good.to_csv(gpath, index=False)
            bad.to_csv(bpath, index=False)
        elif len(good) and not len(bad):
            shutil.copy2(path, gpath)
        else:
            shutil.copy2(path, bpath)
        os.remove(path)
        return {
            "filename": base,
            "n_rows": int(len(df)),
            "n_valid_rows": int(len(good)),
            "n_invalid_rows": int(len(bad)),
            "severity": severity,
            "issues": issues,
            "report_path": None
        }

    @task
    def save_stats(payload: dict):
        eng = create_engine(DB_URL)
        with eng.begin() as c:
            c.execute(text("""                INSERT INTO ingestion_stats(filename, n_rows, n_valid_rows, n_invalid_rows, severity, report_path)                VALUES(:filename, :n_rows, :n_valid_rows, :n_invalid_rows, :severity, :report_path)            """), payload)
            for k,v in payload.get("issues", {}).items():
                c.execute(text("""                    INSERT INTO data_quality_issues(filename, issue_type, count)                    VALUES(:f, :t, :cnt)                """), {"f": payload["filename"], "t": k, "cnt": int(v) if isinstance(v, (int,float)) else 1})

    @task
    def send_alert(payload: dict):
        if not TEAMS_WEBHOOK:
            return "no_webhook"
        import requests
        title = f"Ingestion: {payload['filename']} — {payload['severity'].upper()}"
        summary = json.dumps(payload["issues"], indent=2)
        card = {
          "@type": "MessageCard", "@context": "http://schema.org/extensions",
          "themeColor": "FF0000" if payload["severity"]=="high" else ("FFA500" if payload["severity"]=="medium" else "00FF00"),
          "summary": title,
          "sections": [{"activityTitle": title, "text": f"```{summary}```"}]
        }
        requests.post(TEAMS_WEBHOOK, json=card, timeout=10).raise_for_status()
        return "sent"

    stats = validate_and_split(pick_file())
    save_stats(stats)
    send_alert(stats)

ingestion_titanic()
