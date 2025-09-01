1) Goal

Turn a simple Titanic model into a production-style ML app that anyone can use, monitor, and automate.

2) Data & Prediction

We predict Survived using 5 columns: Pclass, Sex, Age, Fare, Embarked.

Keeping features small lets us focus on operations, not fancy modeling.

3) Model design (why this way)

A single scikit-learn Pipeline bundles preprocessing (impute, one-hot, scale) + Logistic Regression.

One saved file (model.joblib) means training and serving do the exact same transforms—no mismatch bugs.

4) API (FastAPI)

The model is exposed via HTTP:

POST /predict: returns predictions (single or batch) and logs what was predicted.

GET /past-predictions: fetches history.

Centralizing the model behind an API lets Streamlit, schedulers, and anything else reuse it.

5) Database

Each prediction call stores: timestamp, source (webapp/scheduled), input features, prediction, probability.

DB choice is flexible via DATABASE_URL:

Start simple with SQLite locally.

Switch to Postgres by just changing the .env string (no code changes).

6) UI (Streamlit)

Two tabs:

Predict: form + CSV upload → calls the API and shows results.

Past predictions: reads history from the API.

This gives a non-technical user a friendly way to try the model and see past runs.

7) Data quality & ingestion (production mindset)

We “simulate real life” by dropping CSVs into raw-data.

An ingestion job validates rows (e.g., category out of set, negative age), then splits files into good-data and bad-data, and stores stats in the DB.

Why: catch bad data before it reaches the model, and keep metrics to monitor quality over time.

8) Scheduled predictions

A periodic job looks at good-data, sends rows to /predict, and marks files processed so runs are idempotent (no double counting).

If you don’t have Airflow yet, you can do the same with Windows Task Scheduler (every 2 min).

9) Monitoring (Grafana)

Dashboards query the DB to show:

Ingestion quality (invalid % over time, issue types).

Model behavior (rate of predictions, class mix, average probability).

With thresholds and alerts, you notice problems quickly (e.g., all rows invalid, model always predicting one class).

10) Configuration (.env)

We keep secrets and endpoints in a .env file:

DATABASE_URL (SQLite or your Postgres)

FASTAPI_PREDICT_URL etc.

This makes the same code run on your laptop or a server by changing just one line.

11) How the pieces talk (the flow)

Train once → write model.joblib.

API loads that artifact and serves /predict.

Streamlit or the scheduler sends records to /predict.

API returns results and logs them in the DB.

Ingestion feeds/validates raw files and yields good/bad data + quality stats.

Scheduler consumes good data → /predict.

Grafana reads the DB to visualize quality and model behavior.

12) Why this architecture works for the course

Shows end-to-end MLOps: serving, validation, scheduling, logging, monitoring.

Uses a simple, explainable model so the focus is on operationalization.

Is Windows-friendly (runs without Docker), and can later add Docker/Airflow/Postgres with the same code.
