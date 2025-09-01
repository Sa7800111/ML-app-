CREATE TABLE IF NOT EXISTS ingestion_stats(
  id BIGSERIAL PRIMARY KEY,
  filename TEXT NOT NULL,
  ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  n_rows INT NOT NULL,
  n_valid_rows INT NOT NULL,
  n_invalid_rows INT NOT NULL,
  severity TEXT NOT NULL CHECK (severity IN ('low','medium','high')),
  report_path TEXT
);
CREATE TABLE IF NOT EXISTS data_quality_issues(
  id BIGSERIAL PRIMARY KEY,
  filename TEXT NOT NULL,
  issue_type TEXT NOT NULL,
  count INT NOT NULL,
  detected_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS prediction_file_log(
  id BIGSERIAL PRIMARY KEY,
  filename TEXT UNIQUE NOT NULL,
  processed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
