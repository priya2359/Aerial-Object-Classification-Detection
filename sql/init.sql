-- Aerial Detection System — PostgreSQL Schema
-- Run once on container startup via Docker volume mount

-- ─── Database for Airflow (separate from aerial_db) ──────────
-- Created by POSTGRES_DB env var; Airflow uses airflow_db
-- aerial_db is the main application database

-- ─── Table 1: image_registry ─────────────────────────────────
CREATE TABLE IF NOT EXISTS image_registry (
    id              SERIAL PRIMARY KEY,
    gcs_path        TEXT,
    filename        VARCHAR(255) NOT NULL,
    label           VARCHAR(10) NOT NULL CHECK (label IN ('bird', 'drone')),
    class_id        SMALLINT NOT NULL CHECK (class_id IN (0, 1)),
    split           VARCHAR(10) NOT NULL CHECK (split IN ('train', 'valid', 'test')),
    file_hash       VARCHAR(32) NOT NULL UNIQUE,
    image_width     INTEGER,
    image_height    INTEGER,
    ingested_at     TIMESTAMP DEFAULT NOW(),
    status          VARCHAR(20) DEFAULT 'raw' CHECK (status IN ('raw', 'processed', 'augmented')),
    is_augmented    BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_image_registry_label ON image_registry(label);
CREATE INDEX IF NOT EXISTS idx_image_registry_split ON image_registry(split);
CREATE INDEX IF NOT EXISTS idx_image_registry_hash  ON image_registry(file_hash);

-- ─── Table 2: prediction_logs ─────────────────────────────────
CREATE TABLE IF NOT EXISTS prediction_logs (
    id              SERIAL PRIMARY KEY,
    image_hash      VARCHAR(32),
    predicted_label VARCHAR(10) NOT NULL CHECK (predicted_label IN ('bird', 'drone')),
    confidence      FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    model_version   VARCHAR(50),
    inference_time  FLOAT,
    drift_score     FLOAT,
    created_at      TIMESTAMP DEFAULT NOW(),
    is_alert        BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_pred_logs_created  ON prediction_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_pred_logs_label    ON prediction_logs(predicted_label);
CREATE INDEX IF NOT EXISTS idx_pred_logs_alert    ON prediction_logs(is_alert);

-- ─── Table 3: model_registry_log ──────────────────────────────
CREATE TABLE IF NOT EXISTS model_registry_log (
    id              SERIAL PRIMARY KEY,
    model_name      VARCHAR(50) NOT NULL,
    model_version   VARCHAR(50) NOT NULL,
    accuracy        FLOAT,
    f1_score        FLOAT,
    precision_score FLOAT,
    recall_score    FLOAT,
    trained_at      TIMESTAMP DEFAULT NOW(),
    mlflow_run_id   VARCHAR(100),
    is_production   BOOLEAN DEFAULT FALSE
);

-- Only one model can be production at a time
CREATE UNIQUE INDEX IF NOT EXISTS idx_one_production_model
    ON model_registry_log(is_production)
    WHERE is_production = TRUE;

-- ─── Table 4: pipeline_run_logs ───────────────────────────────
CREATE TABLE IF NOT EXISTS pipeline_run_logs (
    id               SERIAL PRIMARY KEY,
    dag_id           VARCHAR(100),
    run_id           VARCHAR(200),
    status           VARCHAR(20) CHECK (status IN ('success', 'failed', 'running')),
    images_ingested  INTEGER DEFAULT 0,
    images_skipped   INTEGER DEFAULT 0,
    started_at       TIMESTAMP,
    completed_at     TIMESTAMP
);
