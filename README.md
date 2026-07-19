# Aerial Object Classification & Detection

Production-grade ML system for real-time aerial threat detection — **Bird vs Drone** classification with security alerting.  
Drone detections with confidence ≥ 90% trigger a real-time security alert.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                              │
│                    Streamlit  :8501                                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ HTTP (requests)
┌──────────────────────────────▼──────────────────────────────────────┐
│                      INFERENCE API                                   │
│                    FastAPI  :8000                                     │
│  POST /predict  →  preprocess → embedding → head → response         │
│                         ↕                                            │
│              Redis :6379          PostgreSQL :5432                   │
│          (embedding cache)      (prediction logs)                    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ metrics scrape
┌──────────────────────────────▼──────────────────────────────────────┐
│              MONITORING                                              │
│   Prometheus :9090  →  Grafana :3000  →  AlertManager :9093         │
└─────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────┐
│              DATA & EXPERIMENT TRACKING                             │
│   Airflow :8080  (ETL DAG)     MLflow :5000  (experiment tracking)  │
└─────────────────────────────────────────────────────────────────────┘
```

**11 Docker services** — all on `aerial-network`.

---

## Cloud Deploy (Render — one-click)

The fastest way to deploy the API + UI publicly:

```bash
# 1. Push this repo to GitHub (public or private)
git push origin main

# 2. Go to render.com → New → Blueprint → select your repo
# Render reads render.yaml and provisions FastAPI + Streamlit + PostgreSQL automatically.
# Free tier: services spin down after 15 min idle (~30s cold start on first request).
```

After deploy, visit `https://aerial-fastapi-<id>.onrender.com/health` once to warm it up,
then open the Streamlit URL shown in your Render dashboard.

---

## Quick Start (Docker Compose)

```bash
# 1. Clone and configure
git clone https://github.com/<your-username>/aerial-detection-production.git
cd aerial-detection-production
cp .env.example .env        # edit POSTGRES_PASSWORD and MODEL_VERSION if needed

# 2. Start all 11 services
docker compose up -d

# 3. Verify all services are healthy
docker compose ps
```

> **Note:** The FastAPI image now bundles a stub model (random ~0.5 confidence) for immediate
> startup. To use real Colab-trained weights, copy `models/*.h5` here then `docker compose build fastapi`.

Once all services show `running` or `healthy`:

| Service | URL | Credentials |
|---|---|---|
| Streamlit UI | http://localhost:8501 | — |
| FastAPI docs | http://localhost:8000/docs | — |
| Grafana | http://localhost:3000 | admin / admin123 |
| MLflow | http://localhost:5000 | — |
| Airflow | http://localhost:8080 | airflow / airflow |
| Prometheus | http://localhost:9090 | — |

---

## Smoke Test Checklist (run after `docker compose up`)

```bash
# All 11 services running
docker compose ps | grep -c "Up"   # expect: 11

# FastAPI healthy
curl -s http://localhost:8000/health | python -m json.tool

# Predict a test image
curl -s -X POST http://localhost:8000/predict \
  -F "file=@data/classification_dataset/test/bird/$(ls data/classification_dataset/test/bird | head -1)" \
  | python -m json.tool

# Prometheus scraping metrics
curl -s http://localhost:9090/api/v1/targets | python -m json.tool | grep "fastapi"

# Grafana datasource working
curl -s http://admin:admin123@localhost:3000/api/datasources | python -m json.tool
```

---

## Dataset Setup

### Classification Dataset (required)
```
data/classification_dataset/
├── train/
│   ├── bird/     (1,414 images)
│   └── drone/    (1,248 images)
├── valid/
│   ├── bird/     (217 images)
│   └── drone/    (225 images)
└── test/
    ├── bird/     (121 images)
    └── drone/    (94 images)
```

### Object Detection Dataset (optional — YOLOv8 only)
```
data/object_detection_Dataset/
├── train/ (2,662 images + .txt annotations)
├── valid/ (442 images)
└── test/  (215 images)
```

Annotation format: `<class_id> <x_center> <y_center> <width> <height>` (normalised 0–1).

---

## Model Training (Google Colab — GPU required)

> **Never train locally.** EfficientNetB0 requires GPU. CPU training = 4–8 hours/epoch.

### Step 1 — Upload Dataset to Google Drive
```
MyDrive/aerial_detection/classification_dataset/
```

### Step 2 — Run EDA Notebook
Open `notebooks/01_EDA.ipynb` in Colab. Run all cells. This produces:
- `data/eda/eda_stats.json` — augmentation config + class weights (consumed by training)
- `data/eda/reference_embeddings.npy` — 200-sample drift baseline

### Step 3 — Train Custom CNN
Open `notebooks/02_CustomCNN.ipynb`. Run all cells.  
Saves `models/custom_cnn_final.h5` and `models/metrics_cnn.json` to Drive.

### Step 4 — Train EfficientNetB0
Open `notebooks/03_TransferLearning.ipynb`. Run all cells.  
Two-stage training: frozen base (Stage 1) → unfreeze top 20 layers (Stage 2).  
Saves `models/efficientnet_final.h5`, `models/metrics_effnet.json`, and `models/production_model.txt`.

### Step 5 — Copy Models to Local `models/` Directory
```bash
# From Colab: download from Drive, then place locally:
models/
├── production_model.txt        # "efficientnet" or "custom_cnn"
├── efficientnet_final.h5
├── custom_cnn_final.h5
├── metrics_effnet.json
└── metrics_cnn.json
```

---

## Local Development (CPU only — no training)

```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# Run FastAPI locally (requires models/ directory populated)
uvicorn api.main:app --reload --port 8000

# Run Streamlit locally
FASTAPI_URL=http://localhost:8000 streamlit run streamlit_app/app.py
```

---

## Running Tests

```bash
# All tests (from project root with venv active)
pytest tests/ -v

# Individual modules
pytest tests/test_preprocessing.py -v
pytest tests/test_drift_detector.py -v
pytest tests/test_api.py -v
```

Tests use mocked models and synthetic embeddings — no GPU, no real `.h5` files, no Redis, no PostgreSQL required.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | API status, Redis/Postgres/Drift connectivity, uptime |
| `POST` | `/predict` | Single image → label, confidence, alert flag, drift score |
| `POST` | `/batch_predict` | Up to 20 images → batch results (single forward pass) |
| `GET` | `/model/info` | Winner model metrics (accuracy, F1, AUC, threshold) |
| `GET` | `/metrics` | Prometheus metrics (8 metrics, text/plain format) |
| `POST` | `/model/reload` | Hot-swap production model without container restart |
| `GET` | `/drift/report` | Evidently DataDriftReport on recent prediction embeddings |

### POST /predict — Response Schema
```json
{
  "predicted_label":   "drone",
  "confidence":        0.9412,
  "inference_time_ms": 45.2,
  "is_alert":          true,
  "cached":            false,
  "model_type":        "efficientnet",
  "model_version":     "v1",
  "drift_score":       0.0821,
  "drift_status":      "normal",
  "image_hash":        "a3f2b91cd7e80456..."
}
```

**Two-threshold design:**
- `predicted_label` uses `optimal_threshold` (Youden's J statistic — maximises TPR−FPR)
- `is_alert=True` requires **both** label="drone" AND confidence ≥ 0.90 (security rule — minimises false alarms)

---

## Streamlit UI

Three tabs at `http://localhost:8501`:

| Tab | Purpose |
|---|---|
| 🔍 Single Predict | Upload one image → prediction with alert banner (🔴 red = alert, 🟡 orange = drone/no alert, 🟢 green = bird) |
| 📦 Batch Predict | Upload up to 20 images → results table + Plotly chart + CSV download |
| 📊 Analytics | Model metrics, service health, drift gauge, session history, admin hot-reload |

Sidebar shows live API status (green/red dots per service) and last 5 session predictions.

---

## Monitoring

### Prometheus Metrics (8 total)
| Metric | Type | Alert Rule |
|---|---|---|
| `aerial_predictions_total` | Counter | HighDroneDetectionRate: >80% drone for 5 min → CRITICAL |
| `aerial_inference_duration_seconds` | Histogram | HighInferenceLatency: P95 > 1s for 3 min → WARNING |
| `aerial_prediction_confidence` | Histogram | LowModelConfidence: avg < 0.60 for 10 min → WARNING |
| `aerial_data_drift_score` | Gauge | DataDriftDetected: score > 0.30 for 5 min → WARNING |
| `aerial_drone_alerts_total` | Counter | — |
| `aerial_image_uploads_total` | Counter | — |
| `aerial_cache_hits_total` | Counter | — |
| `aerial_model_load_time_seconds` | Gauge | APIDown: up == 0 for 1 min → CRITICAL |

### Grafana Dashboard
Import at `http://localhost:3000`. The dashboard (`grafana/dashboards/`) has 5 rows:
1. Overview KPIs (total predictions, alert rate, cache hit rate)
2. Prediction analytics (label distribution, confidence histogram)
3. Latency (P50/P95/P99 inference duration)
4. Drift & Cache (drift score trend, cache hit/miss rate)
5. Infrastructure (CPU, memory, Redis/Postgres status)

---

## Project Structure

```
aerial-detection/
├── api/
│   ├── main.py               ← FastAPI app (7 endpoints)
│   ├── preprocessing.py      ← preprocess_image(), compute_image_hash()
│   ├── metrics.py            ← 8 Prometheus metric objects
│   ├── drift_detector.py     ← DriftDetector (scipy KS + Evidently)
│   └── Dockerfile
├── streamlit_app/
│   ├── app.py                ← Main entry point + sidebar
│   ├── components/
│   │   ├── predictor.py      ← Tab 1: single predict
│   │   ├── batch_predictor.py ← Tab 2: batch predict
│   │   └── analytics.py      ← Tab 3: analytics + admin
│   └── Dockerfile
├── training/
│   ├── data_loader.py        ← ImageDataGenerator + augmentation
│   ├── custom_cnn.py         ← Model A definition + train()
│   ├── transfer_learning.py  ← Model B EfficientNetB0 two-stage
│   └── evaluate.py           ← metrics, ROC, confusion matrix
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_CustomCNN.ipynb
│   ├── 03_TransferLearning.ipynb
│   └── 04_YOLOv8.ipynb       ← optional
├── dags/
│   └── aerial_ingestion_dag.py ← 5-task Airflow ETL
├── feature_store/
│   ├── redis_client.py       ← store/get float32 embeddings
│   └── precompute.py         ← bulk EfficientNetB0 embedding precompute
├── tests/
│   ├── conftest.py
│   ├── test_preprocessing.py
│   ├── test_drift_detector.py
│   └── test_api.py
├── prometheus/
│   ├── prometheus.yml
│   └── alert_rules.yml
├── grafana/
│   ├── datasources/prometheus.yml
│   └── dashboards/dashboard.yml
├── sql/init.sql              ← 4 tables: image_registry, prediction_logs, ...
├── models/                   ← .h5 files (from Colab, gitignored if large)
├── data/                     ← classification_dataset/, object_detection_Dataset/
├── docker-compose.yml        ← 11 services
├── .env.example
├── requirements.txt          ← local dev (CPU)
├── requirements-colab.txt    ← Colab-only packages
└── README.md
```

---

## Model Results

| Model | Test Accuracy | F1 Score | AUC-ROC | Inference Time | Parameters |
|---|---|---|---|---|---|
| Custom CNN (3-block) | _fill after Colab_ | _fill_ | _fill_ | _fill_ ms | ~500K |
| EfficientNetB0 (fine-tuned top-20) | _fill after Colab_ | _fill_ | _fill_ | _fill_ ms | ~5.3M |

> Fill from `models/metrics_cnn.json` and `models/metrics_effnet.json` after Colab training completes.

**Production model:** Written to `models/production_model.txt` — winner selected by F1 → AUC → EfficientNetB0 (tiebreaker).  
**Optimal threshold:** Youden's J statistic (argmax TPR−FPR on ROC curve).  
**Drone alert threshold:** 0.90 (hard business rule — minimises false alarm rate for security context).

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| GAP found by `isinstance()`, not by name | Layer names may not survive `.h5` serialisation |
| `redis_available` bool flag on AppState | Avoids try/except on every hot-path prediction |
| `ThreadedConnectionPool` for PostgreSQL | DB logging runs in FastAPI's thread pool — thread-safe |
| Two-tier drift detection | scipy KS per 20 predictions (fast) + Evidently on demand (accurate) |
| `PCA(n_components=50, svd_solver='randomized')` | 200 samples × 1280 dims is rank-deficient; randomized SVD is stable |
| Atomic model reload | New model loaded into locals first; state never set to `None` |
| `RANDOM_STATE = 42` everywhere | Reproducible splits, weight init, PCA |
| Training on Google Colab | EfficientNetB0 + augmentation requires GPU; local CPU = 4–8 h/epoch |
| `production_model.txt` instead of MLflow registry | File-based MLflow (Colab) doesn't support registry; deferred to Docker MLflow |
| Streamlit is a thin HTTP client | No sklearn/keras imports; all inference via FastAPI; monitoring via Prometheus |

---

## Tech Stack

Python 3.11 · TensorFlow 2.15 / Keras · EfficientNetB0 (ImageNet) · YOLOv8 (ultralytics)  
FastAPI · uvicorn · Pydantic · prometheus-client · Evidently · scikit-learn  
Streamlit · Plotly · requests  
MLflow 2.13 · PostgreSQL 15 · Redis 7 · Apache Airflow 2.9.1  
Docker Compose · Prometheus · Grafana · AlertManager
