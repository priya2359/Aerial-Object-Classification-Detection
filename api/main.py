# filename: api/main.py
# purpose:  FastAPI inference server — 7 endpoints, cache-first, background DB logging, drift
# version:  1.1

# stdlib
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# third-party
import numpy as np
import psycopg2
import psycopg2.pool
import redis as redis_lib
from fastapi import BackgroundTasks, FastAPI, File, Header, HTTPException, UploadFile
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.concurrency import run_in_threadpool
from starlette.responses import Response
from tensorflow import keras

# internal
from api.drift_detector import DriftDetector
from api.metrics import (
    aerial_cache_hits_total,
    aerial_drone_alerts_total,
    aerial_image_uploads_total,
    aerial_inference_duration_seconds,
    aerial_model_load_time_seconds,
    aerial_prediction_confidence,
    aerial_predictions_total,
)
from api.preprocessing import compute_image_hash, preprocess_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

# ── Config (all from environment — never hardcode) ────────────────────────────
MODELS_DIR           = Path(os.getenv("MODELS_DIR", "models"))
DATA_DIR             = Path(os.getenv("DATA_DIR", "data"))
MODEL_VERSION        = os.getenv("MODEL_VERSION", "v1")
# C5 fix: read DRONE_ALERT_THRESHOLD (matches .env) — previously read "CONFIDENCE_THRESHOLD"
# which never matched the env var and was always the hardcoded default
CONFIDENCE_THRESHOLD = float(os.getenv("DRONE_ALERT_THRESHOLD", "0.9"))
REDIS_HOST           = os.getenv("REDIS_HOST", "redis")
REDIS_PORT           = int(os.getenv("REDIS_PORT", "6379"))
POSTGRES_HOST        = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_USER        = os.getenv("POSTGRES_USER", "aerial")
POSTGRES_PASSWORD    = os.getenv("POSTGRES_PASSWORD", "aerial")
POSTGRES_DB          = os.getenv("POSTGRES_DB", "aerial_db")
# C7 fix: secret required in X-Reload-Secret header to call /model/reload
RELOAD_SECRET        = os.getenv("RELOAD_SECRET", "")
# C4 fix: enforce upload size limit (MAX_UPLOAD_SIZE_MB from .env, default 10 MB)
MAX_UPLOAD_BYTES     = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10")) * 1024 * 1024
BATCH_MAX_SIZE       = 20
CACHE_TTL            = 604800  # 7 days

# M5 fix: lock guards the 5-attribute model swap so no request sees a mixed v1/v2 state
_model_lock = threading.Lock()


class AppState:
    """Mutable application state — avoids true module-level globals."""
    embedding_model:   Optional[keras.Model] = None  # winner's GAP output (128 or 1280-dim)
    head_model:        Optional[keras.Model] = None  # winner's classifier head
    model_type:        Optional[str]         = None  # "efficientnet" | "custom_cnn"
    embedding_dim:     Optional[int]         = None  # 1280 for EffNet, 128 for CNN
    optimal_threshold: float                 = 0.5
    redis_client:      Optional[object]      = None
    redis_available:   bool                  = False  # explicit flag — no try/except on hot path
    db_pool:           Optional[object]      = None
    drift_detector:    Optional[DriftDetector] = None
    startup_time:      float                 = 0.0


state = AppState()


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_models() -> tuple:
    """Read production_model.txt → load .h5 → split at GAP layer by type, not name.

    GAP is found by isinstance check (robust regardless of Keras naming).
    Embedding dim is extracted from the layer's output shape — used to detect
    whether drift detection is compatible with the reference embeddings.

    Returns (embedding_model, head_model, model_type, optimal_threshold).
    """
    t0 = time.time()

    prod_txt = MODELS_DIR / "production_model.txt"
    if not prod_txt.exists():
        raise RuntimeError(f"production_model.txt not found: {prod_txt}")

    model_type = prod_txt.read_text().strip()
    model_path = MODELS_DIR / f"{model_type}_final.h5"
    if not model_path.exists():
        raise RuntimeError(f"Model file not found: {model_path}")

    full_model = keras.models.load_model(str(model_path))

    # Find GAP by type — avoids relying on the "gap" name string surviving serialisation
    gap_layer = next(
        (l for l in full_model.layers if isinstance(l, keras.layers.GlobalAveragePooling2D)),
        None,
    )
    if gap_layer is None:
        raise RuntimeError(
            f"No GlobalAveragePooling2D layer found in {model_path}. "
            "Model architecture must contain a GAP layer between base and head."
        )

    embedding_model = keras.Model(
        inputs=full_model.input,
        outputs=gap_layer.output,
        name="embedding_extractor",
    )
    embedding_dim = int(gap_layer.output_shape[-1])  # 1280 (EffNet) or 128 (CNN)

    # Build head model from all layers after the GAP layer
    gap_idx    = full_model.layers.index(gap_layer)
    head_input = keras.Input(shape=(embedding_dim,), name="embedding_input")
    x = head_input
    for layer in full_model.layers[gap_idx + 1:]:
        x = layer(x)
    head_model = keras.Model(inputs=head_input, outputs=x, name="classifier_head")

    # Load optimal threshold from winner's metrics JSON
    metrics_path = MODELS_DIR / f"metrics_{model_type}.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            m = json.load(f)
        optimal_threshold = float(m.get("optimal_threshold", {}).get("threshold", 0.5))
    else:
        optimal_threshold = 0.5
        logger.warning("metrics_%s.json not found — using threshold=0.5", model_type)

    load_time = time.time() - t0
    aerial_model_load_time_seconds.set(load_time)
    logger.info(
        "Model loaded: %s (embedding_dim=%d) in %.2fs (optimal_threshold=%.4f)",
        model_type, embedding_dim, load_time, optimal_threshold,
    )
    return embedding_model, head_model, model_type, embedding_dim, optimal_threshold


def _connect_redis() -> tuple:
    """Attempt Redis connection. Return (client, available) — never raises."""
    try:
        client = redis_lib.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=False,
            socket_connect_timeout=3,
        )
        client.ping()
        logger.info("Redis connected at %s:%d", REDIS_HOST, REDIS_PORT)
        return client, True
    except Exception as e:
        logger.warning("Redis unavailable (%s) — embedding cache disabled", e)
        return None, False


def _connect_postgres() -> Optional[psycopg2.pool.ThreadedConnectionPool]:
    """Create ThreadedConnectionPool. Return None on failure — never raises."""
    try:
        pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=5,
            host=POSTGRES_HOST,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            dbname=POSTGRES_DB,
        )
        logger.info("PostgreSQL pool created at %s/%s", POSTGRES_HOST, POSTGRES_DB)
        return pool
    except Exception as e:
        logger.error("PostgreSQL unavailable (%s) — prediction logging disabled", e)
        return None


def _ensure_schema(pool: psycopg2.pool.ThreadedConnectionPool) -> None:
    """Run sql/init.sql on startup — idempotent (all CREATE IF NOT EXISTS).

    Cloud managed databases (e.g. Render PostgreSQL) don't auto-execute volume-mounted
    init scripts, so schema creation must happen programmatically on first startup.
    """
    sql_path = Path("sql/init.sql")
    if not sql_path.exists():
        logger.warning("sql/init.sql not found — skipping schema initialisation")
        return
    try:
        conn = pool.getconn()
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql_path.read_text())
        pool.putconn(conn)
        logger.info("Database schema verified/initialised via sql/init.sql")
    except Exception as exc:
        logger.error("Schema initialisation failed: %s", exc)


# ── Application lifespan ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    state.startup_time = time.time()

    (
        state.embedding_model,
        state.head_model,
        state.model_type,
        state.embedding_dim,
        state.optimal_threshold,
    ) = _load_models()

    state.redis_client, state.redis_available = _connect_redis()
    state.db_pool = _connect_postgres()
    if state.db_pool:
        _ensure_schema(state.db_pool)

    # Drift detection: reference embeddings are always 1280-dim EfficientNetB0 space.
    # Only enabled when the winner's embedding_dim matches the reference.
    ref_emb = DATA_DIR / "eda" / "reference_embeddings.npy"
    ref_lbl = DATA_DIR / "eda" / "reference_labels.npy"
    if ref_emb.exists() and ref_lbl.exists():
        ref_dim = int(np.load(ref_emb).shape[-1])
        if ref_dim == state.embedding_dim:
            state.drift_detector = DriftDetector(str(ref_emb), str(ref_lbl))
            logger.info("DriftDetector enabled (embedding_dim=%d)", state.embedding_dim)
        else:
            logger.warning(
                "Drift detection disabled: model embedding_dim=%d but reference_dim=%d. "
                "Re-run 01_EDA.ipynb with %s model to regenerate reference embeddings.",
                state.embedding_dim, ref_dim, state.model_type,
            )
    else:
        logger.warning("reference_embeddings.npy not found — drift detection disabled")

    yield

    if state.db_pool:
        state.db_pool.closeall()
    logger.info("FastAPI shutdown complete")


app = FastAPI(
    title="Aerial Object Classification API",
    description="Bird vs Drone binary classification with cache-first inference and drift monitoring",
    version="1.0",
    lifespan=lifespan,
)


# ── Inference helpers ─────────────────────────────────────────────────────────

def _redis_key(image_hash: str) -> str:
    return f"aerial:features:{MODEL_VERSION}:{image_hash}"


def _get_embedding_cached(
    image_arr: np.ndarray,
    image_hash: str,
) -> tuple:
    """Cache-first embedding extraction. Returns (embedding_1d, is_cached)."""
    if state.redis_available:
        raw = state.redis_client.get(_redis_key(image_hash))
        if raw is not None:
            aerial_cache_hits_total.labels(result="hit").inc()
            return np.frombuffer(raw, dtype=np.float32), True

    # Cache miss (or Redis unavailable) — run embedding model
    # model(x, training=False) disables Dropout and uses BN moving statistics
    embedding = state.embedding_model(image_arr, training=False).numpy().ravel()

    if state.redis_available:
        state.redis_client.setex(
            _redis_key(image_hash),
            CACHE_TTL,
            embedding.astype(np.float32).tobytes(),
        )
        aerial_cache_hits_total.labels(result="miss").inc()
    else:
        aerial_cache_hits_total.labels(result="unavailable").inc()

    return embedding, False


def _classify(confidence: float) -> tuple:
    """Apply two-threshold logic. Returns (label, is_alert).

    Two thresholds serve different purposes:
    ─ optimal_threshold (Youden's J): best label assignment — maximises TPR-FPR
    ─ CONFIDENCE_THRESHOLD (0.9): security alert — requires very high confidence
      before triggering a drone alert. FP rate at 0.9 << at optimal_threshold.
    """
    label    = "drone" if confidence >= state.optimal_threshold else "bird"
    is_alert = label == "drone" and confidence >= CONFIDENCE_THRESHOLD
    return label, is_alert


def _run_inference(image_bytes: bytes) -> dict:
    """End-to-end single-image inference. Called by /predict via run_in_threadpool.

    M5 fix: reads model_type under _model_lock so it's consistent with the models
    that will be used for embedding + classification in this same call.
    """
    t0 = time.time()

    aerial_image_uploads_total.inc()
    image_hash = compute_image_hash(image_bytes)
    with _model_lock:
        current_model_type = state.model_type
    image_arr  = preprocess_image(image_bytes, model_type=current_model_type)

    embedding, cached = _get_embedding_cached(image_arr, image_hash)

    # Run winner's head in inference mode (training=False disables Dropout)
    confidence = float(
        state.head_model(
            embedding.reshape(1, -1), training=False
        ).numpy().ravel()[0]
    )

    label, is_alert = _classify(confidence)

    if is_alert:
        aerial_drone_alerts_total.inc()
    aerial_predictions_total.labels(predicted_class=label, model_version=MODEL_VERSION).inc()
    aerial_prediction_confidence.observe(confidence)

    inference_ms = round((time.time() - t0) * 1000, 2)
    aerial_inference_duration_seconds.observe(inference_ms / 1000)

    drift_score  = 0.0
    drift_status = "disabled"
    if state.drift_detector is not None:
        state.drift_detector.add_embedding(embedding, 1 if label == "drone" else 0)
        drift_score  = state.drift_detector.drift_score
        drift_status = state.drift_detector.drift_status

    return {
        "predicted_label":   label,
        "confidence":        round(confidence, 4),
        "inference_time_ms": inference_ms,
        "is_alert":          is_alert,
        "cached":            cached,
        "model_type":        state.model_type,
        "model_version":     MODEL_VERSION,
        "drift_score":       round(drift_score, 4),
        "drift_status":      drift_status,
        "image_hash":        image_hash,
    }


# ── Background DB logging ─────────────────────────────────────────────────────

def _log_to_db(
    pool: Optional[psycopg2.pool.ThreadedConnectionPool],
    prediction: dict,
) -> None:
    """INSERT prediction into prediction_logs. Runs in FastAPI's thread pool — never blocks response.

    C2 fix: column names now match sql/init.sql exactly:
      - inference_time  (was: inference_time_ms — column doesn't exist)
      - drift_score     (was: omitted — column exists in schema)
      - removed: cached (column doesn't exist in prediction_logs schema)
    """
    if pool is None:
        return
    conn = None
    try:
        conn = pool.getconn()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO prediction_logs
                    (image_hash, predicted_label, confidence, model_version,
                     inference_time, drift_score, is_alert)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    prediction["image_hash"],
                    prediction["predicted_label"],
                    prediction["confidence"],
                    MODEL_VERSION,
                    prediction["inference_time_ms"],
                    prediction.get("drift_score", 0.0),
                    prediction["is_alert"],
                ),
            )
        conn.commit()
    except Exception as e:
        logger.error("DB log failed: %s", e)
        if conn:
            conn.rollback()
    finally:
        if conn:
            pool.putconn(conn)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":        "ok",
        "model_type":    state.model_type,
        "model_version": MODEL_VERSION,
        "embedding_dim": state.embedding_dim,
        "threshold":     state.optimal_threshold,
        "redis":         "connected" if state.redis_available else "unavailable",
        "postgres":      "connected" if state.db_pool is not None else "unavailable",
        "drift":         "enabled" if state.drift_detector is not None else "disabled",
        "uptime_s":      round(time.time() - state.startup_time, 1),
    }


@app.post("/predict")
async def predict(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Single-image prediction. Returns label, confidence, alert flag, and drift score."""
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    # C4 fix: enforce upload size limit before spending CPU on inference
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(image_bytes) // 1024}KB). "
                   f"Maximum is {MAX_UPLOAD_BYTES // 1024 // 1024}MB.",
        )
    # C6 fix: _run_inference is CPU-bound (PIL decode + Keras forward pass).
    # run_in_threadpool offloads it to FastAPI's thread pool so the event loop
    # is not blocked and concurrent requests are handled without queuing.
    result = await run_in_threadpool(_run_inference, image_bytes)
    background_tasks.add_task(_log_to_db, state.db_pool, result)
    return result


@app.post("/batch_predict")
async def batch_predict(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
):
    """Batch prediction. All cache misses are processed in a single embedding_model.predict()
    call — 3-5x faster than N sequential calls for large batches."""
    if len(files) > BATCH_MAX_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(files)} exceeds maximum {BATCH_MAX_SIZE}",
        )

    t0 = time.time()

    # Read all uploads first; enforce per-file size limit (C4 fix)
    images_bytes = []
    for f in files:
        data = await f.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File '{f.filename}' is too large ({len(data) // 1024}KB). "
                       f"Maximum is {MAX_UPLOAD_BYTES // 1024 // 1024}MB.",
            )
        images_bytes.append(data)
    hashes       = [compute_image_hash(b) for b in images_bytes]
    embeddings   = [None] * len(images_bytes)
    cache_flags  = [False] * len(images_bytes)
    miss_indices: list[int]        = []
    miss_arrays:  list[np.ndarray] = []

    # Phase 1: check Redis cache for each image
    for i, (b, h) in enumerate(zip(images_bytes, hashes)):
        aerial_image_uploads_total.inc()
        if state.redis_available:
            raw = state.redis_client.get(_redis_key(h))
            if raw is not None:
                embeddings[i] = np.frombuffer(raw, dtype=np.float32)
                cache_flags[i] = True
                aerial_cache_hits_total.labels(result="hit").inc()
                continue
        miss_indices.append(i)
        miss_arrays.append(preprocess_image(b, model_type=state.model_type))

    # Phase 2: single batched forward pass for all cache misses
    if miss_arrays:
        batch = np.concatenate(miss_arrays, axis=0)       # (N_miss, 224, 224, 3)
        batch_embs = state.embedding_model.predict(batch, verbose=0)  # (N_miss, D)
        for j, i in enumerate(miss_indices):
            emb = batch_embs[j]
            embeddings[i] = emb
            if state.redis_available:
                state.redis_client.setex(
                    _redis_key(hashes[i]),
                    CACHE_TTL,
                    emb.astype(np.float32).tobytes(),
                )
                aerial_cache_hits_total.labels(result="miss").inc()
            else:
                aerial_cache_hits_total.labels(result="unavailable").inc()

    # Phase 3: single batched head inference
    emb_batch   = np.stack(embeddings)                    # (N, D)
    confidences = state.head_model.predict(emb_batch, verbose=0).ravel()  # (N,)

    results = []
    for i, (conf, h, cached) in enumerate(zip(confidences, hashes, cache_flags)):
        confidence      = float(conf)
        label, is_alert = _classify(confidence)

        if is_alert:
            aerial_drone_alerts_total.inc()
        aerial_predictions_total.labels(predicted_class=label, model_version=MODEL_VERSION).inc()
        aerial_prediction_confidence.observe(confidence)

        # Update drift buffer for each image
        if state.drift_detector is not None:
            state.drift_detector.add_embedding(
                embeddings[i], 1 if label == "drone" else 0
            )

        pred = {
            "predicted_label": label,
            "confidence":      round(confidence, 4),
            "is_alert":        is_alert,
            "cached":          cached,
            "model_version":   MODEL_VERSION,
            "image_hash":      h,
        }
        results.append(pred)
        background_tasks.add_task(
            _log_to_db,
            state.db_pool,
            {**pred, "inference_time_ms": 0},
        )

    total_ms = round((time.time() - t0) * 1000, 2)
    aerial_inference_duration_seconds.observe(total_ms / 1000)

    return {
        "predictions":  results,
        "total_time_ms": total_ms,
        "n_cached":     sum(cache_flags),
        "n_total":      len(results),
    }


@app.get("/model/info")
def model_info():
    """Return the winner's metrics JSON. Useful for Streamlit analytics tab."""
    metrics_path = MODELS_DIR / f"metrics_{state.model_type}.json"
    if not metrics_path.exists():
        return {
            "model_type":    state.model_type,
            "model_version": MODEL_VERSION,
            "metrics":       "unavailable",
        }
    with open(metrics_path) as f:
        return json.load(f)


@app.get("/metrics")
def metrics():
    """Expose all 8 Prometheus metrics for scraping by prometheus.yml."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/model/reload")
def model_reload(x_reload_secret: str = Header(default="")):
    """Hot-swap the production model without container restart.

    C7 fix: requires X-Reload-Secret header matching RELOAD_SECRET env var.
    Without this, anyone who can reach port 8000 can force a model swap.

    M5 fix: _model_lock guards all 5 attribute assignments as a single unit.
    Without the lock, a concurrent request could read embedding_model (v2) with
    head_model (v1) — mismatched models produce silent wrong predictions.

    Loads into temp variables first — state is never None during the swap,
    so in-flight requests finish against the old model cleanly.
    """
    # C7: authenticate before loading (loading is expensive — don't do it for bad actors)
    if not RELOAD_SECRET or x_reload_secret != RELOAD_SECRET:
        raise HTTPException(status_code=403, detail="Invalid or missing X-Reload-Secret header.")

    new_emb, new_head, new_type, new_dim, new_thr = _load_models()

    # M5: atomic swap under lock — no request sees a mixed-version state
    with _model_lock:
        state.embedding_model   = new_emb
        state.head_model        = new_head
        state.model_type        = new_type
        state.embedding_dim     = new_dim
        state.optimal_threshold = new_thr

    logger.info("Hot-reload complete: %s (embedding_dim=%d, threshold=%.4f)", new_type, new_dim, new_thr)
    return {
        "status":        "reloaded",
        "model_type":    new_type,
        "embedding_dim": new_dim,
        "threshold":     new_thr,
    }


@app.get("/drift/report")
def drift_report():
    """Full Evidently DataDriftReport on PCA-projected embeddings.

    Expensive — runs the full Evidently report across 50 PCA components.
    Call on-demand only (Streamlit analytics tab, not per-prediction).
    """
    if state.drift_detector is None:
        return {
            "status": "disabled",
            "reason": (
                "Drift detection requires reference_embeddings.npy (1280-dim EfficientNetB0). "
                "Disabled because embedding dimensions do not match or reference file not found."
            ),
        }
    return state.drift_detector.compute_full_report()
