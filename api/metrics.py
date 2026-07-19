# filename: api/metrics.py
# purpose:  Prometheus metric objects — instantiated once at import, imported everywhere
# version:  1.0

from prometheus_client import Counter, Gauge, Histogram

# ── Prediction counters ───────────────────────────────────────────────────────

aerial_predictions_total = Counter(
    "aerial_predictions_total",
    "Total predictions made",
    ["predicted_class", "model_version"],
)

aerial_drone_alerts_total = Counter(
    "aerial_drone_alerts_total",
    "High-confidence drone detections that triggered is_alert=True (confidence > 0.9)",
)

aerial_image_uploads_total = Counter(
    "aerial_image_uploads_total",
    "Total images submitted via /predict or /batch_predict",
)

aerial_cache_hits_total = Counter(
    "aerial_cache_hits_total",
    "Redis cache lookup results by outcome",
    ["result"],  # "hit" | "miss" | "unavailable"
)

# ── Latency histogram ─────────────────────────────────────────────────────────

aerial_inference_duration_seconds = Histogram(
    "aerial_inference_duration_seconds",
    "End-to-end prediction latency (preprocessing → embedding → head → response)",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

# ── Score histograms ──────────────────────────────────────────────────────────

aerial_prediction_confidence = Histogram(
    "aerial_prediction_confidence",
    "Distribution of model sigmoid confidence scores across all predictions",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
)

# ── Gauges ────────────────────────────────────────────────────────────────────

aerial_data_drift_score = Gauge(
    "aerial_data_drift_score",
    "Mean KS statistic across 50 PCA components (0=no drift, 1=full drift). "
    "Updated every 20 new predictions via scipy.ks_2samp on PCA-projected embeddings.",
)

# Triggers APIDown CRITICAL alert when up{job='fastapi'} == 0 for >1 min
aerial_model_load_time_seconds = Gauge(
    "aerial_model_load_time_seconds",
    "Seconds taken to load and split the production model at API startup",
)
