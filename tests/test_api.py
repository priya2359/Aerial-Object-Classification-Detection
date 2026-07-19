# filename: tests/test_api.py
# purpose:  FastAPI endpoint tests via TestClient — HTTP codes, response schema, error handling
# version:  1.0
#
# All tests use the `client` fixture from conftest.py which patches:
#   api.main._load_models     → (mock_emb, mock_head, "efficientnet", 1280, 0.5)
#   api.main._connect_redis   → (None, False)   — cache disabled
#   api.main._connect_postgres → None           — logging silently skipped
#
# Mock confidence=0.8, optimal_threshold=0.5 → predicted_label="drone", is_alert=False
# (0.8 < CONFIDENCE_THRESHOLD=0.9)

# stdlib
import io

# third-party
import pytest
from PIL import Image
import numpy as np


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:

    def test_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_response_schema(self, client):
        data = client.get("/health").json()
        required = {"status", "model_type", "model_version", "redis", "postgres",
                    "drift", "uptime_s", "embedding_dim", "threshold"}
        assert required.issubset(data.keys()), \
            f"Missing keys: {required - set(data.keys())}"

    def test_status_ok(self, client):
        assert client.get("/health").json()["status"] == "ok"

    def test_model_type_matches_mock(self, client):
        assert client.get("/health").json()["model_type"] == "efficientnet"

    def test_redis_unavailable_when_mocked_false(self, client):
        assert client.get("/health").json()["redis"] == "unavailable"

    def test_uptime_is_positive(self, client):
        assert client.get("/health").json()["uptime_s"] >= 0


# ── /predict ──────────────────────────────────────────────────────────────────

class TestPredict:

    def test_valid_jpeg_returns_200(self, client, sample_jpeg_bytes):
        r = client.post("/predict", files={"file": ("img.jpg", sample_jpeg_bytes, "image/jpeg")})
        assert r.status_code == 200

    def test_response_schema(self, client, sample_jpeg_bytes):
        data = client.post(
            "/predict",
            files={"file": ("img.jpg", sample_jpeg_bytes, "image/jpeg")},
        ).json()
        required = {
            "predicted_label", "confidence", "inference_time_ms",
            "is_alert", "cached", "model_type", "model_version",
            "drift_score", "drift_status", "image_hash",
        }
        assert required.issubset(data.keys()), \
            f"Missing keys: {required - set(data.keys())}"

    def test_predicted_label_is_drone_at_08_confidence(self, client, sample_jpeg_bytes):
        """Mock returns confidence=0.8, threshold=0.5 → must predict 'drone'."""
        data = client.post(
            "/predict",
            files={"file": ("img.jpg", sample_jpeg_bytes, "image/jpeg")},
        ).json()
        assert data["predicted_label"] == "drone"
        assert abs(data["confidence"] - 0.8) < 0.01

    def test_is_alert_false_below_09_threshold(self, client, sample_jpeg_bytes):
        """Confidence=0.8 < CONFIDENCE_THRESHOLD=0.9 → is_alert must be False."""
        data = client.post(
            "/predict",
            files={"file": ("img.jpg", sample_jpeg_bytes, "image/jpeg")},
        ).json()
        assert data["is_alert"] is False

    def test_cached_false_when_redis_unavailable(self, client, sample_jpeg_bytes):
        """Redis is mocked as unavailable → cached must always be False."""
        data = client.post(
            "/predict",
            files={"file": ("img.jpg", sample_jpeg_bytes, "image/jpeg")},
        ).json()
        assert data["cached"] is False

    def test_image_hash_is_md5_hex(self, client, sample_jpeg_bytes):
        data = client.post(
            "/predict",
            files={"file": ("img.jpg", sample_jpeg_bytes, "image/jpeg")},
        ).json()
        h = data["image_hash"]
        assert len(h) == 32
        assert all(c in "0123456789abcdef" for c in h)

    def test_empty_file_returns_400(self, client):
        """Empty bytes must be rejected with 400 — not cause a 500."""
        r = client.post("/predict", files={"file": ("empty.jpg", b"", "image/jpeg")})
        assert r.status_code == 400

    def test_inference_time_ms_is_positive(self, client, sample_jpeg_bytes):
        data = client.post(
            "/predict",
            files={"file": ("img.jpg", sample_jpeg_bytes, "image/jpeg")},
        ).json()
        assert data["inference_time_ms"] >= 0


# ── /batch_predict ────────────────────────────────────────────────────────────

class TestBatchPredict:

    def _make_jpeg(self, seed: int = 0) -> bytes:
        rng = np.random.default_rng(seed)
        arr = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr, mode="RGB").save(buf, format="JPEG")
        return buf.getvalue()

    def test_two_images_returns_200(self, client):
        files = [
            ("files", (f"img_{i}.jpg", self._make_jpeg(i), "image/jpeg"))
            for i in range(2)
        ]
        r = client.post("/batch_predict", files=files)
        assert r.status_code == 200

    def test_response_schema(self, client):
        files = [("files", ("img.jpg", self._make_jpeg(), "image/jpeg"))]
        data = client.post("/batch_predict", files=files).json()
        assert "predictions" in data
        assert "total_time_ms" in data
        assert "n_cached" in data
        assert "n_total" in data

    def test_n_total_matches_uploaded(self, client):
        n = 3
        files = [
            ("files", (f"img_{i}.jpg", self._make_jpeg(i), "image/jpeg"))
            for i in range(n)
        ]
        data = client.post("/batch_predict", files=files).json()
        assert data["n_total"] == n
        assert len(data["predictions"]) == n

    def test_exceeding_20_files_returns_400(self, client):
        """BATCH_MAX_SIZE=20 enforced in api/main.py — exceeding must return 400."""
        files = [
            ("files", (f"img_{i}.jpg", self._make_jpeg(i), "image/jpeg"))
            for i in range(21)
        ]
        r = client.post("/batch_predict", files=files)
        assert r.status_code == 400

    def test_per_prediction_has_required_keys(self, client):
        files = [("files", ("img.jpg", self._make_jpeg(), "image/jpeg"))]
        pred = client.post("/batch_predict", files=files).json()["predictions"][0]
        assert "predicted_label" in pred
        assert "confidence" in pred
        assert "is_alert" in pred
        assert "cached" in pred


# ── /model/info ───────────────────────────────────────────────────────────────

class TestModelInfo:

    def test_returns_200(self, client):
        r = client.get("/model/info")
        assert r.status_code == 200

    def test_returns_json(self, client):
        data = client.get("/model/info").json()
        assert isinstance(data, dict)

    def test_contains_model_type_when_no_metrics_file(self, client):
        """When metrics JSON is absent, response still returns model_type."""
        data = client.get("/model/info").json()
        # Either a real metrics dict or the fallback schema
        assert "model_type" in data or "metrics" in data


# ── /metrics ──────────────────────────────────────────────────────────────────

class TestPrometheusMetrics:

    def test_returns_200(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200

    def test_content_type_is_prometheus(self, client):
        r = client.get("/metrics")
        assert "text/plain" in r.headers["content-type"]

    def test_contains_aerial_metric_names(self, client):
        body = client.get("/metrics").text
        assert "aerial_predictions_total" in body
        assert "aerial_inference_duration_seconds" in body


# ── /drift/report ─────────────────────────────────────────────────────────────

class TestDriftReport:

    def test_returns_200(self, client):
        r = client.get("/drift/report")
        assert r.status_code == 200

    def test_disabled_when_no_reference_embeddings(self, client):
        """Drift is disabled in test env (no reference_embeddings.npy) → status='disabled'."""
        data = client.get("/drift/report").json()
        # In test env, ref_emb.exists() is False → drift_detector is None
        assert data.get("status") == "disabled"
        assert "reason" in data


# ── /model/reload ─────────────────────────────────────────────────────────────

class TestModelReload:

    def test_reload_returns_200(self, client):
        r = client.post("/model/reload")
        assert r.status_code == 200

    def test_reload_response_schema(self, client):
        data = client.post("/model/reload").json()
        assert "status" in data
        assert data["status"] == "reloaded"
        assert "model_type" in data
        assert "threshold" in data

    def test_reload_preserves_model_type(self, client):
        """After reload, model_type should still be 'efficientnet' (same mock)."""
        data = client.post("/model/reload").json()
        assert data["model_type"] == "efficientnet"
