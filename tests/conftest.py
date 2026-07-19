# filename: tests/conftest.py
# purpose:  Shared pytest fixtures — JPEG bytes, mock models, patched TestClient
# version:  1.0

# stdlib
import io
import time
from unittest.mock import MagicMock, patch

# third-party
import numpy as np
import pytest
from PIL import Image


# ── Shared image fixtures ─────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_jpeg_bytes() -> bytes:
    """224×224 RGB JPEG in memory — generated once per test session.

    PIL-generated JPEG with random pixels. Valid enough for preprocessing tests
    and POST /predict calls. scope="session" avoids regenerating ~50ms per test.
    """
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def sample_jpeg_file(sample_jpeg_bytes: bytes) -> tuple:
    """Tuple ready for requests-style multipart upload to POST /predict."""
    return ("file", ("test_image.jpg", sample_jpeg_bytes, "image/jpeg"))


# ── Synthetic embedding fixtures ──────────────────────────────────────────────

@pytest.fixture(scope="session")
def reference_embeddings() -> np.ndarray:
    """(200, 1280) float32 — same shape/dtype as real EfficientNetB0 GAP output."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((200, 1280)).astype(np.float32)


@pytest.fixture(scope="session")
def reference_labels() -> np.ndarray:
    """(200,) int32 — 100 bird (0) + 100 drone (1), matching EDA composition."""
    return np.array([0] * 100 + [1] * 100, dtype=np.int32)


# ── Mock model helpers ────────────────────────────────────────────────────────

def _make_mock_embedding_model() -> MagicMock:
    """Mock embedding_model that returns zeros((1, 1280)) on __call__ and .predict()."""
    mock = MagicMock()

    # model(arr, training=False).numpy().ravel() → zeros(1280)
    call_result = MagicMock()
    call_result.numpy.return_value = np.zeros((1, 1280), dtype=np.float32)
    mock.return_value = call_result

    # model.predict(batch, verbose=0) → zeros((N, 1280))
    mock.predict.return_value = np.zeros((1, 1280), dtype=np.float32)

    return mock


def _make_mock_head_model() -> MagicMock:
    """Mock head_model that returns 0.8 confidence on __call__ and .predict().

    With optimal_threshold=0.5: confidence=0.8 → predicted_label="drone"
    With CONFIDENCE_THRESHOLD=0.9: 0.8 < 0.9 → is_alert=False
    """
    mock = MagicMock()

    # model(emb, training=False).numpy().ravel()[0] → 0.8
    call_result = MagicMock()
    call_result.numpy.return_value = np.array([[0.8]], dtype=np.float32)
    mock.return_value = call_result

    # model.predict(batch, verbose=0).ravel() → [0.8] per image
    mock.predict.return_value = np.array([[0.8]], dtype=np.float32)

    return mock


# ── TestClient fixture ─────────────────────────────────────────────────────────

@pytest.fixture(scope="function")
def client():
    """FastAPI TestClient with all startup dependencies patched.

    Patches (all in api.main — the module where they are called, per mock.patch rules):
      api.main._load_models       → returns (mock_emb, mock_head, "efficientnet", 1280, 0.5)
      api.main._connect_redis     → returns (None, False) — cache disabled, no Redis needed
      api.main._connect_postgres  → returns None          — DB logging silently skipped

    Drift detection is disabled automatically: the lifespan checks ref_emb.exists() which
    returns False in the test environment, so DriftDetector is never instantiated.

    scope="function" ensures a fresh AppState per test — no cross-test state leakage.
    """
    from fastapi.testclient import TestClient
    from api.main import app

    mock_emb  = _make_mock_embedding_model()
    mock_head = _make_mock_head_model()

    def mock_load_models():
        return mock_emb, mock_head, "efficientnet", 1280, 0.5

    with patch("api.main._load_models",      side_effect=mock_load_models), \
         patch("api.main._connect_redis",     return_value=(None, False)), \
         patch("api.main._connect_postgres",  return_value=None):
        with TestClient(app) as c:
            yield c
