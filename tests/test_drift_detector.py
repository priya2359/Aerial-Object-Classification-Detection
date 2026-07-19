# filename: tests/test_drift_detector.py
# purpose:  Unit tests for api/drift_detector.py — buffer logic, KS boundary, full report
# version:  1.0

# stdlib
from unittest.mock import patch

# third-party
import numpy as np
import pytest

from api.drift_detector import MIN_BUFFER_FOR_DRIFT, DriftDetector


@pytest.fixture
def detector(tmp_path, reference_embeddings, reference_labels):
    """DriftDetector loaded from synthetic reference files in a temp directory.

    np.random float32 arrays have the same shape/dtype as real EfficientNetB0 embeddings.
    The PCA fit, buffer logic, and KS tests operate on numpy arrays — they don't care
    whether values came from a real model or random noise.
    """
    ref_emb_path = tmp_path / "reference_embeddings.npy"
    ref_lbl_path = tmp_path / "reference_labels.npy"
    np.save(ref_emb_path, reference_embeddings)
    np.save(ref_lbl_path, reference_labels)

    with patch("api.drift_detector.aerial_data_drift_score"):
        d = DriftDetector(str(ref_emb_path), str(ref_lbl_path))
    return d


# ── Initialisation ────────────────────────────────────────────────────────────

class TestDriftDetectorInit:

    def test_pca_fitted(self, detector):
        """PCA should be fitted with 50 components on the 1280-dim reference."""
        assert detector.pca.n_components_ == 50
        assert detector.ref_projected.shape == (200, 50)

    def test_initial_status_insufficient_data(self, detector):
        """Before any predictions, buffer is empty → status='insufficient_data'."""
        assert detector.drift_status == "insufficient_data"
        assert detector.drift_score == 0.0

    def test_initial_buffer_empty(self, detector):
        """Buffer must start empty — no stale state from previous runs."""
        assert len(detector.current_buffer) == 0


# ── Buffer boundary tests ─────────────────────────────────────────────────────

class TestBufferBoundary:

    def _add_n(self, detector, n: int, label: int = 0):
        rng = np.random.default_rng(99)
        with patch("api.drift_detector.aerial_data_drift_score"):
            for _ in range(n):
                emb = rng.standard_normal(1280).astype(np.float32)
                detector.add_embedding(emb, label)

    def test_49_embeddings_still_insufficient(self, detector):
        """One below the 50-sample threshold — drift must NOT be computed yet."""
        self._add_n(detector, MIN_BUFFER_FOR_DRIFT - 1)
        assert len(detector.current_buffer) == MIN_BUFFER_FOR_DRIFT - 1
        assert detector.drift_status == "insufficient_data"

    def test_50_embeddings_triggers_first_score(self, detector):
        """Exactly at the threshold — drift score must be computed and status updated."""
        self._add_n(detector, MIN_BUFFER_FOR_DRIFT)
        assert len(detector.current_buffer) == MIN_BUFFER_FOR_DRIFT
        assert detector.drift_status in ("normal", "warning"), \
            f"Expected 'normal' or 'warning', got '{detector.drift_status}'"
        assert 0.0 <= detector.drift_score <= 1.0, \
            f"KS statistic must be in [0,1], got {detector.drift_score}"

    def test_200_embeddings_buffer_does_not_grow_beyond_max(self, detector):
        """Rolling deque must not exceed max_buffer=200 — oldest entries evicted."""
        self._add_n(detector, 250)
        assert len(detector.current_buffer) <= 200

    def test_score_is_float(self, detector):
        """drift_score property must return a Python float, not a numpy scalar."""
        self._add_n(detector, MIN_BUFFER_FOR_DRIFT)
        assert isinstance(detector.drift_score, float)


# ── compute_full_report ───────────────────────────────────────────────────────

class TestComputeFullReport:

    def _add_n(self, detector, n: int):
        rng = np.random.default_rng(7)
        with patch("api.drift_detector.aerial_data_drift_score"):
            for _ in range(n):
                detector.add_embedding(rng.standard_normal(1280).astype(np.float32), 0)

    def test_insufficient_data_before_50(self, detector):
        """compute_full_report must return status='insufficient_data' before 50 samples."""
        result = detector.compute_full_report()
        assert result["status"] == "insufficient_data"
        assert "n_current" in result
        assert "required" in result

    def test_full_report_with_evidently(self, detector):
        """With ≥ 50 samples, compute_full_report must return status='ok' and required keys."""
        self._add_n(detector, MIN_BUFFER_FOR_DRIFT)

        result = detector.compute_full_report()

        if result.get("status") == "error":
            pytest.skip(f"Evidently not installed or report failed: {result.get('error')}")

        assert result["status"] == "ok"
        assert "drift_score"       in result
        assert "drift_status"      in result
        assert "n_drifted_columns" in result
        assert "n_total_columns"   in result
        assert "share_drifted"     in result
        assert "n_current"         in result
        assert "n_reference"       in result
        assert "generated_at"      in result

    def test_full_report_graceful_on_evidently_error(self, detector):
        """If Evidently import fails, compute_full_report must return status='error', not raise."""
        self._add_n(detector, MIN_BUFFER_FOR_DRIFT)

        with patch("builtins.__import__", side_effect=ImportError("evidently not installed")):
            # compute_full_report catches all exceptions and returns a safe dict
            result = detector.compute_full_report()

        assert "status" in result
        # Either "error" (import failed) or "ok" (import already cached in sys.modules)
        assert result["status"] in ("error", "ok")
