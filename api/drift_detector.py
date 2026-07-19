# filename: api/drift_detector.py
# purpose:  Two-tier embedding drift detection — scipy KS per N predictions, Evidently on-demand
# version:  1.0

# stdlib
import logging
from collections import deque
from datetime import datetime, timezone

# third-party
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DRIFT_UPDATE_INTERVAL  = 20    # recompute KS score after every N new embeddings
DRIFT_WARNING_THRESHOLD = 0.3  # mean KS > this → "warning"; matches alert_rules.yml
PCA_N_COMPONENTS       = 50    # retain top 50 components of 1280-dim embeddings
MIN_BUFFER_FOR_DRIFT   = 50    # need at least 50 current samples to compute KS


class DriftDetector:
    """Embedding distribution monitor for the production FastAPI inference server.

    Two-tier design:
    ─ Fast tier (per DRIFT_UPDATE_INTERVAL requests): scipy KS statistic on PCA-projected
      embeddings → updates Prometheus gauge aerial_data_drift_score. ~2ms per update.
    ─ Slow tier (on /drift/report request): Evidently DataDriftReport on all 50 PCA
      components. Expensive — runs once per API call, not per prediction.

    PCA note: reference is (200, 1280). Fitting PCA(n_components=50) on 200 samples < 1280
    dims means the covariance matrix is rank ≤ 199. svd_solver='randomized' estimates top-50
    components reliably without attempting to resolve all 1280 — avoids full SVD instability.
    We capture ~95% of variance with 50 components; remaining components are noise.
    """

    def __init__(
        self,
        reference_embeddings_path: str,
        reference_labels_path: str,
        max_buffer: int = 200,
    ) -> None:
        ref = np.load(reference_embeddings_path).astype(np.float32)   # (200, 1280)
        self.reference_labels = np.load(reference_labels_path)        # (200,) int

        n_components = min(PCA_N_COMPONENTS, ref.shape[1], ref.shape[0] - 1)
        self.pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
        self.ref_projected = self.pca.fit_transform(ref)  # (200, 50)

        self.current_buffer: deque = deque(maxlen=max_buffer)
        self.label_buffer:   deque = deque(maxlen=max_buffer)

        self._drift_score:  float = 0.0
        self._drift_status: str   = "insufficient_data"
        self._n_since_update: int = 0

        logger.info(
            "DriftDetector: reference shape=%s, PCA n_components=%d, "
            "explained_var=%.3f",
            ref.shape,
            n_components,
            self.pca.explained_variance_ratio_.sum(),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def add_embedding(self, embedding: np.ndarray, label: int) -> None:
        """Add embedding to rolling buffer. Recomputes KS score every DRIFT_UPDATE_INTERVAL."""
        self.current_buffer.append(embedding.astype(np.float32))
        self.label_buffer.append(label)
        self._n_since_update += 1

        if (
            self._n_since_update >= DRIFT_UPDATE_INTERVAL
            and len(self.current_buffer) >= MIN_BUFFER_FOR_DRIFT
        ):
            self._update_drift_score()
            self._n_since_update = 0

    @property
    def drift_score(self) -> float:
        return self._drift_score

    @property
    def drift_status(self) -> str:
        if len(self.current_buffer) < MIN_BUFFER_FOR_DRIFT:
            return "insufficient_data"
        return self._drift_status

    def compute_full_report(self) -> dict:
        """Evidently DataDriftReport across all 50 PCA components.

        Called only by /drift/report — expensive, never per-prediction.
        Returns a dict safe to JSON-serialise and return directly from FastAPI.
        """
        n_current = len(self.current_buffer)
        if n_current < MIN_BUFFER_FOR_DRIFT:
            return {
                "status":    "insufficient_data",
                "n_current": n_current,
                "required":  MIN_BUFFER_FOR_DRIFT,
            }

        try:
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset

            n_comp    = self.ref_projected.shape[1]
            col_names = [f"pc_{i}" for i in range(n_comp)]

            ref_df = pd.DataFrame(self.ref_projected, columns=col_names)
            cur_df = pd.DataFrame(
                self.pca.transform(np.array(self.current_buffer)),
                columns=col_names,
            )

            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=ref_df, current_data=cur_df)
            result = report.as_dict()
            drift_result = result["metrics"][0]["result"]

            return {
                "status":              "ok",
                "drift_score":         round(self._drift_score, 4),
                "drift_status":        self._drift_status,
                "n_drifted_columns":   drift_result.get("number_of_drifted_columns", 0),
                "n_total_columns":     n_comp,
                "share_drifted":       round(drift_result.get("share_of_drifted_columns", 0.0), 4),
                "n_current":           n_current,
                "n_reference":         int(self.ref_projected.shape[0]),
                "generated_at":        datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error("Evidently drift report failed: %s", e)
            return {
                "status":      "error",
                "error":       str(e),
                "drift_score": round(self._drift_score, 4),
                "n_current":   n_current,
            }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _update_drift_score(self) -> None:
        """Compute mean KS statistic across PCA components. Updates Prometheus gauge."""
        from api.metrics import aerial_data_drift_score

        cur_proj = self.pca.transform(np.array(self.current_buffer))  # (N, 50)

        ks_scores = [
            ks_2samp(self.ref_projected[:, i], cur_proj[:, i]).statistic
            for i in range(self.ref_projected.shape[1])
        ]
        score = float(np.mean(ks_scores))

        self._drift_score  = score
        self._drift_status = "warning" if score > DRIFT_WARNING_THRESHOLD else "normal"
        aerial_data_drift_score.set(score)

        logger.info("Drift score updated: %.4f (%s)", score, self._drift_status)
