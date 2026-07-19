# filename: training/evaluate.py
# purpose:  Shared evaluation — confusion matrix, ROC, Youden threshold; used by Sections 3 and 4
# version:  1.1

# stdlib
import logging
import time
from pathlib import Path

# third-party
import matplotlib
matplotlib.use("Agg")   # M4 fix: non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    test_generator,
    model_name: str,
    output_dir: Path,
) -> dict:
    """Evaluate model on test_generator. Save confusion matrix + ROC curve PNGs.

    Returns a metrics dict with consistent keys — Section 4 uses the same structure
    to build the model comparison table (model_comparison.csv).

    Includes both default threshold (0.5) and optimal threshold via Youden's J statistic.
    FastAPI (Section 6) should use optimal_threshold for drone alert decisions, not 0.5.

    Args:
        model:          Trained Keras model.
        test_generator: flow_from_directory generator with shuffle=False.
        model_name:     "custom_cnn" or "efficientnet" — used in filenames and JSON.
        output_dir:     Directory for confusion matrix + ROC curve PNGs.

    Returns:
        metrics dict with keys: model_name, metrics_at_default_threshold,
        metrics_at_optimal_threshold, threshold_analysis, eval_time_s.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Reset ensures full-set evaluation from the first sample
    test_generator.reset()

    # Predict over all batches — may produce ceil(n/batch) * batch predictions
    y_pred_prob = model.predict(test_generator, verbose=1).ravel()
    y_pred_prob = y_pred_prob[: test_generator.n]   # trim any batch-padding duplicates

    y_true = test_generator.classes  # shape (n,), 0=bird 1=drone

    # ── Optimal threshold via Youden's J statistic ────────────────────────────
    # J = TPR - FPR; maximising J minimises both missed drones (FN) and false alarms (FP)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    optimal_idx       = int(np.argmax(tpr - fpr))
    optimal_threshold = float(thresholds[optimal_idx])

    # ── Metrics at threshold=0.5 (standard baseline) ──────────────────────────
    y_pred_05 = (y_pred_prob >= 0.5).astype(int)

    # ── Metrics at optimal threshold ──────────────────────────────────────────
    y_pred_opt = (y_pred_prob >= optimal_threshold).astype(int)

    roc_auc    = roc_auc_score(y_true, y_pred_prob)
    eval_time  = round(time.time() - t0, 3)

    def _metrics(y_pred: np.ndarray, thr: float) -> dict:
        return {
            "threshold": thr,
            "accuracy":  round(float(np.mean(y_pred == y_true)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1_score":  round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "auc":       round(float(roc_auc), 4),
        }

    opt = _metrics(y_pred_opt, optimal_threshold)
    metrics = {
        "model_name":   model_name,
        "architecture": (
            "EfficientNetB0 (two-stage fine-tuned)" if "effnet" in model_name
            else "Custom CNN (3-block Conv+GAP)"
        ),
        # C3 fix: flat "metrics" dict — consumed by build_comparison_report() and
        # Streamlit analytics.py. Keys match what those callers read (test_f1, test_auc, etc.)
        "metrics": {
            "test_accuracy":      opt["accuracy"],
            "test_precision":     opt["precision"],
            "test_recall":        opt["recall"],
            "test_f1":            opt["f1_score"],
            "test_auc":           opt["auc"],
            "test_loss":          None,
            # Aliases used by streamlit analytics.py
            "accuracy":           opt["accuracy"],
            "precision":          opt["precision"],
            "recall":             opt["recall"],
            "f1":                 opt["f1_score"],
            "auc_roc":            opt["auc"],
        },
        # Flat optimal_threshold — consumed by build_comparison_report() and analytics.py
        "optimal_threshold":             {"threshold": round(optimal_threshold, 4)},
        # Nested structures kept for completeness / future consumers
        "metrics_at_default_threshold":  _metrics(y_pred_05,  0.5),
        "metrics_at_optimal_threshold":  opt,
        "threshold_analysis": {
            "default_threshold":  0.5,
            "optimal_threshold":  round(optimal_threshold, 4),
            "optimal_criterion":  "youden_j",
            "optimal_tpr":        round(float(tpr[optimal_idx]), 4),
            "optimal_fpr":        round(float(fpr[optimal_idx]), 4),
        },
        "eval_time_s": eval_time,
    }

    logger.info(
        "%s | AUC=%.4f | F1@0.5=%.4f | F1@opt(%.3f)=%.4f",
        model_name, roc_auc,
        metrics["metrics_at_default_threshold"]["f1_score"],
        optimal_threshold,
        metrics["metrics_at_optimal_threshold"]["f1_score"],
    )

    # ── Save artefacts ─────────────────────────────────────────────────────────
    _save_confusion_matrix(y_true, y_pred_opt, model_name, optimal_threshold, output_dir)
    _save_roc_curve(fpr, tpr, roc_auc, optimal_idx, optimal_threshold, model_name, output_dir)

    return metrics


def _save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    threshold: float,
    output_dir: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    classes = ["bird (0)", "drone (1)"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    ax.set_title(f"Confusion Matrix — {model_name}\n(threshold = {threshold:.3f})")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()

    save_path = output_dir / f"confusion_matrix_{model_name}.png"
    plt.savefig(save_path, dpi=100)
    plt.close()  # M4 fix: never call plt.show() in production code — crashes headless servers
    logger.info("Saved: %s", save_path)


def _save_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    optimal_idx: int,
    optimal_threshold: float,
    model_name: str,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random classifier")

    # Mark the Youden-optimal operating point
    ax.scatter(
        fpr[optimal_idx], tpr[optimal_idx],
        color="crimson", s=90, zorder=5,
        label=f"Optimal (Youden J)  thr={optimal_threshold:.3f}",
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    save_path = output_dir / f"roc_curve_{model_name}.png"
    plt.savefig(save_path, dpi=100)
    plt.close()  # M4 fix: release memory; plt.show() removed (headless-server crash)
    logger.info("Saved: %s", save_path)
