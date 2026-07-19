# filename: training/transfer_learning.py
# purpose:  EfficientNetB0 transfer learning — two-stage fine-tuning for bird/drone classification
# version:  1.0

# stdlib
import json
import logging
from pathlib import Path

# third-party
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers

logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
RANDOM_STATE  = 42
IMAGE_SIZE    = (224, 224)
BATCH_SIZE    = 32
EPOCHS_STAGE1 = 20
EPOCHS_STAGE2 = 10
LR_STAGE1     = 1e-3
LR_STAGE2     = 1e-5
FINE_TUNE_AT  = 20      # Unfreeze last 20 of EfficientNetB0's 237 layers (final MBConv blocks)
DROPOUT_RATE  = 0.3     # Lower than CNN's 0.5 — EfficientNetB0 base already regularized (MBConv + SE)
PATIENCE      = 5


def build_efficientnet_model(
    input_shape: tuple = (224, 224, 3),
    dropout_rate: float = DROPOUT_RATE,
) -> keras.Model:
    """Build EfficientNetB0 with frozen base + classification head (Stage 1 ready).

    Architecture:
        Input(224,224,3)
        EfficientNetB0(include_top=False, pooling=None) — all 237 layers frozen
        GlobalAveragePooling2D
        Dense(256, relu)
        Dropout(0.3)  — lower than CNN's 0.5; base is already regularized via MBConv + SE blocks
        Dense(1, sigmoid)

    Stage 1: only the 4-layer head is trainable (Dense → Dropout → Dense + GAP).
    Stage 2: call unfreeze_top_n() to adapt top FINE_TUNE_AT conv layers of the base.
    """
    tf.random.set_seed(RANDOM_STATE)

    inputs = keras.Input(shape=input_shape, name="input")

    # All base layers frozen — BN layers will run in inference mode during Stage 1
    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        pooling=None,
    )
    base.trainable = False

    # Call base as a sub-model — preserved as a named layer for unfreeze_top_n()
    x = base(inputs)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="dense_head")(x)
    x = layers.Dropout(dropout_rate, name="dropout_head")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="efficientnet_classifier")
    return model


def compile_stage1(
    model: keras.Model,
    learning_rate: float = LR_STAGE1,
) -> keras.Model:
    """Compile for Stage 1 — trains only the classification head at lr=1e-3.

    High initial LR is safe here: base is frozen, only the randomly-initialised
    Dense head trains. Using the same lr=1e-5 as Stage 2 would make head training
    extremely slow (thousands of epochs to converge).
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def unfreeze_top_n(model: keras.Model, n: int = FINE_TUNE_AT) -> None:
    """Unfreeze the top n conv layers of the EfficientNetB0 base for Stage 2 fine-tuning.

    Steps:
        1. base.trainable = True  — enables gradient flow through the base
        2. Freeze all layers except the top n — preserves low-level edge/texture detectors
        3. Re-freeze ALL BatchNorm layers — prevents corrupting ImageNet-calibrated statistics

    BatchNorm treatment (critical for small datasets):
        EfficientNetB0's BN running mean/variance are calibrated on 1.2M ImageNet images.
        Fine-tuning BN on ~2,600 domain images would corrupt those statistics with noisy
        batch-level estimates, hurting performance. Keeping all BN frozen preserves stable
        statistics while the convolutional weights still adapt to aerial imagery.

    Why top 20 of 237?
        Layers [0:-20] detect generic features — edges, textures, simple shapes — that
        transfer well to any visual domain. Keep them frozen. Layers [-20:] detect
        high-level ImageNet-specific patterns (final MBConv blocks) that need adaptation
        to our bird/drone domain.

    CRITICAL: model.compile() MUST be called after this function.
    Keras only picks up trainability changes at the next compile() call.
    """
    base = model.get_layer("efficientnetb0")
    base.trainable = True

    # Step 2 — freeze all layers except the top n
    for layer in base.layers[:-n]:
        layer.trainable = False

    # Step 3 — re-freeze ALL BatchNorm layers regardless of their position in the top-n window
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    trainable_conv = sum(
        1 for lyr in base.layers
        if lyr.trainable and not isinstance(lyr, tf.keras.layers.BatchNormalization)
    )
    logger.info(
        "Stage 2 ready: top %d layers unfrozen, %d trainable conv layers, all BN frozen",
        n, trainable_conv,
    )


def compile_stage2(
    model: keras.Model,
    learning_rate: float = LR_STAGE2,
) -> keras.Model:
    """Compile for Stage 2 fine-tuning at lr=1e-5.

    Very low LR is mandatory: high LR on partially-unfrozen pretrained weights
    causes catastrophic forgetting — the ImageNet features trained over days are
    destroyed in a few steps. Must call after unfreeze_top_n().
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def get_callbacks(
    checkpoint_path: str,
    patience: int = PATIENCE,
    monitor: str = "val_loss",
    mode: str = "min",
) -> list:
    """Return [EarlyStopping, ModelCheckpoint, ReduceLROnPlateau].

    Use separate checkpoint paths for Stage 1 and Stage 2:
        Stage 1: models/effnet_stage1_best.h5
        Stage 2: models/effnet_stage2_best.h5  ← production model

    Args:
        checkpoint_path: .h5 path for ModelCheckpoint.
        patience:        EarlyStopping patience (epochs without improvement).
        monitor:         Metric to watch — 'val_loss' or 'val_auc'.
        mode:            'min' for val_loss, 'max' for val_auc.
    """
    return [
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            mode=mode,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=monitor,
            save_best_only=True,
            save_format="h5",
            mode=mode,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=max(2, patience // 2),
            min_lr=1e-7,
            mode=mode,
            verbose=1,
        ),
    ]


def build_comparison_report(
    metrics_cnn_path: Path,
    metrics_effnet_path: Path,
    output_dir: Path,
) -> str:
    """Compare CNN vs EfficientNetB0, save model_comparison.csv, return winner name.

    Winner selection (ordered):
        1. Higher test_f1  — handles class imbalance better than accuracy
        2. Higher test_auc — ranking quality, threshold-independent (breaks F1 ties)
        3. EfficientNetB0  — second tiebreaker; higher capacity, expected better generalisation

    Side effects:
        Saves {output_dir}/model_comparison.csv with both models side by side.
        Writes {output_dir}/production_model.txt — one line, winner model name.
        FastAPI Section 6 reads production_model.txt at startup to select the .h5 to load.

    Returns:
        "efficientnet" or "custom_cnn"
    """
    output_dir = Path(output_dir)

    with open(metrics_cnn_path) as f:
        cnn = json.load(f)
    with open(metrics_effnet_path) as f:
        effnet = json.load(f)

    def _row(m: dict) -> dict:
        return {
            "model_name":        m["model_name"],
            "architecture":      m["architecture"],
            "test_accuracy":     m["metrics"]["test_accuracy"],
            "test_precision":    m["metrics"]["test_precision"],
            "test_recall":       m["metrics"]["test_recall"],
            "test_f1":           m["metrics"]["test_f1"],
            "test_auc":          m["metrics"]["test_auc"],
            "test_loss":         m["metrics"]["test_loss"],
            "optimal_threshold": m["optimal_threshold"]["threshold"],
            "training_time_min": m.get("training_time_minutes", "N/A"),
        }

    df = pd.DataFrame([_row(cnn), _row(effnet)])
    csv_path = output_dir / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved: %s", csv_path)

    # ── Winner selection ──────────────────────────────────────────────────────
    cnn_f1    = cnn["metrics"]["test_f1"]
    effnet_f1 = effnet["metrics"]["test_f1"]

    if effnet_f1 > cnn_f1:
        winner = "efficientnet"
    elif cnn_f1 > effnet_f1:
        winner = "custom_cnn"
    else:
        # F1 tie — compare AUC
        winner = (
            "efficientnet"
            if effnet["metrics"]["test_auc"] >= cnn["metrics"]["test_auc"]
            else "custom_cnn"
        )

    # Write winner for FastAPI to read at startup — replaces MLflow registry on Colab
    prod_txt = output_dir / "production_model.txt"
    prod_txt.write_text(winner)
    logger.info("Winner: %s → written to %s", winner, prod_txt)

    return winner
