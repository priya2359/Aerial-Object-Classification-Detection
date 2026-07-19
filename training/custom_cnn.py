# filename: training/custom_cnn.py
# purpose:  Custom CNN Model A — 3-block Conv + GAP + Dense + Sigmoid for bird/drone classification
# version:  1.0

# stdlib
import logging

# third-party
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
RANDOM_STATE       = 42
IMAGE_SIZE         = (224, 224)
DROPOUT_RATE       = 0.5
LEARNING_RATE      = 1e-3
EPOCHS_CNN         = 50
PATIENCE           = 7
LR_REDUCE_PATIENCE = 3
LR_REDUCE_FACTOR   = 0.5
MIN_LR             = 1e-6


def build_custom_cnn(
    input_shape: tuple = (224, 224, 3),
    dropout_rate: float = DROPOUT_RATE,
) -> keras.Model:
    """Build 3-block Custom CNN for binary bird/drone classification.

    Architecture:
        Block 1: Conv2D(32, relu) → BatchNorm → MaxPool(2×2)
        Block 2: Conv2D(64, relu) → BatchNorm → MaxPool(2×2)
        Block 3: Conv2D(128, relu) → BatchNorm → MaxPool(2×2)
        Head:    GlobalAveragePooling2D → Dense(128, relu) → Dropout → Dense(1, sigmoid)

    Design decisions:
        Conv(relu) → BatchNorm → MaxPool order:
            BN after activation normalises activated outputs before spatial reduction.
            Empirically stable for small datasets; simpler than pre-activation BN.

        GlobalAveragePooling2D over Flatten:
            After 3 MaxPool(2×2): 224→112→56→28 px spatial dimension.
            Feature map at head = 28×28×128.
            Flatten → Dense(128): 28×28×128 × 128 = 12.8M parameters (overfit risk).
            GAP    → Dense(128): 128 × 128         = 16,384 parameters.
            GAP is also translation-invariant and a standard head for modern CNNs.
    """
    tf.random.set_seed(RANDOM_STATE)

    inputs = keras.Input(shape=input_shape, name="input")

    # Block 1 — detect low-level features: edges, corners, colour gradients
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)   # 224 → 112

    # Block 2 — detect mid-level patterns: textures, wing shapes, rotor outlines
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)   # 112 → 56

    # Block 3 — detect high-level semantics: "drone body", "bird silhouette"
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)   # 56 → 28

    # Classification head
    x = layers.GlobalAveragePooling2D(name="gap")(x)          # 28×28×128 → 128
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="custom_cnn")
    return model


def compile_model(
    model: keras.Model,
    learning_rate: float = LEARNING_RATE,
) -> keras.Model:
    """Compile with Adam, binary_crossentropy, and four evaluation metrics."""
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


def get_callbacks(checkpoint_path: str) -> list:
    """Return [EarlyStopping, ModelCheckpoint, ReduceLROnPlateau].

    EarlyStopping with restore_best_weights=True ensures the returned model
    is from the best epoch, not the final epoch.
    """
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            save_format="h5",
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_REDUCE_FACTOR,
            patience=LR_REDUCE_PATIENCE,
            min_lr=MIN_LR,
            verbose=1,
        ),
    ]
