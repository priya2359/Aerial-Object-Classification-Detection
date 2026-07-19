# filename: api/create_stub_model.py
# purpose:  Create a minimal stub Keras model for demo/cloud deployment without real trained weights.
#           Skips creation if the .h5 already exists (real Colab weights take priority).
# version:  1.0

import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
MODEL_TYPE = "custom_cnn"
MODEL_PATH = MODELS_DIR / f"{MODEL_TYPE}_final.h5"


def _build_stub_model():
    """Tiny 3-layer CNN matching the expected GAP-based architecture.

    architecture: Input(224,224,3) → Conv2D(32) → GAP → Dense(128) → Dropout → Dense(1,sigmoid)
    The GlobalAveragePooling2D layer is required — FastAPI splits the model at this boundary.
    """
    import tensorflow as tf
    from tensorflow import keras

    tf.random.set_seed(42)

    inp = keras.Input(shape=(224, 224, 3), name="input_image")
    x   = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inp)
    x   = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x   = keras.layers.Dense(128, activation="relu")(x)
    x   = keras.layers.Dropout(0.5)(x)
    out = keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="custom_cnn_stub")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists():
        logger.info("Model already exists at %s — skipping stub creation.", MODEL_PATH)
        sys.exit(0)

    logger.info("No model found at %s — building stub model …", MODEL_PATH)
    model = _build_stub_model()
    model.save(str(MODEL_PATH))
    size_kb = MODEL_PATH.stat().st_size / 1024
    logger.info(
        "Stub model saved: %s  (%.0f KB, %d parameters)",
        MODEL_PATH, size_kb, model.count_params(),
    )
    logger.warning(
        "STUB MODEL ACTIVE — predictions are random (~0.5 confidence). "
        "Run notebooks/02_CustomCNN.ipynb on Colab, copy the .h5 here, then rebuild the image."
    )


if __name__ == "__main__":
    main()
