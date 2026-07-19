# filename: feature_store/precompute.py
# purpose:  Bulk-precompute EfficientNetB0 embeddings for all dataset images and cache in Redis
# version:  1.1

# stdlib
import hashlib
import logging
import os
import time
from pathlib import Path

# third-party
import numpy as np

# internal
from feature_store.redis_client import build_key, get_redis_client, store_embedding

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
BATCH_SIZE    = 32
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
IMAGE_SIZE    = (224, 224)
DATASET_DIR   = Path(os.getenv("DATASET_DIR", "data/classification_dataset"))
MODELS_DIR    = Path(os.getenv("MODELS_DIR", "models"))
STAGING_DIR   = Path(os.getenv("STAGING_DIR", "data/staging"))
SPLITS        = ("train", "valid", "test")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def compute_md5(image_path: Path) -> str:
    """Return MD5 hex digest of raw image bytes — matches DAG transform step."""
    return hashlib.md5(image_path.read_bytes()).hexdigest()


def load_and_preprocess(image_path: Path) -> np.ndarray:
    """Load image, resize to IMAGE_SIZE, apply EfficientNetB0 preprocess_input.

    CRITICAL: must use preprocess_input (maps [0,255]→[-1,1]), NOT /255.
    The classification head was trained on preprocess_input-normalised embeddings.
    Using /255 produces embeddings in a different space → wrong predictions.
    """
    from PIL import Image  # local import — Pillow always available
    from tensorflow.keras.applications.efficientnet import preprocess_input

    img = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32)   # [0, 255]
    return preprocess_input(arr)             # [-1,  1] — matches api/preprocessing.py


def _collect_image_paths() -> list[Path]:
    """Walk DATASET_DIR/{split}/{class}/ and return deduplicated image paths.

    Deduplication by lowercase filename is required on case-insensitive filesystems
    (Windows, macOS HFS+) where *.jpg and *.JPG match the same files.
    """
    seen: dict[str, Path] = {}
    for split in SPLITS:
        for label in ("bird", "drone"):
            folder = DATASET_DIR / split / label
            if not folder.exists():
                logger.warning("Directory not found, skipping: %s", folder)
                continue
            for ext in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"):
                for p in folder.glob(ext):
                    key = str(p.parent / p.name.lower())
                    if key not in seen:
                        seen[key] = p
    return list(seen.values())


def _batches(items: list, size: int):
    """Yield successive chunks of `size` from `items`."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ─── Main Function ────────────────────────────────────────────────────────────

def _load_embedding_model():
    """Load the fine-tuned production embedding model split at the GAP layer.

    Reads production_model.txt → loads the full fine-tuned .h5 → returns the
    sub-model from input to GlobalAveragePooling2D output.

    This MUST use the fine-tuned model (not the base ImageNet EfficientNetB0)
    because the classification head was trained on fine-tuned embeddings.
    Base model embeddings occupy a different feature space → wrong predictions.
    """
    from tensorflow import keras

    prod_txt = MODELS_DIR / "production_model.txt"
    if not prod_txt.exists():
        logger.warning(
            "production_model.txt not found in %s — falling back to base ImageNet EfficientNetB0. "
            "Embeddings will be WRONG if the head was fine-tuned. Run training first.",
            MODELS_DIR,
        )
        from tensorflow.keras.applications import EfficientNetB0
        base = EfficientNetB0(include_top=False, weights="imagenet", pooling="avg")
        base.trainable = False
        return base

    model_type = prod_txt.read_text().strip()
    model_path  = MODELS_DIR / f"{model_type}_final.h5"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Production model not found: {model_path}. "
            "Copy the trained .h5 from Colab to models/ before running precompute."
        )

    logger.info("Loading fine-tuned production model: %s", model_path)
    full_model = keras.models.load_model(str(model_path))

    gap_layer = next(
        (l for l in full_model.layers if isinstance(l, keras.layers.GlobalAveragePooling2D)),
        None,
    )
    if gap_layer is None:
        raise RuntimeError(f"No GlobalAveragePooling2D layer in {model_path}")

    embedding_model = keras.Model(
        inputs=full_model.input,
        outputs=gap_layer.output,
        name="embedding_extractor",
    )
    embedding_model.trainable = False
    logger.info(
        "Embedding model ready: %s → GAP output shape %s",
        model_type, gap_layer.output_shape,
    )
    return embedding_model


def precompute_all() -> dict:
    """Compute production-model embeddings for every image in DATASET_DIR.

    Skips images already cached in Redis (idempotent).
    On Redis connection error, logs the failed batch index and continues.
    Returns a summary dict with counts.
    """
    logger.info("Loading production embedding model ...")
    base = _load_embedding_model()
    # output shape: (None, D) where D=1280 for EfficientNetB0, 128 for Custom CNN

    image_paths = _collect_image_paths()
    total       = len(image_paths)
    logger.info("Found %d images in %s", total, DATASET_DIR)

    if total == 0:
        logger.warning("No images found. Check DATASET_DIR=%s", DATASET_DIR)
        return {"total": 0, "processed": 0, "skipped_cached": 0, "failed_batches": []}

    client         = get_redis_client()
    processed      = 0
    skipped_cached = 0
    failed_batches: list[int] = []
    t0             = time.perf_counter()

    for batch_idx, batch_paths in enumerate(_batches(image_paths, BATCH_SIZE)):
        try:
            # Separate paths that need embedding from those already cached
            to_embed: list[Path] = []
            hashes:   list[str]  = []

            for p in batch_paths:
                h = compute_md5(p)
                if client.get(build_key(h, MODEL_VERSION)) is not None:
                    skipped_cached += 1
                else:
                    to_embed.append(p)
                    hashes.append(h)

            if not to_embed:
                continue

            # Preprocess batch and run through frozen base
            batch_array  = np.stack([load_and_preprocess(p) for p in to_embed])
            embeddings   = base.predict(batch_array, verbose=0)  # shape (N, 1280)

            for h, emb in zip(hashes, embeddings):
                store_embedding(image_hash=h, embedding=emb, model_version=MODEL_VERSION)
                processed += 1

            logger.info(
                "Batch %d/%d — embedded %d, cached %d",
                batch_idx + 1,
                -(-total // BATCH_SIZE),  # ceil division
                len(to_embed),
                skipped_cached,
            )

        except Exception as exc:  # noqa: BLE001
            import redis as redis_lib

            if isinstance(exc, redis_lib.ConnectionError):
                logger.error("Redis unavailable on batch %d: %s — skipping batch", batch_idx, exc)
            else:
                logger.error("Unexpected error on batch %d: %s — skipping batch", batch_idx, exc)
            failed_batches.append(batch_idx)
            continue

    elapsed = time.perf_counter() - t0
    logger.info(
        "Precompute complete in %.1fs — processed=%d skipped_cached=%d failed_batches=%s",
        elapsed, processed, skipped_cached, failed_batches,
    )

    if failed_batches:
        STAGING_DIR.mkdir(parents=True, exist_ok=True)
        failed_file = STAGING_DIR / "failed_batches.txt"
        failed_file.write_text("\n".join(str(b) for b in failed_batches))
        logger.warning("Failed batch indices written to %s — rerun to retry", failed_file)

    return {
        "total":           total,
        "processed":       processed,
        "skipped_cached":  skipped_cached,
        "failed_batches":  failed_batches,
    }


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    summary = precompute_all()
    logger.info("Summary: %s", summary)
