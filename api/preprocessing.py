# filename: api/preprocessing.py
# purpose:  Image preprocessing and MD5 hash computation for FastAPI inference
# version:  1.1

# stdlib
import hashlib
import io

# third-party
import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import HTTPException
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess

IMAGE_SIZE = (224, 224)


def preprocess_image(
    image_bytes: bytes,
    target_size: tuple = IMAGE_SIZE,
    model_type: str = "efficientnet",
) -> np.ndarray:
    """Decode, resize, and normalise image bytes for model inference.

    Args:
        image_bytes: Raw bytes from an uploaded file.
        target_size: (H, W) to resize to — must match training IMAGE_SIZE.
        model_type:  "efficientnet" → preprocess_input (scales [0,255] to [-1,1])
                     "custom_cnn"   → divide by 255.0 (scales [0,255] to [0,1])

    Returns:
        (1, H, W, 3) float32 array ready for model.predict() or model(x, training=False).

    Raises:
        HTTPException(400) for corrupt, truncated, or non-image files.
        Previously these raised PIL exceptions which FastAPI caught as 500 errors.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(target_size)
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot decode image: {exc}. Upload a valid JPEG or PNG file.",
        ) from exc

    arr = np.array(img, dtype=np.float32)

    if model_type == "efficientnet":
        arr = effnet_preprocess(arr)
    else:
        arr = arr / 255.0

    return np.expand_dims(arr, axis=0)  # (1, H, W, 3)


def compute_image_hash(image_bytes: bytes) -> str:
    """Return MD5 hex digest of raw image bytes.

    Used as the content fingerprint in the Redis cache key:
        aerial:features:{MODEL_VERSION}:{image_hash}

    MD5 is not cryptographically secure, but collision resistance is not a
    security requirement here — only a reliable content fingerprint.
    """
    return hashlib.md5(image_bytes).hexdigest()
