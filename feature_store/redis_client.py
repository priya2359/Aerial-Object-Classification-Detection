# filename: feature_store/redis_client.py
# purpose:  Redis helpers for storing and retrieving EfficientNetB0 embeddings
# version:  1.1

# stdlib
import logging
import os

# third-party
import numpy as np
import redis

logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
REDIS_HOST    = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT    = int(os.getenv("REDIS_PORT", "6379"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
TTL_SECONDS   = int(os.getenv("REDIS_TTL_DAYS", "7")) * 86400  # convert days → seconds

# M3 fix: module-level singleton — avoids creating a new TCP connection on every
# store_embedding() / get_embedding() call. The precompute script processes ~3,319
# images with 2 Redis calls each = 6,638 connections without this fix.
_redis_client: redis.Redis | None = None


# ─── Public API ───────────────────────────────────────────────────────────────

def get_redis_client() -> redis.Redis:
    """Return the module-level Redis client, creating it once on first call.

    decode_responses=False is mandatory — we store raw binary numpy bytes.
    Raises redis.ConnectionError if the server is unreachable.
    """
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=False,
        )
        _redis_client.ping()  # fail fast if Redis is down on first connect
        logger.debug("Redis client created: %s:%d", REDIS_HOST, REDIS_PORT)
    return _redis_client


def build_key(image_hash: str, model_version: str | None = None) -> str:
    """Construct the canonical Redis key for an embedding.

    Format: aerial:features:{model_version}:{image_hash}
    The model_version prefix is mandatory — v1 and v2 embeddings must never mix.
    """
    mv = model_version or MODEL_VERSION
    return f"aerial:features:{mv}:{image_hash}"


def store_embedding(
    image_hash: str,
    embedding: np.ndarray,
    model_version: str | None = None,
    ttl_seconds: int | None = None,
) -> None:
    """Serialise a float32 embedding array to raw bytes and store in Redis with TTL.

    Args:
        image_hash:    MD5 hex string of the original image bytes.
        embedding:     1-D numpy array, shape (1280,), dtype float32.
        model_version: Override MODEL_VERSION env var (rarely needed).
        ttl_seconds:   Override default TTL (default: REDIS_TTL_DAYS * 86400).
    """
    key  = build_key(image_hash, model_version)
    ttl  = ttl_seconds if ttl_seconds is not None else TTL_SECONDS
    data = embedding.astype(np.float32).tobytes()

    client = get_redis_client()
    client.setex(name=key, time=ttl, value=data)
    logger.debug("Stored embedding: key=%s ttl=%ds bytes=%d", key, ttl, len(data))


def get_embedding(
    image_hash: str,
    model_version: str | None = None,
) -> np.ndarray | None:
    """Retrieve a cached embedding from Redis.

    Returns:
        numpy float32 array of shape (1280,) on cache hit.
        None on cache miss.
    """
    key    = build_key(image_hash, model_version)
    client = get_redis_client()
    raw    = client.get(key)

    if raw is None:
        logger.debug("Cache miss: key=%s", key)
        return None

    embedding = np.frombuffer(raw, dtype=np.float32)
    logger.debug("Cache hit:  key=%s shape=%s", key, embedding.shape)
    return embedding
