# filename: tests/test_preprocessing.py
# purpose:  Unit tests for api/preprocessing.py — shape, dtype, range, hash determinism
# version:  1.0

# third-party
import numpy as np
import pytest

from api.preprocessing import compute_image_hash, preprocess_image

# ── preprocess_image ──────────────────────────────────────────────────────────

class TestPreprocessImage:

    def test_output_shape(self, sample_jpeg_bytes):
        """Output must be (1, 224, 224, 3) — batch dimension added by expand_dims."""
        arr = preprocess_image(sample_jpeg_bytes)
        assert arr.shape == (1, 224, 224, 3)

    def test_output_dtype_float32(self, sample_jpeg_bytes):
        """Output must be float32 — TF models expect float32 input."""
        arr = preprocess_image(sample_jpeg_bytes)
        assert arr.dtype == np.float32

    def test_efficientnet_range(self, sample_jpeg_bytes):
        """EfficientNetB0 preprocess_input (torch mode): maps [0,255] → ~[-2.2, 2.7].

        We verify negative values exist (confirms normalisation ran) and values
        are outside [0, 1] (confirms /255 was NOT applied separately).
        Range check uses loose bounds to avoid coupling tests to ImageNet stats.
        """
        arr = preprocess_image(sample_jpeg_bytes, model_type="efficientnet")
        assert arr.min() < 0.0, "efficientnet preprocess_input should produce negative values"
        assert arr.max() > 1.0, "efficientnet output should exceed 1.0 (not clamped to [0,1])"
        assert arr.min() >= -4.0, "sanity: no extreme negatives for valid JPEG input"
        assert arr.max() <= 4.0,  "sanity: no extreme positives for valid JPEG input"

    def test_custom_cnn_range(self, sample_jpeg_bytes):
        """Custom CNN: divide by 255.0 → values in [0.0, 1.0] exactly."""
        arr = preprocess_image(sample_jpeg_bytes, model_type="custom_cnn")
        assert arr.min() >= 0.0, "custom_cnn /255 must produce non-negative values"
        assert arr.max() <= 1.0, "custom_cnn /255 must produce values ≤ 1.0"

    def test_efficientnet_and_cnn_produce_different_values(self, sample_jpeg_bytes):
        """The two preprocessing paths must differ — catches accidental shared state."""
        eff = preprocess_image(sample_jpeg_bytes, model_type="efficientnet")
        cnn = preprocess_image(sample_jpeg_bytes, model_type="custom_cnn")
        assert not np.allclose(eff, cnn), \
            "efficientnet and custom_cnn preprocessing must produce different values"

    def test_custom_target_size(self, sample_jpeg_bytes):
        """Caller can override target_size — used if model is retrained at different input size."""
        arr = preprocess_image(sample_jpeg_bytes, target_size=(128, 128))
        assert arr.shape == (1, 128, 128, 3)

    def test_default_model_type_is_efficientnet(self, sample_jpeg_bytes):
        """Default preprocessing matches EfficientNetB0 — production winner."""
        arr_default = preprocess_image(sample_jpeg_bytes)
        arr_eff     = preprocess_image(sample_jpeg_bytes, model_type="efficientnet")
        assert np.allclose(arr_default, arr_eff)


# ── compute_image_hash ────────────────────────────────────────────────────────

class TestComputeImageHash:

    def test_returns_hex_string(self, sample_jpeg_bytes):
        h = compute_image_hash(sample_jpeg_bytes)
        assert isinstance(h, str)
        assert len(h) == 32, "MD5 hex digest must be 32 characters"
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self, sample_jpeg_bytes):
        """Same bytes → same hash every time (no random state in MD5)."""
        h1 = compute_image_hash(sample_jpeg_bytes)
        h2 = compute_image_hash(sample_jpeg_bytes)
        assert h1 == h2

    def test_different_content_different_hash(self, sample_jpeg_bytes):
        """Different image bytes must produce different hashes (no collision in test set)."""
        import io
        from PIL import Image
        alt_arr = np.zeros((224, 224, 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(alt_arr, mode="RGB").save(buf, format="JPEG")
        h_orig = compute_image_hash(sample_jpeg_bytes)
        h_alt  = compute_image_hash(buf.getvalue())
        assert h_orig != h_alt

    def test_single_byte_change_changes_hash(self):
        """Avalanche effect — 1 bit change should produce a completely different hash."""
        b1 = b"aerial_detection_test_bytes_v1"
        b2 = b"aerial_detection_test_bytes_v2"
        assert compute_image_hash(b1) != compute_image_hash(b2)
