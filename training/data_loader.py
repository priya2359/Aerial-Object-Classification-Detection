# filename: training/data_loader.py
# purpose:  Build train/valid/test ImageDataGenerators from eda_stats.json augmentation config
# version:  1.0

# stdlib
import json
import logging
from pathlib import Path

# third-party
from tensorflow.keras.preprocessing.image import ImageDataGenerator

logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
IMAGE_SIZE   = (224, 224)
BATCH_SIZE   = 32
RANDOM_STATE = 42


def build_generators(
    dataset_dir: Path,
    eda_stats_path: Path,
    batch_size: int = BATCH_SIZE,
    image_size: tuple = IMAGE_SIZE,
    preprocessing_fn=None,
) -> tuple:
    """Return (train_gen, valid_gen, test_gen, class_weight, class_indices).

    Augmentation config and class_weight are read from eda_stats.json — never hardcoded.
    Train generator uses full augmentation. Valid/test generators use rescale only.

    brightness_range is passed to flow_from_directory (not flow on arrays) — this is
    the only ImageDataGenerator argument that requires flow_from_directory to take effect.

    Args:
        preprocessing_fn: Optional callable (e.g. efficientnet.preprocess_input).
            When provided, replaces rescale=1/255 — the two are mutually exclusive.
            None  → rescale=1/255          (Custom CNN, Section 3)
            fn    → preprocessing_function=fn  (EfficientNetB0, Section 4)

    Returns:
        train_gen:     augmented training generator
        valid_gen:     rescale-only validation generator
        test_gen:      rescale-only test generator (shuffle=False, for evaluation)
        class_weight:  {0: float, 1: float} from eda_stats.json
        class_indices: {"bird": 0, "drone": 1} — alphabetical order confirmed
    """
    dataset_dir    = Path(dataset_dir)
    eda_stats_path = Path(eda_stats_path)

    with open(eda_stats_path) as f:
        eda_stats = json.load(f)

    aug_cfg      = eda_stats["augmentation_config"]
    class_weight = {int(k): v for k, v in eda_stats["class_weight"].items()}

    # Preprocessing — preprocessing_fn and rescale are mutually exclusive.
    # preprocess_input (EfficientNetB0) expects [0,255] and scales internally.
    # Dividing by 255 first then running preprocess_input would produce wrong range.
    if preprocessing_fn is not None:
        base_kwargs = {"preprocessing_function": preprocessing_fn}
    else:
        base_kwargs = {"rescale": 1.0 / 255.0}

    # ── Training generator — full augmentation ────────────────────────────────
    train_datagen = ImageDataGenerator(
        **base_kwargs,
        rotation_range=aug_cfg["rotation_range"],
        zoom_range=aug_cfg["zoom_range"],
        width_shift_range=aug_cfg["width_shift_range"],
        height_shift_range=aug_cfg["height_shift_range"],
        horizontal_flip=aug_cfg["horizontal_flip"],
        brightness_range=aug_cfg["brightness_range"],  # works with flow_from_directory
        fill_mode=aug_cfg["fill_mode"],
    )

    # ── Valid/test generators — no augmentation ───────────────────────────────
    infer_datagen = ImageDataGenerator(**base_kwargs)

    common_kwargs = dict(
        target_size=image_size,
        batch_size=batch_size,
        class_mode="binary",   # sigmoid output → binary labels
        seed=RANDOM_STATE,
    )

    train_gen = train_datagen.flow_from_directory(
        dataset_dir / "train",
        shuffle=True,
        **common_kwargs,
    )
    valid_gen = infer_datagen.flow_from_directory(
        dataset_dir / "valid",
        shuffle=False,
        **common_kwargs,
    )
    test_gen = infer_datagen.flow_from_directory(
        dataset_dir / "test",
        shuffle=False,    # must be False for evaluation — preserves label order
        **common_kwargs,
    )

    # Verify class assignment — flow_from_directory assigns alphabetically
    # "bird" < "drone" → bird=0, drone=1 — must match class_weight keys
    class_indices = train_gen.class_indices
    assert class_indices.get("bird") == 0 and class_indices.get("drone") == 1, (
        f"Unexpected class assignment: {class_indices}. "
        "Expected bird=0, drone=1 (alphabetical order). "
        "This would misalign with class_weight from eda_stats.json."
    )

    logger.info("class_indices: %s", class_indices)
    logger.info("class_weight:  %s", class_weight)
    logger.info(
        "train=%d  valid=%d  test=%d images",
        train_gen.n, valid_gen.n, test_gen.n,
    )

    return train_gen, valid_gen, test_gen, class_weight, class_indices
