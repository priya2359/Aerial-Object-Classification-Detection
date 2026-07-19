# filename: training/yolov8_train.py
# purpose:  YOLOv8n fine-tuning helpers — YAML creation, dataset verification, training, evaluation
# version:  1.0

# stdlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

# third-party
import yaml

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_STATE   = 42
MODEL_NAME     = "yolov8n.pt"   # nano: 3.2M params — appropriate for 3K-image dataset
IMGSZ          = 640            # YOLOv8 standard (separate from classification 224×224)
EPOCHS         = 50
BATCH          = 16             # fits T4 GPU with yolov8n at imgsz=640
FREEZE_LAYERS  = 3              # freeze early backbone only — nano has ~9-10 total backbone layers;
                                # freeze=10 would freeze the entire backbone (defeats fine-tuning)
PATIENCE       = 10             # early stopping on mAP50
CONFIDENCE_THR = 0.5            # confidence threshold for evaluation
DEVICE         = 0              # GPU device index
PROJECT_DIR    = "runs/detect"
RUN_NAME       = "aerial_nano"

# Must match class IDs in YOLO .txt annotation files (class 0 = bird, class 1 = drone)
CLASS_NAMES = {0: "bird", 1: "drone"}


def create_dataset_yaml(
    dataset_root: str,
    output_path: str = "data/yolo_dataset.yaml",
) -> str:
    """Write Ultralytics dataset YAML using path + relative subdirectory pattern.

    Ultralytics resolves train/val/test relative to `path`, NOT relative to the YAML file.
    Putting full paths in train/val/test fields causes double-prefix resolution errors.
    """
    config = {
        "path":  str(dataset_root),  # absolute path to dataset root on Colab Drive
        "train": "train/images",      # resolved as: dataset_root/train/images
        "val":   "valid/images",
        "test":  "test/images",
        "nc":    len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info("Dataset YAML written to %s", out)
    return str(out)


def verify_dataset_structure(dataset_root: str) -> dict:
    """Verify images/ + labels/ subdirectory structure and image-label pairing.

    YOLOv8 silently skips images without matching labels — producing misleading metrics.
    This check runs before training so unpaired files are caught, not hidden.
    """
    root = Path(dataset_root)
    report = {}
    for split in ("train", "valid", "test"):
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        if not img_dir.exists():
            logger.error("images/ directory missing: %s", img_dir)
            report[split] = {"error": f"missing {img_dir}"}
            continue
        exts = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG")
        images = [p for ext in exts for p in img_dir.glob(ext)]
        labels = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []
        img_stems = {p.stem for p in images}
        lbl_stems = {p.stem for p in labels}
        unpaired = img_stems.symmetric_difference(lbl_stems)
        report[split] = {
            "n_images":   len(images),
            "n_labels":   len(labels),
            "n_unpaired": len(unpaired),
        }
        if unpaired:
            logger.warning(
                "%s: %d unpaired files (first 5: %s)",
                split, len(unpaired), list(unpaired)[:5],
            )
        else:
            logger.info("%s: %d images, %d labels — all paired ✓", split, len(images), len(labels))
    return report


def sample_annotation_check(dataset_root: str, n_samples: int = 3) -> None:
    """Print n_samples annotation files from training set for manual class ID verification.

    Class ID mismatch is a silent failure: class 0=drone in annotations + class 0=bird in YAML
    trains inverted labels, converges normally, and produces plausible-looking metrics.
    This 30-second check catches the error before 15 minutes of GPU time is wasted.
    """
    root = Path(dataset_root)
    label_files = sorted((root / "train" / "labels").glob("*.txt"))[:n_samples]
    for lbl in label_files:
        lines = lbl.read_text().strip().splitlines()
        print(f"\n{lbl.name}:")
        for line in lines[:3]:
            parts = line.split()
            class_id   = int(parts[0])
            class_name = CLASS_NAMES.get(class_id, f"unknown({class_id})")
            print(f"  class_id={class_id} ({class_name})  bbox={[round(float(x),4) for x in parts[1:]]}")
    print("\nExpected: class_id=0 → bird, class_id=1 → drone")


def train(
    yaml_path: str,
    model_name: str = MODEL_NAME,
    epochs: int = EPOCHS,
    imgsz: int = IMGSZ,
    batch: int = BATCH,
    freeze: int = FREEZE_LAYERS,
    patience: int = PATIENCE,
    device: int = DEVICE,
    seed: int = RANDOM_STATE,
    project: str = PROJECT_DIR,
    name: str = RUN_NAME,
):
    """Fine-tune YOLOv8n on the aerial dataset.

    MLflow env vars (MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME) MUST be set
    before calling this function. Ultralytics checks for MLflow on the first
    training step — setting them after model.train() starts has no effect.

    Returns (model, elapsed_minutes).
    Best weights saved to: {project}/{name}/weights/best.pt
    """
    from ultralytics import YOLO

    model = YOLO(model_name)
    t0 = time.time()
    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        freeze=freeze,
        patience=patience,
        device=device,
        seed=seed,
        project=project,
        name=name,
        exist_ok=True,
    )
    elapsed_min = (time.time() - t0) / 60
    logger.info("Training complete in %.1f min — weights: %s/%s/weights/best.pt", elapsed_min, project, name)
    return model, elapsed_min


def evaluate(
    model,
    yaml_path: str,
    split: str = "test",
    conf: float = CONFIDENCE_THR,
) -> dict:
    """Run model.val() on the specified split. Returns metrics dict.

    model.val() returns a Metrics object with .box.map50, .box.map, .box.mp, .box.mr.
    """
    results = model.val(data=yaml_path, split=split, conf=conf, verbose=False)
    metrics = {
        "map50":                round(float(results.box.map50), 4),
        "map50_95":             round(float(results.box.map),   4),
        "precision":            round(float(results.box.mp),    4),
        "recall":               round(float(results.box.mr),    4),
        "confidence_threshold": conf,
    }
    logger.info("Test metrics: %s", metrics)
    return metrics


def export_metrics(
    metrics: dict,
    training_time_min: float,
    epochs_run: int,
    early_stopped: bool,
    mlflow_run_id: str,
    output_path: str = "models/metrics_yolov8.json",
) -> None:
    """Save metrics_yolov8.json — same schema pattern as metrics_cnn.json / metrics_effnet.json."""
    record = {
        "model_name": "yolov8n",
        "task":       "object_detection",
        "training": {
            "epochs_run":    epochs_run,
            "epochs_max":    EPOCHS,
            "early_stopped": early_stopped,
            "freeze_layers": FREEZE_LAYERS,
            "imgsz":         IMGSZ,
            "batch":         BATCH,
        },
        "metrics": metrics,
        "artifacts": {
            "weights_file":  "models/yolov8n_best.pt",
            "mlflow_run_id": mlflow_run_id,
        },
        "training_time_minutes": round(training_time_min, 2),
        "generated_at":          datetime.now(timezone.utc).isoformat(),
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(record, f, indent=2)
    logger.info("Metrics saved: %s", out)
