# filename: dags/aerial_ingestion_dag.py
# purpose:  Airflow ETL — ingest classification images into PostgreSQL, then trigger Redis precompute
# version:  1.0

# stdlib
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Airflow worker has feature_store/ at /opt/airflow; add to path for Task 5 import
sys.path.insert(0, "/opt/airflow")

# third-party (Airflow)
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.dates import days_ago

logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
DAG_ID       = "aerial_image_ingestion"
DATASET_DIR  = Path("/opt/airflow/data/classification_dataset")
STAGING_DIR  = Path("/opt/airflow/data/staging")
SPLITS       = ("train", "valid", "test")
LABEL_MAP    = {"bird": 0, "drone": 1}
PG_CONN_ID   = "postgres_aerial"

STAGING_DIR.mkdir(parents=True, exist_ok=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _staging_path(prefix: str, run_id: str) -> Path:
    safe_id = run_id.replace(":", "_").replace("+", "_")
    return STAGING_DIR / f"{prefix}_{safe_id}.json"


def _compute_md5(file_path: Path) -> str:
    return hashlib.md5(file_path.read_bytes()).hexdigest()


def _log_pipeline_run(
    cur,
    dag_id: str,
    run_id: str,
    status: str,
    ingested: int,
    skipped: int,
    started_at: datetime,
) -> None:
    cur.execute(
        """INSERT INTO pipeline_run_logs
           (dag_id, run_id, status, images_ingested, images_skipped, started_at, completed_at)
           VALUES (%s, %s, %s, %s, %s, %s, NOW())""",
        (dag_id, run_id, status, ingested, skipped, started_at),
    )


# ─── Task Functions ───────────────────────────────────────────────────────────

def extract_from_local(**kwargs) -> None:
    """Walk classification_dataset, build metadata list, write to staging JSON.

    XCom payload: {"staging_file": "<path>", "count": N}
    Swap walk() for google.cloud.storage client to ingest from GCS in production.
    """
    ti     = kwargs["ti"]
    run_id = kwargs["run_id"]

    # Deduplicate by (folder, lowercase_filename) — on case-insensitive filesystems
    # (Windows, macOS HFS+) globbing *.jpg and *.JPG returns the same file twice.
    seen: dict[str, dict] = {}
    for split in SPLITS:
        for label, class_id in LABEL_MAP.items():
            folder = DATASET_DIR / split / label
            if not folder.exists():
                logger.warning("Directory not found, skipping: %s", folder)
                continue
            for ext in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"):
                for img_path in folder.glob(ext):
                    key = str(img_path.parent / img_path.name.lower())
                    if key not in seen:
                        seen[key] = {
                            "filepath":  str(img_path),
                            "filename":  img_path.name,
                            "label":     label,
                            "split":     split,
                            "class_id":  class_id,
                        }
    records: list[dict] = list(seen.values())

    staging_file = _staging_path("raw", run_id)
    staging_file.write_text(json.dumps(records))

    logger.info("extract_from_local: found %d images → %s", len(records), staging_file)
    ti.xcom_push(key="result", value={"staging_file": str(staging_file), "count": len(records)})


def validate_images(**kwargs) -> None:
    """Open each image with Pillow; drop corrupt files. Write validated list to staging JSON.

    XCom payload: {"staging_file": "<path>", "count": M}
    """
    from PIL import Image, UnidentifiedImageError

    ti     = kwargs["ti"]
    run_id = kwargs["run_id"]

    prev       = ti.xcom_pull(task_ids="extract_from_local", key="result")
    records    = json.loads(Path(prev["staging_file"]).read_text())
    valid: list[dict] = []

    for rec in records:
        img_path = Path(rec["filepath"])
        name_lower = img_path.name.lower()

        if not name_lower.endswith((".jpg", ".jpeg", ".png")):
            logger.warning("Unexpected extension, skipping: %s", img_path)
            continue

        try:
            with Image.open(img_path) as img:
                if img.size[0] == 0 or img.size[1] == 0:
                    raise ValueError("zero-dimension image")
                img.verify()  # detects truncated files
            valid.append(rec)
        except (UnidentifiedImageError, ValueError, OSError) as exc:
            logger.warning("Corrupt image, skipping %s: %s", img_path.name, exc)

    staging_file = _staging_path("valid", run_id)
    staging_file.write_text(json.dumps(valid))

    dropped = len(records) - len(valid)
    logger.info("validate_images: %d valid, %d dropped → %s", len(valid), dropped, staging_file)
    ti.xcom_push(key="result", value={"staging_file": str(staging_file), "count": len(valid)})


def transform_images(**kwargs) -> None:
    """Compute MD5 hash and image dimensions for each validated image.

    Does NOT resize — resize happens in training/data_loader.py.
    Decouples DAG from model architecture (changing to 299x299 won't touch this file).

    Hash note: MD5 is used for deduplication speed, not security. For ~3,000 local images
    the collision probability is negligible. For untrusted external uploads, switch to SHA-256
    — the schema column VARCHAR(32) would need widening to VARCHAR(64).

    XCom payload: {"staging_file": "<path>", "count": N}
    """
    from PIL import Image

    ti     = kwargs["ti"]
    run_id = kwargs["run_id"]

    prev    = ti.xcom_pull(task_ids="validate_images", key="result")
    records = json.loads(Path(prev["staging_file"]).read_text())

    for rec in records:
        img_path = Path(rec["filepath"])
        rec["file_hash"] = _compute_md5(img_path)
        with Image.open(img_path) as img:
            rec["image_width"], rec["image_height"] = img.size

    staging_file = _staging_path("transformed", run_id)
    staging_file.write_text(json.dumps(records))

    logger.info("transform_images: enriched %d records → %s", len(records), staging_file)
    ti.xcom_push(key="result", value={"staging_file": str(staging_file), "count": len(records)})


def load_to_postgres(**kwargs) -> None:
    """Bulk-insert transformed records into image_registry. Transaction-wrapped.

    ON CONFLICT (file_hash) DO NOTHING makes re-runs fully idempotent.
    run_id and dag_id are read from Airflow context and written to pipeline_run_logs.

    XCom payload: {"inserted": N, "skipped": M}
    """
    ti         = kwargs["ti"]
    run_id     = kwargs["run_id"]
    dag_id     = kwargs["dag"].dag_id
    started_at = kwargs["data_interval_start"] or datetime.utcnow()

    prev    = ti.xcom_pull(task_ids="transform_images", key="result")
    records = json.loads(Path(prev["staging_file"]).read_text())

    rows = [
        (
            rec["filename"],
            rec["label"],
            rec["class_id"],
            rec["split"],
            rec["file_hash"],
            rec["image_width"],
            rec["image_height"],
        )
        for rec in records
    ]

    hook = PostgresHook(postgres_conn_id=PG_CONN_ID)
    conn = hook.get_conn()
    cur  = conn.cursor()

    try:
        hashes = [r[4] for r in rows]

        # Count pre-existing rows before insert — these will be skipped by ON CONFLICT
        cur.execute(
            "SELECT COUNT(*) FROM image_registry WHERE file_hash = ANY(%s)", (hashes,)
        )
        pre_existing = cur.fetchone()[0]

        cur.executemany(
            """INSERT INTO image_registry
               (filename, label, class_id, split, file_hash, image_width, image_height)
               VALUES (%s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (file_hash) DO NOTHING""",
            rows,
        )

        inserted = len(rows) - pre_existing
        skipped  = pre_existing

        _log_pipeline_run(cur, dag_id, run_id, "success", inserted, skipped, started_at)
        conn.commit()

    except Exception:
        conn.rollback()
        # Log failure record in a separate transaction
        try:
            _log_pipeline_run(cur, dag_id, run_id, "failed", 0, 0, started_at)
            conn.commit()
        except Exception:
            pass
        raise
    finally:
        cur.close()
        conn.close()

    logger.info("load_to_postgres: inserted=%d skipped=%d", inserted, skipped)
    ti.xcom_push(key="result", value={"inserted": inserted, "skipped": skipped})


def trigger_feature_computation(**kwargs) -> None:
    """Call precompute_all() if new images were inserted; skip otherwise."""
    ti   = kwargs["ti"]
    prev = ti.xcom_pull(task_ids="load_to_postgres", key="result")
    inserted = prev.get("inserted", 0)

    if inserted == 0:
        logger.info("No new images inserted — skipping feature precompute.")
        return

    logger.info("%d new images — starting Redis embedding precompute ...", inserted)

    # Direct import (sys.path patched at module top) — avoids subprocess fragility
    from feature_store.precompute import precompute_all
    summary = precompute_all()
    logger.info("Precompute summary: %s", summary)


# ─── DAG Definition ───────────────────────────────────────────────────────────

default_args = {
    "owner":            "aerial-team",
    "retries":          1,
    "email_on_failure": False,
}

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["aerial", "etl", "section-1"],
) as dag:

    t1 = PythonOperator(task_id="extract_from_local",        python_callable=extract_from_local)
    t2 = PythonOperator(task_id="validate_images",           python_callable=validate_images)
    t3 = PythonOperator(task_id="transform_images",          python_callable=transform_images)
    t4 = PythonOperator(task_id="load_to_postgres",          python_callable=load_to_postgres)
    t5 = PythonOperator(task_id="trigger_feature_computation", python_callable=trigger_feature_computation)

    t1 >> t2 >> t3 >> t4 >> t5
