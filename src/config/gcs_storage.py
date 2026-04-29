"""GCS Model Storage — Download/upload models from Google Cloud Storage.

Usage:
    Local dev:  Uses local filesystem (models/current/)
    Cloud Run:  Downloads from GCS bucket on startup, uploads after training

Environment:
    GCS_BUCKET:     Bucket name (e.g. "weather-models-vn")
    GCS_MODEL_DIR:  Prefix in bucket (default: "current/")
    USE_GCS:        "true" to enable GCS, otherwise local filesystem
"""
import os
import shutil
from pathlib import Path

from src.config.settings import PROJECT_ROOT

LOCAL_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "current")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
GCS_MODEL_PREFIX = os.getenv("GCS_MODEL_DIR", "current/")
USE_GCS = os.getenv("USE_GCS", "false").lower() == "true"

_gcs_client = None


def _get_gcs_client():
    """Lazy-init GCS client to avoid import cost on cold start."""
    global _gcs_client
    if _gcs_client is None:
        from google.cloud import storage
        _gcs_client = storage.Client()
    return _gcs_client


def download_model(filename: str, local_dir: str = LOCAL_MODELS_DIR) -> str:
    """Download a model file from GCS to local directory.

    Returns local file path. Falls back to local if GCS is disabled.
    """
    local_path = os.path.join(local_dir, filename)

    if not USE_GCS:
        if os.path.exists(local_path):
            return local_path
        raise FileNotFoundError(f"Model not found locally: {local_path}")

    os.makedirs(local_dir, exist_ok=True)

    # Skip download if already cached locally
    if os.path.exists(local_path):
        return local_path

    client = _get_gcs_client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(f"{GCS_MODEL_PREFIX}{filename}")

    if not blob.exists():
        raise FileNotFoundError(f"Model not found in GCS: gs://{GCS_BUCKET}/{GCS_MODEL_PREFIX}{filename}")

    blob.download_to_filename(local_path)
    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f"✓ Downloaded {filename} from GCS ({size_mb:.1f} MB)")
    return local_path


def upload_model(local_path: str, filename: str = "") -> str:
    """Upload a model file to GCS.

    Returns GCS URI (gs://bucket/prefix/filename).
    """
    if not filename:
        filename = os.path.basename(local_path)

    if not USE_GCS:
        # Local mode: copy to models/current/
        dest = os.path.join(LOCAL_MODELS_DIR, filename)
        os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)
        if os.path.abspath(local_path) != os.path.abspath(dest):
            shutil.copy2(local_path, dest)
        return dest

    client = _get_gcs_client()
    bucket = client.bucket(GCS_BUCKET)
    blob_path = f"{GCS_MODEL_PREFIX}{filename}"
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)

    gcs_uri = f"gs://{GCS_BUCKET}/{blob_path}"
    print(f"✓ Uploaded {filename} to {gcs_uri}")
    return gcs_uri


def download_all_models(filenames: list, local_dir: str = LOCAL_MODELS_DIR) -> dict:
    """Download multiple model files. Returns {filename: local_path}."""
    results = {}
    for fname in filenames:
        try:
            results[fname] = download_model(fname, local_dir)
        except FileNotFoundError:
            print(f"⚠ Model not found: {fname}")
    return results


def upload_directory(local_dir: str, gcs_prefix: str = "") -> int:
    """Upload all files in a directory to GCS. Returns count of uploaded files."""
    if not USE_GCS:
        return 0

    prefix = gcs_prefix or GCS_MODEL_PREFIX
    count = 0
    client = _get_gcs_client()
    bucket = client.bucket(GCS_BUCKET)

    for root, _, files in os.walk(local_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel_path = os.path.relpath(local_path, local_dir)
            blob_path = f"{prefix}{rel_path}".replace("\\", "/")
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            count += 1

    print(f"✓ Uploaded {count} files to gs://{GCS_BUCKET}/{prefix}")
    return count


def archive_models(version_tag: str) -> None:
    """Archive current models to archive/version_tag/ in GCS."""
    if not USE_GCS:
        archive_dir = os.path.join(PROJECT_ROOT, "models", "archive", version_tag)
        os.makedirs(archive_dir, exist_ok=True)
        for fname in os.listdir(LOCAL_MODELS_DIR):
            src = os.path.join(LOCAL_MODELS_DIR, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(archive_dir, fname))
        print(f"📦 Archived to {archive_dir}")
        return

    client = _get_gcs_client()
    bucket = client.bucket(GCS_BUCKET)

    # Copy current/* to archive/version_tag/*
    blobs = bucket.list_blobs(prefix=GCS_MODEL_PREFIX)
    for blob in blobs:
        new_name = blob.name.replace(GCS_MODEL_PREFIX, f"archive/{version_tag}/")
        bucket.copy_blob(blob, bucket, new_name)

    print(f"📦 Archived to gs://{GCS_BUCKET}/archive/{version_tag}/")
