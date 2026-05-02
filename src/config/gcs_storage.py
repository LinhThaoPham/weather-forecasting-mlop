"""Google Cloud Storage helpers for model artifact sync."""
from __future__ import annotations

import os
from pathlib import Path

from google.cloud import storage

from src.config.gcp import GCS_BUCKET_NAME, GCS_MODELS_PREFIX, USE_GCS


def _client() -> storage.Client:
    return storage.Client()


def _bucket() -> storage.Bucket:
    return _client().bucket(GCS_BUCKET_NAME)


def upload_file(local_path: str, blob_path: str) -> None:
    if not USE_GCS:
        return
    path = Path(local_path)
    if not path.exists():
        raise FileNotFoundError(f"Local artifact not found: {local_path}")

    blob = _bucket().blob(blob_path)
    blob.upload_from_filename(str(path))


def download_file(blob_path: str, local_path: str) -> bool:
    if not USE_GCS:
        return False
    blob = _bucket().blob(blob_path)
    if not blob.exists():
        return False

    path = Path(local_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(path))
    return True


def sync_models_from_gcs(local_models_dir: str) -> int:
    """Download all model files from GCS models prefix to local directory."""
    if not USE_GCS:
        return 0

    bucket = _bucket()
    local_dir = Path(local_models_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    prefix = GCS_MODELS_PREFIX.rstrip("/") + "/"
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith("/"):
            continue
        filename = os.path.basename(blob.name)
        target = local_dir / filename
        blob.download_to_filename(str(target))
        downloaded += 1

    return downloaded


def upload_models_dir(local_models_dir: str) -> int:
    """Upload all files from local models/current to GCS models prefix."""
    if not USE_GCS:
        return 0

    local_dir = Path(local_models_dir)
    if not local_dir.exists():
        return 0

    bucket = _bucket()
    uploaded = 0
    prefix = GCS_MODELS_PREFIX.rstrip("/")
    for file_path in local_dir.glob("*"):
        if file_path.is_file():
            blob = bucket.blob(f"{prefix}/{file_path.name}")
            blob.upload_from_filename(str(file_path))
            uploaded += 1
    return uploaded
