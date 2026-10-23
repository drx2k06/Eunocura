"""Object storage with a zero-key local fallback.

The project used to require Cloudflare R2 credentials just to boot. It no
longer does:

* R2 credentials present  → upload to the bucket, return the public URL.
* Anything missing         → write to `backend/data/uploads/` and return
  `{PUBLIC_BASE_URL}/files/<name>`, served by the app's own `/files` mount.

Callers keep using `upload_to_r2()` unchanged; the function name is historical.
`storage_mode()` is what `/healthz` reports.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import BinaryIO, Optional

from src.config import (
    PUBLIC_BASE_URL,
    R2_ACCESS_KEY,
    R2_BUCKET,
    R2_ENDPOINT,
    R2_PUBLIC_URL,
    R2_SECRET_KEY,
    UPLOAD_DIR,
)

_client = None
_client_error: Optional[str] = None


def storage_mode() -> str:
    """'r2' when fully configured, otherwise 'local'."""
    required = (R2_ENDPOINT, R2_ACCESS_KEY, R2_SECRET_KEY, R2_BUCKET)
    return "r2" if all(required) else "local"


def _get_client():
    global _client, _client_error
    if _client is not None:
        return _client
    if storage_mode() != "r2":
        return None
    try:
        import boto3  # imported lazily so a bare install still boots

        _client = boto3.client(
            "s3",
            endpoint_url=R2_ENDPOINT,
            aws_access_key_id=R2_ACCESS_KEY,
            aws_secret_access_key=R2_SECRET_KEY,
            region_name="auto",
        )
    except Exception as exc:  # noqa: BLE001 - degrade, never crash
        _client_error = str(exc)
        print(f"R2 unavailable ({exc}); falling back to local storage")
        _client = None
    return _client


def local_path(file_name: str) -> Path:
    """Absolute path of a file in the local fallback store."""
    return UPLOAD_DIR / Path(file_name).name


def upload_to_r2(file_content: BinaryIO, file_name: str) -> Optional[str]:
    """Store a file and return a URL the frontend can open.

    Returns None only if the write itself fails.
    """
    safe_name = Path(file_name).name
    if not safe_name:
        return None

    client = _get_client()
    if client is not None:
        try:
            client.upload_fileobj(
                file_content,
                R2_BUCKET,
                safe_name,
                ExtraArgs={"ContentType": "application/octet-stream"},
            )
            return f"{R2_PUBLIC_URL}/{safe_name}" if R2_PUBLIC_URL else None
        except Exception as exc:  # noqa: BLE001 - fall through to local disk
            print(f"R2 upload failed ({exc}); writing to local storage instead")

    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        destination = local_path(safe_name)
        with destination.open("wb") as out:
            file_content.seek(0)
            shutil.copyfileobj(file_content, out)
        return f"{PUBLIC_BASE_URL.rstrip('/')}/files/{safe_name}"
    except Exception as exc:  # noqa: BLE001
        print(f"Local storage write failed: {exc}")
        return None
