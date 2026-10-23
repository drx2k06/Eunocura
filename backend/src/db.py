"""MongoDB client.

Two rules learned the hard way:

1. Constructing this module must never raise. A judge with no MongoDB and no
   `.env` gets a booting server that reports `degraded` on `/healthz`, not a
   stack trace.
2. Connection is lazy (pymongo connects on first operation), so a missing
   database only breaks the routes that actually need data.
"""

import os
from datetime import datetime
from typing import Any

from pymongo import AsyncMongoClient, server_api

from src.config import MONGODB_URI

# Set MONGODB_ALLOW_INVALID_CERTS=false in .env for production (default: true for development)
ALLOW_INVALID_CERTS = os.getenv("MONGODB_ALLOW_INVALID_CERTS", "true").lower() == "true"

# A local default keeps the process alive when nothing is configured. `/healthz`
# reports `mongo.configured: false` so nobody mistakes it for a working DB.
DEFAULT_URI = "mongodb://localhost:27017/eunocura"
MONGO_CONFIGURED = bool(MONGODB_URI)
URI = MONGODB_URI or DEFAULT_URI

connection_kwargs = {
    "server_api": server_api.ServerApi(version="1", strict=False, deprecation_errors=True),
    # Never let a missing database stall a request (or the demo) for 30 seconds.
    "serverSelectionTimeoutMS": 1500,
    "connectTimeoutMS": 1500,
    "socketTimeoutMS": 4000,
}

# Allow invalid certificates for development (MongoDB Atlas SSL certificate verification)
if ALLOW_INVALID_CERTS:
    connection_kwargs["tlsAllowInvalidCertificates"] = True

try:
    client = AsyncMongoClient[Any](URI, **connection_kwargs)
    db = client.get_database("InterIIIT")

    appointment_collection = db.get_collection("appointments")
    user_collection = db.get_collection("users")
    chat_history_collection = db.get_collection("chat_histories")
except Exception as exc:  # noqa: BLE001 - never block import
    print(f"WARNING: MongoDB client could not be created ({exc}). "
          "Data routes will fail; /healthz will report degraded.")


async def ping(timeout_ms: int = 1500) -> bool:
    """True when the server answers. Used by /healthz and preflight."""
    try:
        await client.admin.command("ping", maxTimeMS=timeout_ms)
        return True
    except Exception:  # noqa: BLE001
        return False


async def test_connection():
    try:
        await client.admin.command("ping")
        print("MongoDB connection successful!")
    except Exception as e:
        print(f"MongoDB connection test failed: {e}")
