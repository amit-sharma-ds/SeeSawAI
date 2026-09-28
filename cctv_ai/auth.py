"""API key protection for the HTTP API.

Keys come from two places, and both are read live so the server never needs a restart:

  * the CCTV_API_KEYS environment variable, comma separated
  * the api_keys.txt file in the project folder, one key per line with an optional label

With no keys configured the API stays open, which is fine on your own PC. As soon as one
key exists, every endpoint except /health and the docs pages needs a valid key, sent as
an `X-API-Key` header or, for browser <img> stream tags, an `api_key` query parameter.
"""
from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Dict, Optional

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader, APIKeyQuery

KEY_FILE = Path(os.environ.get("CCTV_API_KEY_FILE", Path(__file__).resolve().parent.parent / "api_keys.txt"))
OPEN_PATHS = {"/", "/health", "/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}

header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False, description="API key issued with make_api_key.py")
query_scheme = APIKeyQuery(name="api_key", auto_error=False, description="Same key as a query parameter, for <img> stream tags")

_file_cache: Dict[str, object] = {"token": None, "keys": {}}


def _keys_from_env() -> Dict[str, str]:
    raw = os.environ.get("CCTV_API_KEYS", "")
    return {part.strip(): "env" for part in raw.split(",") if part.strip()}


def _keys_from_file() -> Dict[str, str]:
    try:
        stat = KEY_FILE.stat()
    except FileNotFoundError:
        _file_cache.update(token=None, keys={})
        return {}
    token = (stat.st_mtime, stat.st_size)
    if _file_cache["token"] != token:
        keys: Dict[str, str] = {}
        for line in KEY_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            keys[parts[0]] = parts[1].strip() if len(parts) > 1 else ""
        _file_cache.update(token=token, keys=keys)
    return dict(_file_cache["keys"])  # type: ignore[arg-type]


def load_keys() -> Dict[str, str]:
    """All valid keys mapped to their label."""
    keys = _keys_from_file()
    keys.update(_keys_from_env())
    return keys


def auth_enabled() -> bool:
    return bool(load_keys())


def generate_key() -> str:
    return secrets.token_urlsafe(32)


def add_key(label: str = "") -> str:
    """Create a new key, append it to the key file and return it."""
    key = generate_key()
    KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with KEY_FILE.open("a", encoding="utf-8") as fh:
        fh.write(f"{key} {label}".rstrip() + "\n")
    return key


def _matches(provided: str, keys: Dict[str, str]) -> Optional[str]:
    candidate = provided.encode("utf-8")
    for key, label in keys.items():
        if secrets.compare_digest(candidate, key.encode("utf-8")):
            return label
    return None


async def require_api_key(
    request: Request,
    header_key: Optional[str] = Security(header_scheme),
    query_key: Optional[str] = Security(query_scheme),
) -> Optional[str]:
    """FastAPI dependency. Returns the key's label, or None when auth is off or the path is open."""
    keys = load_keys()
    if not keys or request.url.path in OPEN_PATHS:
        return None
    provided = header_key or query_key
    if provided:
        label = _matches(provided, keys)
        if label is not None:
            return label
    raise HTTPException(
        status_code=401,
        detail="missing or invalid API key: send it as the X-API-Key header or the api_key query parameter",
        headers={"WWW-Authenticate": "ApiKey"},
    )
