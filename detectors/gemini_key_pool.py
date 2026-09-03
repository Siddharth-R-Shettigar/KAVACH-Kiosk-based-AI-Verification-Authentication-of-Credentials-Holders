"""
Round-robin + fail-over pool for Gemini API keys.
Keys should come from different Google Cloud / AI Studio projects
to actually increase quota.
"""

import os
import time
import threading
from dotenv import load_dotenv

load_dotenv()

_lock = threading.Lock()
_index = 0
_cooldown_until = {}  # key_suffix -> unix time when usable again


def _load_keys():
    keys = []

    # Comma-separated list
    blob = os.getenv("GEMINI_API_KEYS", "") or ""
    for part in blob.split(","):
        k = part.strip()
        if k:
            keys.append(k)

    # Numbered keys GEMINI_API_KEY_1 ... _30
    for i in range(1, 31):
        k = os.getenv(f"GEMINI_API_KEY_{i}", "").strip()
        if k:
            keys.append(k)

    # Single classic key
    single = os.getenv("GEMINI_API_KEY", "").strip()
    if single:
        keys.append(single)

    # Deduplicate, keep order
    seen = set()
    out = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _key_id(key: str) -> str:
    return key[-6:] if len(key) >= 6 else key


def mark_key_cooling(key: str, seconds: float = 65.0):
    """After 429 / quota errors, rest this key briefly."""
    with _lock:
        _cooldown_until[_key_id(key)] = time.time() + seconds


def get_next_gemini_key():
    """
    Returns (api_key, index_in_pool) or (None, -1) if none available.
    Skips keys still in cooldown.
    """
    global _index
    keys = _load_keys()
    if not keys:
        return None, -1

    now = time.time()
    with _lock:
        n = len(keys)
        for _ in range(n):
            key = keys[_index % n]
            _index = (_index + 1) % n
            until = _cooldown_until.get(_key_id(key), 0)
            if now >= until:
                return key, (_index - 1) % n
        # All cooling — return next anyway (caller may still fail)
        key = keys[_index % n]
        _index = (_index + 1) % n
        return key, (_index - 1) % n


def gemini_model_name() -> str:
    return os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"