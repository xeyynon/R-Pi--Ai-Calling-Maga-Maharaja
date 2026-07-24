"""
retry.py

Minimal retry-with-backoff for transient Google Cloud API errors
(network blips, momentary 5xx). One retry after a short delay — not a
full backoff/jitter framework, since this is a single low-volume phone
line, not a high-QPS service. Each caller still wraps this in its own
try/except and returns its existing safe fallback if both attempts
fail, so this never changes what happens on a genuine, persistent
failure — it only recovers the rare one-off blip.
"""

import logging
import time

log = logging.getLogger("retry")


def with_retry(fn, attempts: int = 2, delay: float = 0.4):
    """Calls fn() up to `attempts` times, sleeping `delay`s between tries.
    Raises the last exception if every attempt fails."""
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            log.warning(f"[RETRY] attempt {attempt}/{attempts} failed: {e}")
            if attempt < attempts:
                time.sleep(delay)
    raise last_exc
