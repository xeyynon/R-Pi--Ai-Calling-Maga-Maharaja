"""
stt_provider.py

Thin dispatcher selecting which STT BACKEND the pipeline talks to, via
the STT_PROVIDER env var: "google" (default) or "sarvam". This is a
separate axis from GOOGLE_STT_CONFIG (inside stt_google.py), which
selects between Google's own "telephony" and "multilang" presets —
that switch only matters when STT_PROVIDER=google.

2026-07-30: added so the three real STT options this project has
evidence for — Google's new telephony/8kHz preset (best measured
English accuracy so far), Google's original multilang/16kHz preset
(the 3-language design used all session before that), and Sarvam
(untested against this pipeline, but purpose-built for Indian-
language/code-switched speech) — can be pointed at, one at a time, by
an env var, and A/B'd on real calls without editing code. Exposes the
exact same interface every backend module does
(transcribe/warmup/start_streaming_session), so ai_caller_final2.py
only ever imports from here, never from a specific backend directly.
"""

import logging
import os

log = logging.getLogger("stt_provider")

STT_PROVIDER = os.environ.get("STT_PROVIDER", "google").lower()

if STT_PROVIDER == "sarvam":
    from . import stt_sarvam as _backend
elif STT_PROVIDER == "google":
    from . import stt_google as _backend
else:
    raise ValueError(
        f"STT_PROVIDER={STT_PROVIDER!r} is not a known backend "
        f"(expected 'google' or 'sarvam')"
    )

log.info(f"[STT-PROVIDER] Using backend: {STT_PROVIDER}")

transcribe = _backend.transcribe
warmup = _backend.warmup
start_streaming_session = _backend.start_streaming_session
