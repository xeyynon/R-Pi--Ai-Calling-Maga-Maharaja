"""
stt_provider.py

Thin dispatcher selecting which STT BACKEND the pipeline talks to, via
the STT_PROVIDER env var: "sarvam" (default, v3.0) or "google" (kept
as a fallback/comparison option — see stt_google.py's module comment
for why its old "telephony" preset was deleted).

2026-07-30: added so the real STT options this project has evidence
for — Google's multilang/16kHz preset (3-language, en-IN/hi-IN/te-IN)
and Sarvam (purpose-built for Indian-language/code-switched speech,
and the one of the two that has correctly recognized Telugu) — can be
pointed at, one at a time, by an env var, and A/B'd on real calls
without editing code.
Exposes the exact same interface every backend module does
(warmup/start_streaming_session), so ai_caller_final3.py only ever
imports from here, never from a specific backend directly.

Default set to "sarvam" for v3.0 (Sarvam STT + Google LLM + Google
TTS) per the version plan: v1 was the offline/Groq testing phase, v2
was Google STT+TTS end to end, v3.0 keeps Google for the LLM and TTS
but moves STT to Sarvam. A Sarvam TTS evaluation (and its costing) is
a deliberately separate, later decision — not part of v3.0.
"""

import logging
import os

log = logging.getLogger("stt_provider")

STT_PROVIDER = os.environ.get("STT_PROVIDER", "sarvam").lower()

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

warmup = _backend.warmup
start_streaming_session = _backend.start_streaming_session
