"""
tts_google.py

Replaces Groq Orpheus TTS with Google Cloud Text-to-Speech.

Orpheus only has English voices, so a Telugu/Hindi reply would come
out read in an English accent. This module picks an actual matching
voice per detected language (te-IN, hi-IN, en-IN) so the caller hears
natural pronunciation, not Tenglish read by an English mouth.

Voice names aren't hardcoded — they're looked up once at startup via
list_voices() and cached, since exact voice names/tiers (Standard,
Wavenet, Neural2, Chirp3-HD) vary by what's available in your project
and region. This also means if Google adds better voices later, you
automatically start using them without a code change.
"""

import logging
import time

from google.cloud import texttospeech

from .retry import with_retry

log = logging.getLogger("tts_google")

# Preferred voice tiers, in order — first match found for the language
# wins. Chirp3-HD/Neural2 sound most natural; Standard is the fallback
# that's virtually always available.
PREFERRED_TIER_ORDER = ["Chirp3-HD", "Neural2", "Wavenet", "Standard"]

# Fallback if a language has no explicit voice cached yet.
DEFAULT_LANGUAGE = "en-IN"

# Without an explicit deadline, a stalled network request can block
# this call forever without ever raising — with_retry() can't help
# against that since it only retries on an actual exception.
REQUEST_TIMEOUT_SEC = 20

_client = None
_voice_cache: dict[str, str] = {}  # language_code -> voice name


def _get_client():
    global _client
    if _client is None:
        _client = texttospeech.TextToSpeechClient()
    return _client


def _normalize_language_code(language_code: str) -> str:
    """
    BCP-47 canonical casing (lang lowercase, region uppercase), e.g.
    "en-in" -> "en-IN". Without this, "en-IN" (the greeting's hardcoded
    casing) and "en-in" (from language detection elsewhere) hit
    different _voice_cache keys, causing a "cache miss" — and a fresh
    list_voices() call plus its log line — every time the casing
    differs, even though it's the same language and the same voice
    was already picked moments earlier.
    """
    parts = language_code.split("-")
    if len(parts) == 2:
        return f"{parts[0].lower()}-{parts[1].upper()}"
    return language_code


def _pick_voice_for_language(language_code: str) -> str:
    """Looks up the best available voice for a language and caches it."""
    if language_code in _voice_cache:
        return _voice_cache[language_code]

    client = _get_client()
    try:
        response = client.list_voices(language_code=language_code, timeout=REQUEST_TIMEOUT_SEC)
    except Exception as e:
        log.error(f"[TTS] Could not list voices for {language_code}: {e}")
        _voice_cache[language_code] = ""
        return ""

    voices_by_tier = {}
    for voice in response.voices:
        for tier in PREFERRED_TIER_ORDER:
            if tier in voice.name:
                voices_by_tier.setdefault(tier, voice.name)

    for tier in PREFERRED_TIER_ORDER:
        if tier in voices_by_tier:
            chosen = voices_by_tier[tier]
            log.info(f"[TTS] Selected voice for {language_code}: {chosen}")
            _voice_cache[language_code] = chosen
            return chosen

    # No tiered match — just take whatever the API returned first.
    if response.voices:
        chosen = response.voices[0].name
        log.warning(f"[TTS] No preferred-tier voice for {language_code}, using {chosen}")
        _voice_cache[language_code] = chosen
        return chosen

    log.error(f"[TTS] No voices available at all for {language_code}")
    _voice_cache[language_code] = ""
    return ""


def synthesize(text: str, language_code: str = DEFAULT_LANGUAGE) -> bytes | None:
    """
    Synthesizes speech for `text` using a voice matching `language_code`
    (e.g. "te-IN", "hi-IN", "en-IN"). Returns LINEAR16 WAV bytes at
    16kHz (matching the rest of the pipeline), or None on failure.
    """
    language_code = _normalize_language_code(language_code)
    client = _get_client()
    voice_name = _pick_voice_for_language(language_code)

    if not voice_name:
        log.warning(f"[TTS] Falling back to {DEFAULT_LANGUAGE} — no voice for {language_code}")
        language_code = DEFAULT_LANGUAGE
        voice_name = _pick_voice_for_language(DEFAULT_LANGUAGE)

    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice_params = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        name=voice_name if voice_name else None,
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
    )

    try:
        t0 = time.time()
        response = with_retry(lambda: client.synthesize_speech(
            input=synthesis_input,
            voice=voice_params,
            audio_config=audio_config,
            timeout=REQUEST_TIMEOUT_SEC,
        ))
        log.info(f"[TTS] ({language_code}, {time.time()-t0:.2f}s)")
        return response.audio_content
    except Exception as e:
        log.error(f"[TTS] Google TTS failed: {e}")
        return None
