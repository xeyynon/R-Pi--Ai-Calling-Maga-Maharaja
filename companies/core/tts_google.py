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
import queue
import re
import threading
import time
from xml.sax.saxutils import escape as _xml_escape

from google.cloud import texttospeech
from google.cloud import texttospeech_v1beta1 as texttospeech_streaming

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

# Phone numbers and UPI IDs get read as one huge cardinal number
# ("nine billion...") by Google TTS's default number normalization,
# which is useless to a caller trying to note one down. 6+ consecutive
# digits is comfortably above any real product price in this business
# (max ₹1550, 4 digits) but safely catches 10-digit Indian mobile
# numbers/UPI IDs — so prices/weights/quantities keep reading as
# normal numbers, only long digit runs get spelled out one digit at
# a time.
_DIGIT_RUN_PATTERN = re.compile(r"\d{6,}")


def needs_digit_spelling(text: str) -> bool:
    """True if `text` contains a long digit run (phone number, UPI ID)
    that the SSML digit-by-digit trick applies to — streaming
    synthesis has no SSML support, so the caller should fall back to
    the blocking synthesize() for any sentence this returns True for."""
    return bool(_DIGIT_RUN_PATTERN.search(text))


def _to_ssml(text: str) -> str:
    """Wraps long digit runs in <say-as interpret-as="characters">
    so they're read digit-by-digit instead of as one giant number."""
    parts = []
    last_end = 0
    for m in _DIGIT_RUN_PATTERN.finditer(text):
        parts.append(_xml_escape(text[last_end:m.start()]))
        parts.append(f'<say-as interpret-as="characters">{m.group()}</say-as>')
        last_end = m.end()
    parts.append(_xml_escape(text[last_end:]))
    return f"<speak>{''.join(parts)}</speak>"


def _get_client():
    global _client
    if _client is None:
        _client = texttospeech.TextToSpeechClient()
    return _client


# All three languages this pipeline ever needs a voice for — warmed at
# boot so the first real caller in any of them doesn't pay for the
# list_voices() round trip + cache-miss the module docstring already
# claimed happened "at startup" but previously only happened lazily on
# each language's first real use.
_WARMUP_LANGUAGES = ("en-IN", "hi-IN", "te-IN")


def warmup():
    """
    Pre-populates the voice cache for every supported language and
    forces the TextToSpeechClient's credential/channel setup to happen
    at process boot, not on the first caller's first reply.

    Also warms the SEPARATE streaming client (texttospeech_v1beta1) —
    this is a genuinely different client object from the one above, so
    warming the default client alone left it cold. Measured: first
    streamed sentence of a whole process's life took 2.4s to first
    chunk instead of the ~0.25s every sentence after it got, entirely
    client/channel setup cost. A tiny real streaming call here (not
    just constructing the client) pays that cost once at boot instead
    of on a real caller's first streamed reply.
    """
    for lang in _WARMUP_LANGUAGES:
        _pick_voice_for_language(lang)
    log.info(f"[TTS] Client warmed, voices cached for {_WARMUP_LANGUAGES}.")

    try:
        synth = start_streaming_synthesis("hi", DEFAULT_LANGUAGE)
        for _ in synth.chunks():
            pass
        log.info(f"[TTS-STREAM] Client warmed (failed={synth.failed}).")
    except Exception as e:
        log.warning(f"[TTS-STREAM] Warmup call failed (non-fatal, first real call will just be cold): {e}")


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

    synthesis_input = texttospeech.SynthesisInput(ssml=_to_ssml(text))
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


# ─────────────────────────────────────────────────────
# STREAMING — first audio chunk arrives in ~0.25s instead of waiting
# for the whole sentence to synthesize (measured 0.86-1.6s on real
# calls for the blocking call above). Verified against real synthesis
# before writing this: same voice, PCM encoding (not LINEAR16 — that
# encoding is rejected for streaming, "Unsupported audio encoding"),
# 16kHz, headerless raw chunks matching what the rest of the pipeline
# already expects.
#
# StreamingSynthesisInput has no `ssml` field (only text/markup/
# multi_speaker_markup/prompt), so the digit-run spelling-out trick
# _to_ssml() does (reading a phone number/UPI ID digit-by-digit) can't
# be ported directly — the caller of this module falls back to the
# blocking synthesize() above for any sentence containing a long digit
# run, and only streams the common case that doesn't need it.
# ─────────────────────────────────────────────────────

SAMPLE_RATE = 16000

_streaming_client = None


def _get_streaming_client():
    global _streaming_client
    if _streaming_client is None:
        _streaming_client = texttospeech_streaming.TextToSpeechClient()
    return _streaming_client


class StreamingSynthesis:
    """
    Runs Google Cloud TTS's StreamingSynthesize call in its own daemon
    thread, exposing raw 16kHz mono s16 PCM chunks via a queue as they
    arrive. A consumer (playback_thread) can start playing the first
    chunk long before the rest of the sentence has finished
    synthesizing, instead of waiting on one blocking synthesize_speech
    call that only returns once the entire clip is ready.
    """

    def __init__(self, text: str, language_code: str):
        self._queue: queue.Queue = queue.Queue()
        self._error = None
        self._thread = threading.Thread(
            target=self._run, args=(text, language_code), daemon=True
        )
        self._thread.start()

    def chunks(self):
        """Generator yielding raw PCM chunks; ends when the stream
        finishes normally OR failed — check .failed after exhausting
        to tell the two apart."""
        while True:
            chunk = self._queue.get()
            if chunk is None:
                return
            yield chunk

    @property
    def failed(self) -> bool:
        return self._error is not None

    def _run(self, text: str, language_code: str):
        client = _get_streaming_client()
        language_code = _normalize_language_code(language_code)
        voice_name = _pick_voice_for_language(language_code)
        if not voice_name:
            language_code = DEFAULT_LANGUAGE
            voice_name = _pick_voice_for_language(DEFAULT_LANGUAGE)

        streaming_config = texttospeech_streaming.StreamingSynthesizeConfig(
            voice=texttospeech_streaming.VoiceSelectionParams(
                language_code=language_code,
                name=voice_name if voice_name else None,
            ),
            streaming_audio_config=texttospeech_streaming.StreamingAudioConfig(
                audio_encoding=texttospeech_streaming.AudioEncoding.PCM,
                sample_rate_hertz=SAMPLE_RATE,
            ),
        )

        def requests():
            yield texttospeech_streaming.StreamingSynthesizeRequest(
                streaming_config=streaming_config,
            )
            yield texttospeech_streaming.StreamingSynthesizeRequest(
                input=texttospeech_streaming.StreamingSynthesisInput(text=text),
            )

        t0 = time.time()
        n_chunks = 0
        try:
            for response in client.streaming_synthesize(requests()):
                if response.audio_content:
                    n_chunks += 1
                    self._queue.put(bytes(response.audio_content))
        except Exception as e:
            self._error = str(e)
            log.error(f"[TTS-STREAM] {e}")
        finally:
            self._queue.put(None)  # sentinel: no more chunks

        if not self._error:
            log.info(f"[TTS-STREAM] ({language_code}, {n_chunks} chunks, {time.time()-t0:.2f}s)")


def start_streaming_synthesis(text: str, language_code: str) -> StreamingSynthesis:
    return StreamingSynthesis(text, language_code)
