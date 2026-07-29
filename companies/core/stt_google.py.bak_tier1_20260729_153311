"""
stt_google.py

Google Cloud Speech-to-Text v2 with multi-language recognition.

Migrated from v1 to v2 because v1's alternative_language_codes field
is accepted but doesn't actually re-decode — it force-decodes through
the primary language model (en-IN), producing fluent-looking but wrong
transcripts for Hindi/Telugu speech.

v2's language_codes with the "latest_long" model in the "global"
location provides genuine multi-language recognition: the same audio
is evaluated against all candidate languages, and the best match is
returned. Tested against 6 real caller recordings — v2 correctly
captures Hindi code-switched words that v1 force-decoded as English
(e.g. "Mai order Kyon karun" vs v1's "My order Kyon karun").

Uses explicit_decoding_config (not auto_decoding_config): the caller
passes raw headerless LINEAR16 PCM straight from the recorder, not a
WAV/FLAC file with format markers. auto_decoding_config can only
identify encoding from a file's own header bytes — against headerless
PCM it fails every request with "Audio data does not appear to be in
a supported encoding" (caught live in production logs on 2026-07-28,
missed during dev testing because that testing was against on-disk
.wav files, which do have headers). Must declare the encoding
explicitly, matching what v1's RecognitionConfig(encoding=..., ...)
did.
"""

import logging
import os
import time

from google.cloud import speech_v2

from .retry import with_retry

log = logging.getLogger("stt_google")

CANDIDATE_LANGUAGES = ["en-IN", "hi-IN", "te-IN"]
STT_MODEL = "latest_long"
STT_LOCATION = "global"

SAMPLE_RATE = 16000

REQUEST_TIMEOUT_SEC = 20

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = speech_v2.SpeechClient()
    return _client


def _get_recognizer() -> str:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    return f"projects/{project}/locations/{STT_LOCATION}/recognizers/_"


def transcribe(pcm: bytes) -> tuple[str, str]:
    """
    Transcribes raw 16kHz mono s16 PCM audio using STT v2.

    Returns (transcript, language_code):
    - Real speech recognized: (text, one of "en-IN"/"te-IN"/"hi-IN")
    - No speech found (silence/noise): ("", "")
    - STT failed after retries: ("", "ERROR")
    """
    client = _get_client()
    recognizer = _get_recognizer()

    config = speech_v2.RecognitionConfig(
        explicit_decoding_config=speech_v2.ExplicitDecodingConfig(
            encoding=speech_v2.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
            audio_channel_count=1,
        ),
        language_codes=CANDIDATE_LANGUAGES,
        model=STT_MODEL,
    )

    try:
        t0 = time.time()
        response = with_retry(lambda: client.recognize(
            recognizer=recognizer,
            config=config,
            content=pcm,
            timeout=REQUEST_TIMEOUT_SEC,
        ))
        elapsed = time.time() - t0
    except Exception as e:
        log.error(f"[STT] Google Speech-to-Text v2 failed: {e}")
        return "", "ERROR"

    if not response.results:
        log.info(
            f"[STT] No speech recognized "
            f"({len(pcm) / (SAMPLE_RATE * 2):.2f}s audio, {elapsed:.2f}s)"
        )
        return "", ""

    result = response.results[0]
    transcript = result.alternatives[0].transcript.strip()
    detected_lang = result.language_code or "en-IN"

    log.info(f"[STT] '{transcript}' (lang={detected_lang}, {elapsed:.2f}s)")
    return transcript, detected_lang
