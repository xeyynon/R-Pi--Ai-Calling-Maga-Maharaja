"""
stt_google.py

Replaces Groq Whisper transcription with Google Cloud Speech-to-Text.

Key difference from the Groq version: instead of forcing STT_LANG="en"
(which would garble Telugu/Hindi speech), this requests transcription
with multiple candidate languages and lets Google's STT auto-detect
which one was actually spoken. The detected language code is returned
alongside the transcript, so the rest of the pipeline (LLM, TTS) can
respond in the same language the caller used.
"""

import logging
import time

from google.cloud import speech

log = logging.getLogger("stt_google")

# Candidate languages for auto-detection. Order matters slightly —
# first is the "primary" language hint, others are alternatives.
# BCP-47 codes: Telugu = te-IN, Hindi = hi-IN, English (India) = en-IN
LANGUAGE_CODE = "en-IN"
ALTERNATIVE_LANGUAGE_CODES = ["te-IN", "hi-IN"]

SAMPLE_RATE = 16000  # must match ai_caller_final2.py's SAMPLE_RATE

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = speech.SpeechClient()
    return _client


def transcribe(pcm: bytes) -> tuple[str, str]:
    """
    Transcribes raw 16kHz mono s16 PCM audio.

    Returns (transcript, language_code). language_code will be one of
    "en-IN", "te-IN", "hi-IN" (whichever Google detected), or ""
    on failure / no speech recognized.
    """
    client = _get_client()

    audio = speech.RecognitionAudio(content=pcm)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code=LANGUAGE_CODE,
        alternative_language_codes=ALTERNATIVE_LANGUAGE_CODES,
        enable_automatic_punctuation=True,
        model="latest_long",  # alternative_language_codes isn't supported by
                               # "phone_call" — only default/latest_long/
                               # latest_short/command_and_search allow it
    )

    try:
        t0 = time.time()
        response = client.recognize(config=config, audio=audio)
        elapsed = time.time() - t0
    except Exception as e:
        log.error(f"[STT] Google Speech-to-Text failed: {e}")
        return "", ""

    if not response.results:
        log.debug("[STT] No speech recognized.")
        return "", ""

    result = response.results[0]
    transcript = result.alternatives[0].transcript.strip()
    detected_lang = result.language_code or LANGUAGE_CODE

    log.info(f"[STT] '{transcript}' (lang={detected_lang}, {elapsed:.2f}s)")
    return transcript, detected_lang
