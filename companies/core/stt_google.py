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
import queue
import threading
import time

from google.cloud import speech_v2

log = logging.getLogger("stt_google")

# 2026-07-31: made configurable (was a hardcoded "global") specifically
# to A/B chirp/chirp_2 against latest_long — chirp models don't exist
# in "global" at all (confirmed via Google's own regional-availability
# docs: chirp_2 is only in us-central1, europe-west4, asia-southeast1;
# neither asia-south1 nor asia-south2 support it). Testing chirp_2
# means setting BOTH of these together:
#   STT_MODEL=chirp_2 STT_LOCATION=asia-southeast1 python3 ai_caller_final3.py
STT_LOCATION = os.environ.get("STT_LOCATION", "global")

# ─────────────────────────────────────────────────────
# STREAMING CONFIG — 2026-07-30: this used to be one of two selectable
# presets. The other, `telephony` (8kHz, en-IN only), was a real,
# measured improvement for English after a spectral analysis proved
# the Bluetooth SCO link is narrowband (CVSD, 8kHz) — but it flatly
# rejected te-IN and mistagged Hindi under Devanagari when hi-IN was
# added alongside it. Deleted 2026-07-30 once Sarvam became the
# default STT provider (see stt_provider.py) and Google was demoted to
# a fallback/comparison option — at that point telephony's English-
# only ceiling made it strictly worse than Sarvam for the fallback
# role, while `multilang` (below) still covers all three languages
# Google can attempt at all, however imperfectly. If Google is ever
# reconsidered as primary, telephony's config is preserved in git
# history (search for GOOGLE_STT_CONFIG) rather than reintroduced here
# speculatively.
#
# 2026-07-31: STT_MODEL added so chirp/chirp_2 can be A/B'd against
# latest_long on real calls, same reasoning as the old GOOGLE_STT_CONFIG
# switch — an env var, not a code change, so nothing has to be "decided"
# to try it. NEITHER chirp NOR chirp_2 has been tested against this
# project's real captured audio yet — Google's docs confirm en-IN/hi-IN/
# te-IN are supported under both, but:
#   - chirp_2's STREAMING API (StreamingRecognize, what this file uses)
#     is documented as having "limited" language support vs batch, with
#     only single-language examples shown — whether passing all 3
#     candidates via language_codes (the way latest_long does) actually
#     works for chirp_2 in streaming mode is UNCONFIRMED. If it silently
#     only honors the first code, single-language accuracy may look
#     fine while multi-language callers get force-decoded, the exact
#     v1-vs-v2 bug this file's module docstring describes.
#   - chirp (v1) may not support streaming at all in every region.
# Verify against real captured audio (stt_regression_test.py) before
# trusting either on a live call — same discipline as every other STT
# change made in this project.
STREAMING_STT_MODEL = os.environ.get("STT_MODEL", "latest_long")
STREAMING_LANGUAGE_CODES = ["en-IN", "hi-IN", "te-IN"]
STREAMING_SAMPLE_RATE_HZ = 16000

_client = None


def _get_client():
    # 2026-07-31: a regional STT_LOCATION (needed for chirp/chirp_2,
    # which don't exist in "global" — see STT_LOCATION comment above)
    # is USELESS without also pointing the client at that region's own
    # endpoint. SpeechClient() with no client_options always talks to
    # the global endpoint regardless of what location the recognizer
    # path itself names — confirmed live: STT_LOCATION=asia-southeast1
    # alone produced "400 Expected resource location to be global, but
    # found asia-southeast1 in resource name" on every single request.
    global _client
    if _client is None:
        if STT_LOCATION == "global":
            _client = speech_v2.SpeechClient()
        else:
            _client = speech_v2.SpeechClient(
                client_options={"api_endpoint": f"{STT_LOCATION}-speech.googleapis.com"}
            )
    return _client


def warmup():
    """
    Forces the SpeechClient's credential loading and gRPC channel setup
    to happen at process boot instead of on the first real caller's
    utterance. Constructing the client is enough for the auth handshake
    to run — unlike TTS/LLM there's no free trivial STT call to make
    (any real request needs actual audio), so this doesn't fire a
    dummy recognize() call, just eagerly builds the client.
    """
    _get_client()
    log.info(
        "[STT] Client warmed. model=%s, location=%s, languages=%s, rate=%dHz",
        STREAMING_STT_MODEL, STT_LOCATION, STREAMING_LANGUAGE_CODES, STREAMING_SAMPLE_RATE_HZ,
    )


def _get_recognizer() -> str:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    return f"projects/{project}/locations/{STT_LOCATION}/recognizers/_"


# ─────────────────────────────────────────────────────
# STREAMING — overlaps recognition with the caller still speaking,
# instead of waiting for the full utterance + VAD hangover before
# transcription even starts. Verified against real audio (2026-07-29):
# the final result arrived ~10ms after the last audio frame was fed,
# vs. batch recognize()'s measured 0.75-3.4s starting from a cold
# standstill after the whole clip was already buffered.
# ─────────────────────────────────────────────────────

# How long finish() waits for the background streaming thread to
# deliver a final result after end-of-audio is signaled, before giving
# up. Generous: real calls finish in well under a second of this, but
# a stalled network request must not hang the recorder thread forever.
STREAMING_FINISH_TIMEOUT_SEC = 10

# Cached per distinct language_codes tuple. Normally just one entry
# (the full 3-language list) — a second appears once session-sticky
# language locking (ai_caller_final3.py, 2026-07-31) narrows a call
# down to a single confirmed language after the first turn.
_STREAMING_CONFIG_CACHE: dict[tuple[str, ...], "speech_v2.StreamingRecognitionConfig"] = {}


def _get_streaming_config(language_codes: tuple[str, ...]):
    cached = _STREAMING_CONFIG_CACHE.get(language_codes)
    if cached is None:
        cached = speech_v2.StreamingRecognitionConfig(
            config=speech_v2.RecognitionConfig(
                explicit_decoding_config=speech_v2.ExplicitDecodingConfig(
                    encoding=speech_v2.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=STREAMING_SAMPLE_RATE_HZ,
                    audio_channel_count=1,
                ),
                language_codes=list(language_codes),
                model=STREAMING_STT_MODEL,
                # No phrase-hint boosting (SpeechAdaptation): tried
                # 2026-07-30, broke every utterance in production —
                # "400 Recognizer does not support feature:
                # speech_adaptation_boost" against the wildcard "_"
                # recognizer this pipeline uses. Needs an explicitly
                # CREATED Recognizer resource to use at all — a bigger
                # change (recognizer lifecycle management) than this
                # was scoped for. See memory/commit history for the
                # full phrase-hint list if this is revisited properly.
            ),
            streaming_features=speech_v2.StreamingRecognitionFeatures(
                interim_results=True,
            ),
        )
        _STREAMING_CONFIG_CACHE[language_codes] = cached
    return cached


class StreamingSession:
    """
    One utterance's worth of streaming recognition. Started at speech
    onset (recorder_thread), fed frames as they're captured, told to
    finish once VAD hangover fires — all our own endpointing logic in
    recorder_thread is unchanged; this only changes WHEN transcription
    work happens (overlapped with capture) not HOW end-of-speech is
    decided.

    The actual bidirectional streaming_recognize() call runs in its own
    daemon thread, fed via a queue, so the calling thread (recorder_
    thread) is never blocked on network I/O — it just pushes frames and
    keeps reading the mic in its own loop exactly as before.
    """

    def __init__(self, language_codes: list[str] | None = None):
        # Defaults to the full 3-language candidate list. A caller
        # (ai_caller_final3.py, once a call's language is locked) can
        # instead pass a single-element list to restrict recognition to
        # just that language for the rest of the call — see module docs
        # on _get_streaming_config's per-language-tuple cache.
        self._language_codes = tuple(language_codes) if language_codes else tuple(STREAMING_LANGUAGE_CODES)
        self._queue: queue.Queue = queue.Queue()
        self._result_transcript = ""
        self._result_language = ""
        self._error = None
        # Latest interim (non-final) transcript, updated as Google STT
        # streams partials back — while the caller is still mid-
        # utterance, not just once they've finished. Lets a consumer
        # (recorder_thread) kick off speculative RAG retrieval on this
        # partial text before the final transcript is even ready,
        # instead of only starting retrieval after finish() returns.
        # Best-effort/no lock: a reader seeing a slightly stale value
        # is harmless here, there's no correctness requirement, only
        # "as fresh as convenient."
        self.latest_partial = ""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def feed(self, chunk: bytes) -> None:
        self._queue.put(chunk)

    def finish(self) -> tuple[str, str]:
        """Signals end-of-audio and waits for the final result. Returns
        (transcript, language_code), same contract as transcribe()."""
        self._queue.put(None)  # sentinel: no more audio coming
        self._thread.join(timeout=STREAMING_FINISH_TIMEOUT_SEC)
        if self._thread.is_alive():
            log.error("[STT-STREAM] Timed out waiting for final result.")
            return "", "ERROR"
        if self._error:
            log.error(f"[STT-STREAM] {self._error}")
            return "", "ERROR"
        return self._result_transcript, self._result_language

    def _requests(self):
        recognizer = _get_recognizer()
        yield speech_v2.StreamingRecognizeRequest(
            recognizer=recognizer,
            streaming_config=_get_streaming_config(self._language_codes),
        )
        while True:
            chunk = self._queue.get()
            if chunk is None:
                return
            yield speech_v2.StreamingRecognizeRequest(audio=chunk)

    def _run(self):
        client = _get_client()
        t0 = time.time()
        try:
            for response in client.streaming_recognize(requests=self._requests()):
                for result in response.results:
                    if not result.alternatives:
                        continue
                    alt = result.alternatives[0]
                    if result.is_final:
                        self._result_transcript = alt.transcript.strip()
                        self._result_language = result.language_code or "en-IN"
                    else:
                        self.latest_partial = alt.transcript.strip()
        except Exception as e:
            self._error = str(e)
            return

        elapsed = time.time() - t0
        if self._result_transcript:
            log.info(f"[STT-STREAM] '{self._result_transcript}' "
                      f"(lang={self._result_language}, {elapsed:.2f}s)")
        else:
            log.info(f"[STT-STREAM] No speech recognized ({elapsed:.2f}s)")


def start_streaming_session(language_code: str | None = None, is_stale=None) -> StreamingSession:
    """
    language_code=None (default): recognize against the full 3-language
    candidate list. Pass a single BCP-47 code (e.g. "hi-IN") to restrict
    recognition to just that language — used for session-sticky
    language locking once a call's language is confirmed.

    is_stale: accepted for interface parity with stt_sarvam.py's
    version (ai_caller_final3.py calls this through the shared
    stt_provider dispatcher without knowing which backend is active).
    Unused here — Google's streaming path has no concurrency semaphore
    to abandon early; each StreamingSession call goes out immediately.
    """
    return StreamingSession([language_code] if language_code else None)
