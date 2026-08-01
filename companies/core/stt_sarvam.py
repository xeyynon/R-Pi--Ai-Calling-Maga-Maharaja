"""
stt_sarvam.py

Sarvam AI speech-to-text integration — added 2026-07-30 as the third
option in a three-way STT comparison (see stt_provider.py), specifically
because Sarvam is purpose-built for Indian-language and code-switched
("Hinglish"/"Teluglish") speech, which is exactly where both Google
configs (telephony/en-IN and multilang/3-lang) have real, measured
weaknesses (see the 2026-07-30 optimization review, §3/§6).

Endpoint, auth, model name, and request/response shape below were
confirmed by reading Sarvam's own docs (docs.sarvam.ai) on 2026-07-30:
  - REST endpoint: POST https://api.sarvam.ai/speech-to-text
  - Auth header:   api-subscription-key: <key>
  - Request:       multipart form — file, model ("saaras:v3"),
                    mode ("transcribe" | "translate" | "verbatim" |
                    "translit" | "codemix")
  - Response JSON: {"request_id", "transcript", "language_code"}
  - Official SDK:  pip install -U sarvamai  ->  from sarvamai import SarvamAI

====================================================================
STILL NOT VERIFIED AGAINST A LIVE CALL. No API key was available in
the environment this was written in — the shape above is read
directly from Sarvam's documentation, not exercised against a real
response. Run this module directly (see __main__ block) against a
real key BEFORE trusting it on a live phone call — same lesson as the
Google phrase-hint regression earlier today: a config that LOOKS
right is not the same as one that has actually been called.
====================================================================

Sarvam ALSO has a genuine real-time streaming API (WebSocket, with
interim results and start/end-of-speech VAD signals) — and as of
2026-07-30 it's confirmed working and IS what StreamingSession uses
below, live-verified against a real key and real captured audio
before being wired in (the exact lesson from the earlier Google
phrase-hint regression: verify against a real call, not just docs).
Verified facts, not assumptions:
  - `AsyncSarvamAI(...).speech_to_text_streaming.connect(...)` accepts
    `language_code="unknown"` for auto-detect, same as the batch REST
    path — confirmed via the SDK's own connect() signature.
  - Audio chunks must be sent as RAW, CONTINUOUS PCM — base64-encoded
    — via `ws.transcribe(audio=b64_string, encoding="audio/wav",
    sample_rate=16000)`. Wrapping each individual chunk in its own WAV
    header (the natural-looking alternative) was tested side by side
    on the same real utterance and silently produced ZERO events —
    confirmed broken, not just unconfirmed.
  - `audio` must be a base64-encoded STRING (`AudioData.data` is a
    pydantic str field) — passing raw bytes raises a clear
    ValidationError, not a silent failure, which is how this was
    caught.
  - Sarvam runs its own VAD server-side (`vad_signals=True` emits
    START_SPEECH/END_SPEECH events) and returns a `data` message with
    `transcript`/`language_code` — in the live test, that final
    message arrived while audio was still being fed (simulated
    real-time), i.e. genuine overlap with capture, not batch-after-
    the-fact — the whole reason this upgrade is worth doing.

`mode="codemix"` is worth trying specifically for this project's
biggest open problem (code-switched Hindi/Telugu/English in one
utterance) — set SARVAM_MODE=codemix to test it; "transcribe" is the
safer default for comparison against the two Google presets.
"""

import asyncio
import base64
import io
import logging
import os
import queue
import threading
import time
import wave

from sarvamai import AsyncSarvamAI, SarvamAI

log = logging.getLogger("stt_sarvam")

SAMPLE_RATE = 16000  # matches the recorder's native capture rate —
                      # Sarvam takes 16kHz directly, no forced-8kHz
                      # workaround needed the way Google's telephony
                      # preset required.

SARVAM_MODEL = os.environ.get("SARVAM_MODEL", "saaras:v3")
# "transcribe" is the safe default; "codemix" is the mode most directly
# relevant to this project's real problem (code-switched speech) — see
# module docstring. Try both, they are NOT the same and may perform
# very differently on real caller audio.
SARVAM_MODE = os.environ.get("SARVAM_MODE", "transcribe")
# "unknown" = auto-detect across Sarvam's supported languages, same as
# the batch REST path (which never took a language_code at all — this
# is the streaming path's equivalent). SUPPORTED_LANGUAGE_CODES below
# still filters the result before it ever reaches the rest of the
# pipeline, exactly as it does for the batch path.
SARVAM_STREAMING_LANGUAGE_CODE = os.environ.get("SARVAM_STREAMING_LANGUAGE_CODE", "unknown")

REQUEST_TIMEOUT_SEC = 20
# How long finish() waits for the streaming session's background
# thread to deliver a result after end-of-audio is signaled. Generous
# for the same reason stt_google.py's equivalent constant is: real
# calls resolve in well under a second of this once the last chunk is
# fed, but a stalled connection must not hang the recorder forever.
# Raised from 10s given the concurrency semaphore below: with
# SARVAM_MAX_CONCURRENT_SESSIONS=1, an utterance can now legitimately
# spend real time just WAITING for a prior slow call's turn before its
# own work even starts — that queueing time must not itself look like
# a stall and report a spurious ERROR.
STREAMING_FINISH_TIMEOUT_SEC = 20
# After the last audio chunk is sent, how long to keep listening for a
# trailing final transcript before giving up and using whatever was
# last received. Sarvam's own server-side VAD often finalizes WHILE
# still receiving audio (verified: the live test's final message
# arrived before the simulated feed had finished sending), but this
# covers the case where the last bit of speech is right at the very
# end of the buffer.
STREAMING_GRACE_SEC = float(os.getenv("SARVAM_STREAMING_GRACE_SEC", "1.5"))

# 2026-07-30: recorder_thread's async-finish fix (see ai_caller_final3.py)
# correctly stopped blocking mic capture on a slow STT call, but it also
# removed the one thing that used to guarantee only ONE Sarvam call was
# ever in flight at a time — every utterance now spawns its own thread,
# each opening its own WebSocket connection. On a real call this
# produced a genuine, severe stall: one utterance's transcript took 53
# SECONDS to arrive, and a separate one hit "Timed out waiting for
# streaming result" after ~20s. Both are consistent with Sarvam (or the
# Pi's own network stack under several concurrent async event loops)
# queuing/throttling overlapping connections rather than truly running
# them in parallel. Bounding concurrency here directly targets that —
# a real caller's utterances are naturally sequential anyway (nobody
# has two genuine utterances in flight at once), so serializing Sarvam
# calls costs nothing real while preventing unbounded concurrent
# connections. Started conservative (1) since Sarvam's actual
# concurrent-connection limit for this account is unknown, with an
# explicit note to raise it once real-call testing showed it
# unnecessarily serializing genuinely distinct, back-to-back
# utterances — which is exactly what happened, twice, 2026-07-31: a
# short, low-energy "tentative" onset (background noise, not real
# speech) still spawns a full real Sarvam session, and a second
# genuine utterance just 5 seconds later had to queue 14.6s behind it
# for the ONE available slot. Raised to 2, which helped but was NOT
# enough: 2026-07-31, later the same day, a real call showed 3-4 onset-
# triggered captures landing within a ~10s window (the caller saying
# "Hello" twice after the bot's reply was slow enough that they
# repeated themselves — a real, observed vicious cycle: slow reply ->
# caller repeats -> more concurrent STT load -> even slower) produced
# stt_wait_ms=8129 on one turn and a full 20s timeout on another. Raised
# to 4 as an immediate step, then actually looked up Sarvam's real
# documented limit instead of guessing again (docs.sarvam.ai/api/
# getting-started/ratelimits): STT WebSocket streaming concurrency is
# 20 (Starter plan) or 100 (Pro/Business), account-wide. Every number
# used here (1, 2, 4) was self-imposed and nowhere near the real
# ceiling — this project's own artificially low limit was the
# bottleneck all along, not Sarvam's actual capacity. One documented
# nuance: the guard reacts to how FAST connections open, not just how
# many are held open at once (rapid bursts can be rejected — WebSocket
# close code 1003 — well under the stated ceiling; the same number
# opened a few hundred ms apart succeeds). Real observed load on this
# single phone line is 3-4 sessions over ~10s, nowhere near a rapid
# burst, so this should be safe. Set with real headroom (half the
# Starter-tier ceiling) instead of nudging up reactively again — if a
# genuine 1003 rejection ever appears in the logs (a distinct failure
# shape from a client-side "Timed out" — that one's just our own
# STREAMING_FINISH_TIMEOUT_SEC giving up, not a Sarvam rejection), that
# would be the actual burst-rate guard, not this ceiling.
SARVAM_MAX_CONCURRENT_SESSIONS = int(os.getenv("SARVAM_MAX_CONCURRENT_SESSIONS", "10"))
_sarvam_concurrency = threading.Semaphore(SARVAM_MAX_CONCURRENT_SESSIONS)

# 2026-07-30: real-call testing showed Sarvam's auto language-ID
# confidently hallucinating short, grammatically real-looking phrases
# in languages nobody was speaking, from what was almost certainly
# noise/breathing/silence — e.g. 'আচ্ছা।' tagged bn-IN, 'बघा.' tagged
# mr-IN, 'બાકીની જ રીતે છે.' tagged gu-IN, 'ಹ್ಞೂ.' tagged kn-IN, all in
# one test call, with the Gujarati one actually making the bot reply
# in Gujarati to a caller who never said a word of it. This business
# only ever serves English/Hindi/Telugu callers — there is no
# legitimate reason a real utterance should ever need a 5th language.
# Restricting accepted results to the business's actual supported
# languages is a cheap, well-targeted proxy: every CROSS-language
# hallucination observed landed on an unsupported language tag, and
# every genuine utterance (even short ones like "Hello"/"Okay")
# landed on one of these three. Anything else is treated as no speech
# — same downstream handling as silence, not a wrong-language reply.
SUPPORTED_LANGUAGE_CODES = {"en-in", "hi-in", "te-in"}

# 2026-07-30: the language-code filter above only catches
# CROSS-language hallucinations. A regression test against real
# captured audio (stt_regression_test.py) found a second failure mode
# it can't touch: silence/noise hallucinated into a plausible-looking
# phrase in a SUPPORTED language — e.g. 'इस तक कि।' (Hindi filler,
# lang_probability=0.118) and a Malayalam-script hallucination tagged
# en-IN-adjacent nonsense (lang_probability=0.14), both from utterances
# where a DIFFERENT STT engine (Google) independently found "no
# speech" in the same audio. This field is genuinely exposed by
# Sarvam's API (SpeechToTextTranscriptionData.language_probability) —
# not a guessed heuristic like energy or word-count, both of which
# were checked first and rejected for real, demonstrated overlap with
# legitimate quiet speech.
#
# Threshold recalibrated 2026-07-30 after a real call showed 0.5 was
# too aggressive: it discarded two GENUINE utterances — "Repeat in
# Hindi" (0.41) and "मगा महाराजा क्या है?" ("what is Maga Maharaja?",
# 0.39) — silently, forcing the caller to repeat themselves. Combined
# evidence now available: confirmed hallucinations score 0.118-0.14;
# confirmed genuine-but-rejected utterances score 0.39-0.41; confirmed
# genuine, correctly-accepted utterances score 1.0. 0.25 sits in the
# real gap between the hallucination cluster and the legitimate-content
# cluster — catches the demonstrated hallucinations without repeating
# the false-positive that just happened. Still an approximation with
# only a handful of data points either side; revisit if either failure
# mode recurs.
SARVAM_MIN_LANGUAGE_PROBABILITY = float(os.getenv("SARVAM_MIN_LANGUAGE_PROBABILITY", "0.25"))

_client = None


def _get_client() -> SarvamAI:
    # Read fresh, not frozen at import time — matches stt_google.py's
    # _get_recognizer() pattern. SARVAM_API_KEY has a local_config.py
    # fallback wired in ai_caller_final3.py that runs AFTER this
    # module's own top-of-file imports; freezing it as a module-level
    # constant here (the original version of this file did) reads ""
    # every time, since the fallback hasn't landed in os.environ yet
    # by then — a real bug caught by boot-testing on 2026-07-30.
    global _client
    api_key = os.environ.get("SARVAM_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "SARVAM_API_KEY is not set. See the setup instructions given "
            "alongside this integration."
        )
    if _client is None:
        _client = SarvamAI(api_subscription_key=api_key)
    return _client


def warmup():
    """
    Constructing the client only validates that a key is PRESENT, not
    that it's valid or that the endpoint is reachable — there is no
    free trivial call the way Google's client construction doubles as
    an auth handshake. A genuine end-to-end check needs real audio;
    run this module directly against a captured WAV first.
    """
    _get_client()
    log.info(
        f"[STT-SARVAM] Key present, model={SARVAM_MODEL} mode={SARVAM_MODE} "
        f"streaming_language={SARVAM_STREAMING_LANGUAGE_CODE}."
    )


def _pcm_to_wav_bytes(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def _call_sarvam(wav_bytes: bytes) -> tuple[str, str]:
    """
    One-shot batch REST call via the official SDK — used only by the
    __main__ smoke-test below (run this module directly against a real
    WAV to sanity-check the API key/model before trusting a live call).
    The real call path uses StreamingSession, not this. Returns
    (transcript, language_code): "" transcript means no speech
    recognized; "ERROR" language_code means the request itself failed
    (network, auth, bad audio, etc.).
    """
    client = _get_client()
    try:
        t0 = time.time()
        response = client.speech_to_text.transcribe(
            file=io.BytesIO(wav_bytes),
            model=SARVAM_MODEL,
            mode=SARVAM_MODE,
        )
        elapsed = time.time() - t0
    except Exception as e:
        log.error(f"[STT-SARVAM] Request failed: {e}")
        return "", "ERROR"

    # The SDK returns a response object (not raw JSON) — support both
    # attribute access (the documented shape) and dict access, in case
    # the installed SDK version differs from what the docs describe.
    transcript = (getattr(response, "transcript", None)
                  or (response.get("transcript") if isinstance(response, dict) else None)
                  or "").strip()
    language_code = (getattr(response, "language_code", None)
                      or (response.get("language_code") if isinstance(response, dict) else None)
                      or "")

    language_probability = (getattr(response, "language_probability", None)
                             or (response.get("language_probability") if isinstance(response, dict) else None))

    if transcript and language_code.lower() not in SUPPORTED_LANGUAGE_CODES:
        log.warning(
            f"[STT-SARVAM] Discarding '{transcript}' — tagged "
            f"{language_code!r}, not one of this business's supported "
            f"languages (likely a hallucination from noise, not real "
            f"speech in an unsupported language)."
        )
        return "", ""

    if transcript and language_probability is not None and language_probability < SARVAM_MIN_LANGUAGE_PROBABILITY:
        log.warning(
            f"[STT-SARVAM] Discarding '{transcript}' — "
            f"language_probability={language_probability:.2f} is below "
            f"{SARVAM_MIN_LANGUAGE_PROBABILITY} (likely a same-language "
            f"hallucination from noise, not confident real speech)."
        )
        return "", ""

    if transcript:
        log.info(f"[STT-SARVAM] '{transcript}' (lang={language_code}, {elapsed:.2f}s)")
    else:
        log.info(f"[STT-SARVAM] No speech recognized ({elapsed:.2f}s)")
    return transcript, language_code


class StreamingSession:
    """
    Genuine streaming, via Sarvam's WebSocket API — see module
    docstring for what was verified before this was trusted. Presents
    the same feed()/finish()/latest_partial interface as
    stt_google.py's StreamingSession: a background thread owns its own
    asyncio event loop (mirroring how the Google version owns a gRPC-
    consuming thread), fed via a plain queue.Queue so recorder_thread
    (a normal, non-async thread) never has to know asyncio exists.

    latest_partial updates from Sarvam's own "data" messages as they
    arrive — in the live test this was the same message as the final
    transcript (Sarvam's "transcribe" mode did not appear to emit
    growing word-by-word partials the way Google's interim_results
    does), but it's still real, and still lets speculative RAG fire
    earlier than waiting for finish() — an improvement over the old
    batch version, where latest_partial never populated at all.
    """

    def __init__(self, language_code: str | None = None, is_stale=None):
        # Defaults to "unknown" (auto-detect across all supported
        # languages). A caller (ai_caller_final3.py, once a call's
        # language is locked) can instead pass a specific BCP-47 code
        # (e.g. "hi-IN") to skip auto-detect for the rest of the call.
        self._language_code = language_code or SARVAM_STREAMING_LANGUAGE_CODE
        # 2026-07-31: real call log showed background-noise onset
        # triggers spawning several StreamingSessions in a row, each
        # queueing for the single SARVAM_MAX_CONCURRENT_SESSIONS slot —
        # and NONE of them got cancelled when the call that spawned
        # them ended. Each one still had to wait its full turn (up to
        # STREAMING_FINISH_TIMEOUT_SEC) before releasing the slot,
        # compounding into a backlog that measurably delayed the NEXT
        # call's own STT (observed: 34.7s/28.8s/25.0s/24.7s waits
        # logged AFTER the next call had already started). is_stale, if
        # given, is polled while waiting for the slot — a session whose
        # call has already ended abandons immediately instead of
        # consuming its turn in the queue.
        self._is_stale = is_stale
        self._sync_queue = queue.Queue()  # feed() is called from recorder_thread (sync)
        self._result_transcript = ""
        self._result_language = ""
        self._result_language_probability = 1.0  # absent field => trust it
        self._error = None
        self.latest_partial = ""
        self._thread = threading.Thread(target=self._run_thread, daemon=True)
        self._thread.start()

    def feed(self, chunk: bytes) -> None:
        self._sync_queue.put(chunk)

    def finish(self) -> tuple[str, str]:
        self._sync_queue.put(None)  # sentinel: no more audio coming
        self._thread.join(timeout=STREAMING_FINISH_TIMEOUT_SEC)
        if self._thread.is_alive():
            log.error("[STT-SARVAM] Timed out waiting for streaming result.")
            return "", "ERROR"
        if isinstance(self._error, str) and self._error.startswith("stale"):
            # Abandoned while waiting for a concurrency slot because its
            # call already ended — not a real failure, and "ERROR"
            # would make the caller speak an apology into a call that's
            # already over (and get discarded downstream anyway, but
            # only after wasting a TTS call). Treat it like clean
            # silence instead.
            return "", ""
        if self._error:
            log.error(f"[STT-SARVAM] {self._error}")
            return "", "ERROR"

        transcript = self._result_transcript
        language_code = self._result_language
        if transcript and language_code.lower() not in SUPPORTED_LANGUAGE_CODES:
            log.warning(
                f"[STT-SARVAM] Discarding '{transcript}' — tagged "
                f"{language_code!r}, not one of this business's "
                f"supported languages (likely a hallucination from "
                f"noise, not real speech in an unsupported language)."
            )
            return "", ""
        if transcript and self._result_language_probability < SARVAM_MIN_LANGUAGE_PROBABILITY:
            log.warning(
                f"[STT-SARVAM] Discarding '{transcript}' — "
                f"language_probability={self._result_language_probability:.2f} is "
                f"below {SARVAM_MIN_LANGUAGE_PROBABILITY} (likely a same-language "
                f"hallucination from noise, not confident real speech)."
            )
            return "", ""
        if transcript:
            log.info(f"[STT-SARVAM] '{transcript}' (lang={language_code}, streaming)")
        else:
            log.info("[STT-SARVAM] No speech recognized (streaming)")
        return transcript, language_code

    def _run_thread(self):
        t_wait0 = time.time()
        acquired = False
        while True:
            if self._is_stale is not None and self._is_stale():
                log.info(
                    "[STT-SARVAM] Abandoning session while waiting for a "
                    "streaming slot — its call already ended."
                )
                self._error = "stale — call ended before a streaming slot freed up"
                return
            if _sarvam_concurrency.acquire(timeout=0.5):
                acquired = True
                break
        wait_s = time.time() - t_wait0
        if wait_s > 0.5:
            log.warning(
                f"[STT-SARVAM] Waited {wait_s:.1f}s for a free streaming "
                f"slot (SARVAM_MAX_CONCURRENT_SESSIONS={SARVAM_MAX_CONCURRENT_SESSIONS}) "
                f"— another utterance's Sarvam call was still in flight."
            )
        try:
            asyncio.run(self._run_async())
        except Exception as e:
            self._error = str(e)
        finally:
            _sarvam_concurrency.release()

    async def _run_async(self):
        api_key = os.environ.get("SARVAM_API_KEY", "")
        if not api_key:
            raise RuntimeError("SARVAM_API_KEY is not set.")

        client = AsyncSarvamAI(api_subscription_key=api_key)
        loop = asyncio.get_event_loop()

        # 2026-08-01: REQUEST_TIMEOUT_SEC was declared but never
        # actually applied to anything — the WS handshake below had no
        # timeout of its own. STREAMING_FINISH_TIMEOUT_SEC (used below)
        # only bounds send/receive AFTER the connection is already
        # open; a stall during the handshake itself (a silent packet
        # drop, not an immediate refusal) could block here far past
        # 20s while still holding this session's _sarvam_concurrency
        # slot, since that slot is only released in the outer
        # `finally` once _run_async actually returns. Left unbounded,
        # a sustained connectivity blip could leak enough slots to
        # exhaust the whole pool for the rest of the call.
        ctx = client.speech_to_text_streaming.connect(
            model=SARVAM_MODEL,
            mode=SARVAM_MODE,
            language_code=self._language_code,
            high_vad_sensitivity=True,
            vad_signals=True,
        )
        try:
            ws = await asyncio.wait_for(ctx.__aenter__(), timeout=REQUEST_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            log.error(f"[STT-SARVAM] Connection handshake timed out after {REQUEST_TIMEOUT_SEC}s")
            raise
        try:
            sender_done = asyncio.Event()

            async def sender():
                while True:
                    # feed() runs on recorder_thread (sync) — bridge via
                    # a blocking Queue.get() offloaded to a worker thread
                    # so it doesn't block this event loop.
                    chunk = await loop.run_in_executor(None, self._sync_queue.get)
                    if chunk is None:
                        break
                    audio_b64 = base64.b64encode(chunk).decode("ascii")
                    try:
                        await ws.transcribe(audio=audio_b64, encoding="audio/wav", sample_rate=SAMPLE_RATE)
                    except Exception as e:
                        log.error(f"[STT-SARVAM] Streaming send failed: {e}")
                        break
                sender_done.set()

            # 2026-07-30: `async for message in ws` never terminates on
            # its own — Sarvam doesn't close the socket once it's sent
            # a final transcript, it just waits for more. An earlier
            # version of this method waited on that loop with a plain
            # timeout, which meant EVERY utterance paid the full
            # STREAMING_GRACE_SEC unconditionally, even when the
            # transcript had already arrived — measured 2.35-7.04s per
            # utterance that way, actually SLOWER than the old batch
            # REST path. Fix: an Event set the instant a transcript
            # arrives, raced against actually finishing the send —
            # only fall through to the grace-period wait for the
            # genuine "nothing came back" case (silence/no speech).
            got_transcript = asyncio.Event()

            async def receiver():
                async for message in ws:
                    if message.type == "data":
                        transcript = getattr(message.data, "transcript", None)
                        if transcript:
                            self._result_transcript = transcript
                            self.latest_partial = transcript
                            lang = getattr(message.data, "language_code", None)
                            if lang:
                                self._result_language = lang
                            prob = getattr(message.data, "language_probability", None)
                            if prob is not None:
                                self._result_language_probability = prob
                            got_transcript.set()

            send_task = asyncio.create_task(sender())
            recv_task = asyncio.create_task(receiver())
            got_transcript_task = asyncio.create_task(got_transcript.wait())

            try:
                await asyncio.wait_for(
                    asyncio.gather(send_task, got_transcript_task),
                    timeout=STREAMING_FINISH_TIMEOUT_SEC - STREAMING_GRACE_SEC - 1,
                )
            except asyncio.TimeoutError:
                if not send_task.done():
                    send_task.cancel()
                if not got_transcript.is_set():
                    try:
                        await asyncio.wait_for(got_transcript.wait(), timeout=STREAMING_GRACE_SEC)
                    except asyncio.TimeoutError:
                        pass  # genuinely no speech — finish() returns "" as usual

            recv_task.cancel()
            try:
                await recv_task
            except (asyncio.CancelledError, Exception):
                pass
        finally:
            await ctx.__aexit__(None, None, None)


def start_streaming_session(language_code: str | None = None, is_stale=None) -> StreamingSession:
    """
    language_code=None (default): "unknown", Sarvam auto-detects across
    all supported languages. Pass a specific BCP-47 code (e.g. "hi-IN")
    to skip auto-detect — used for session-sticky language locking once
    a call's language is confirmed. Confirmed working on a real call
    2026-07-31 (session-sticky en-IN held for the whole call).

    is_stale: optional zero-arg callable, polled while this session
    waits for a free SARVAM_MAX_CONCURRENT_SESSIONS slot. If it returns
    True, the session abandons the wait immediately instead of
    consuming its turn in the queue — see StreamingSession's docstring
    for the real backlog this fixes.
    """
    return StreamingSession(language_code, is_stale)


if __name__ == "__main__":
    # Quick end-to-end check against a real Sarvam API key, using one
    # of the WAV files already captured in the 2026-07-30 STT
    # diagnostic (see stt_diagnose.py) — reuses real caller audio
    # instead of needing a fresh call just to smoke-test this
    # integration. Usage:
    #   SARVAM_API_KEY=sk_... python3 stt_sarvam.py /path/to/utt.wav
    #   SARVAM_API_KEY=sk_... SARVAM_MODE=codemix python3 stt_sarvam.py /path/to/utt.wav
    import sys

    if len(sys.argv) < 2:
        print("Usage: SARVAM_API_KEY=... python3 stt_sarvam.py <wav_file>")
        sys.exit(1)

    with open(sys.argv[1], "rb") as f:
        wav_bytes = f.read()
    warmup()
    transcript, lang = _call_sarvam(wav_bytes)
    print(f"transcript={transcript!r} language_code={lang!r}")
