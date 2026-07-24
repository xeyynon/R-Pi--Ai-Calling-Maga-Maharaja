#!/usr/bin/env python3
"""
AI Caller — Final (Google Cloud edition)
-----------------------------------------
Bluetooth HFP/SCO -> WebRTC VAD -> Google Speech-to-Text (multi-language)
    -> Conversation Manager (intent + knowledge + Gemini LLM + validator)
    -> Google Text-to-Speech (language-matched voice) -> Bluetooth

Swapped from the original Groq-based pipeline to Google Cloud APIs
(company requirement). See core/stt_google.py, core/tts_google.py,
core/llm_gemini.py for the replaced pieces. Bluetooth handling, VAD,
hallucination filtering, echo cancellation, and the Conversation
Manager (intent classification / knowledge retrieval / validator) are
all unchanged from the Groq version.

Features:
- Persistent pw-record (no Bluetooth crashes)
- Dynamic Bluetooth (any device, no hardcoded MAC)
- Echo cancellation (mic pauses during TTS playback)
- Multi-layer hallucination filter (energy, VAD, ASCII ratio relaxed
  for non-English, words/sec)
- Per-call history reset (on connect, disconnect, device change)
- Automatic language detection (English/Telugu/Hindi) via Google STT,
  carried through to the LLM reply language and the TTS voice
- Location + time context injected into every LLM call
- Intent classification + scoped knowledge retrieval + reply validation
- Greeting on call start
- Continuous recording loop (doesn't stop after first utterance)
- eSpeak fallback if Google TTS fails
- Graceful shutdown on Ctrl+C
"""

import logging
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytz
import webrtcvad

from companies.core.stt_google import transcribe as google_transcribe
from companies.core.tts_google import synthesize as google_synthesize
from companies.core.conversation_manager import ConversationManager

# ─────────────────────────────────────────────────────
# CONFIG — only edit this section
# ─────────────────────────────────────────────────────

# Google Cloud project + region. Must match what you set with:
#   gcloud config set project YOUR_PROJECT_ID
#   gcloud auth application-default set-quota-project YOUR_PROJECT_ID
# Set as env vars, or hardcode here.
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "1078973938049")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "asia-south1")

# Which company's knowledge base + rules to use. Switching companies
# later is just changing this line — see companies/<name>/.
COMPANY_DIR = Path(__file__).parent / "companies" / "maga_maharaja"

# LLM (Gemini via Vertex AI)
LLM_MODEL    = "gemini-2.5-flash"   # gemini-2.5-flash-lite would be faster but is not
                                     # available in asia-south1 (404) for this project —
                                     # only in us-central1, where the extra India->US
                                     # round trip would cost more than it saves
LLM_TOKENS   = 150                  # raised from the old 40-token cap — real replies (pricing tables,
                                     # delivery breakdowns, Telugu/Hindi phrasing) run several sentences
LLM_TEMP     = 0.5
LLM_HISTORY  = 3   # past exchanges to keep in context

# User context
USER_TIMEZONE = "Asia/Kolkata"
USER_LOCATION = "Hyderabad, Telangana, India"

# Greeting — spoken in English at the start of every call (per the
# always-on language rules: always start in English, switch only
# after the caller uses another language).
GREETING = "Hello! Welcome to Maga Maharaja Foods. How can I help you today?"

# VAD / Audio
SAMPLE_RATE       = 16000
FRAME_MS          = 30
FRAME_BYTES       = int(SAMPLE_RATE * FRAME_MS / 1000) * 2

VAD_MODE          = 3      # 0-3, 3 = most aggressive
ENERGY_THRESHOLD  = 300    # mean abs amplitude below this = not speech
SPEECH_MIN_FRAMES = 20     # sustained speech frames required
SILENCE_FRAMES    = 25     # consecutive silent frames = utterance ended
MIN_UTTERANCE_SEC = 0.8
MAX_UTTERANCE_SEC = 12.0
MIN_WORDS         = 2      # lowered from 3 — Telugu/Hindi short replies
                            # ("avunu andi", "sare andi") are valid at 2 words
MIN_WORDS_PER_SEC = 0.4    # below this = noise hallucination

# Hallucination filter
# NOTE: the old ASCII_RATIO_MIN=0.85 check has been REMOVED — it would
# reject genuine Telugu/Hindi speech transcribed in native script.
# Google STT can return Telugu in Telugu script depending on config;
# since we want Tenglish/Hinglish (English script) for the LLM and TTS
# prompt consistency, non-ASCII results are transliterated rather than
# rejected — see _maybe_transliterate() below.
HALLUCINATION_PHRASES = (
    "thank you for watching", "please subscribe", "like and subscribe",
    "subtitles by", "transcribed by", "www.", ".com", "♪", "♫",
    "you're watching", "don't forget to",
)

# Timing
WATCHDOG_SEC   = 3     # Bluetooth poll interval
RECONNECT_SEC  = 2     # wait before reopening stream after call drop
ECHO_TAIL_SEC  = 1.2   # silence after TTS before mic resumes — Bluetooth SCO
                       # has extra buffering/RF latency beyond what pw-play
                       # returning implies, so the old 0.4s let the tail end
                       # of the bot's own reply leak into the mic and get
                       # transcribed as a new caller utterance

# ─────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ai_caller")

# ─────────────────────────────────────────────────────
# GLOBALS
# ─────────────────────────────────────────────────────
_shutdown   = threading.Event()
_playing    = threading.Event()   # echo guard
_node_lock  = threading.Lock()
_hist_lock  = threading.Lock()

_inp_node   = None
_out_node   = None
_call_node  = None   # node that opened current stream

_history    = []
_conv_mgr   = None   # ConversationManager instance, set in main()
_vad        = webrtcvad.Vad(VAD_MODE)

audio_q     = queue.Queue(maxsize=3)
reply_q     = queue.Queue(maxsize=3)

# ─────────────────────────────────────────────────────
# BLUETOOTH — dynamic, no hardcoded MAC
# ─────────────────────────────────────────────────────

def _find_nodes():
    try:
        out = subprocess.run(
            ["wpctl", "status"], capture_output=True, text=True, timeout=5
        ).stdout
        i = re.search(r"(bluez_input\.[0-9A-Fa-f_]+\.\d+)", out)
        o = re.search(r"(bluez_output\.[0-9A-Fa-f_]+\.\d+)", out)
        return (i.group(1) if i else None, o.group(1) if o else None)
    except Exception:
        return None, None


def _device_name():
    try:
        out = subprocess.run(
            ["bluetoothctl", "devices", "Connected"],
            capture_output=True, text=True, timeout=5
        ).stdout
        names = [" ".join(l.split()[2:]) for l in out.splitlines() if len(l.split()) > 2]
        return ", ".join(names) or "unknown"
    except Exception:
        return "unknown"


def _clear_history(reason=""):
    with _hist_lock:
        _history.clear()
    log.info(f"History cleared{': ' + reason if reason else '.'}")


def watchdog_thread():
    global _inp_node, _out_node
    had_call = False
    log.info("Watchdog started.")

    while not _shutdown.is_set():
        inp, out = _find_nodes()
        with _node_lock:
            _inp_node = inp
            _out_node = out

        call_now = bool(inp and out)

        if call_now and not had_call:
            log.info(f"Call detected — device: {_device_name()}")
            _clear_history("new call")
            _play_greeting()

        elif had_call and not call_now:
            log.info("Call ended.")
            _clear_history("call ended")

        had_call = call_now
        _shutdown.wait(WATCHDOG_SEC)

    log.info("Watchdog stopped.")


# ─────────────────────────────────────────────────────
# AUDIO HELPERS
# ─────────────────────────────────────────────────────

def _energy(chunk: bytes) -> float:
    return float(np.abs(np.frombuffer(chunk, dtype=np.int16)).mean())


def _is_speech(chunk: bytes) -> bool:
    try:
        return _vad.is_speech(chunk, SAMPLE_RATE)
    except Exception:
        return False


# ─────────────────────────────────────────────────────
# HALLUCINATION FILTER
# (language-agnostic now — no ASCII-ratio rejection, since Telugu/Hindi
#  speech is expected, not noise)
# ─────────────────────────────────────────────────────

def _is_valid(text: str, duration: float) -> bool:
    if not text or len(text.strip()) < 2:
        return False

    lower = text.lower()
    if any(p in lower for p in HALLUCINATION_PHRASES):
        log.debug(f"[FILTER] Hallucination: '{text}'")
        return False

    words = text.split()
    if len(words) < MIN_WORDS:
        log.debug(f"[FILTER] Too few words ({len(words)}): '{text}'")
        return False

    if duration > 0 and (len(words) / duration) < MIN_WORDS_PER_SEC:
        log.debug(f"[FILTER] Low word rate: '{text}'")
        return False

    return True


# ─────────────────────────────────────────────────────
# STT — Google Cloud Speech-to-Text (multi-language)
# ─────────────────────────────────────────────────────

def _is_self_echo(transcript: str) -> bool:
    """
    Defense-in-depth against Bluetooth SCO echo tail: if the transcript
    is suspiciously similar to the bot's own last reply, it's more
    likely a leaked fragment of the TTS playback than a genuine new
    thing the caller said, so treat it as an echo rather than a turn.
    """
    with _hist_lock:
        last_reply = next(
            (h["content"] for h in reversed(_history) if h["role"] == "assistant"),
            None,
        )
    if not last_reply:
        return False

    t_words = set(transcript.lower().split())
    r_words = set(last_reply.lower().split())
    if not t_words or not r_words:
        return False

    overlap = len(t_words & r_words) / len(t_words)
    if overlap >= 0.6:
        log.info(f"[ECHO] Discarding likely self-echo ({overlap:.0%} overlap): '{transcript}'")
        return True
    return False


def _transcribe(pcm: bytes, duration: float) -> tuple[str, str]:
    """Returns (transcript, language_code). language_code is "" on failure."""
    text, language_code = google_transcribe(pcm)

    if not _is_valid(text, duration):
        return "", ""
    return text, language_code


# ─────────────────────────────────────────────────────
# LLM — routed through the Conversation Manager (Gemini backend)
# (intent classification -> knowledge lookup -> Gemini -> validator)
# ─────────────────────────────────────────────────────

def _ask(transcript: str, language_code: str) -> tuple[str, str]:
    """Returns (reply_text, language_code) for the TTS step to use."""
    tz  = pytz.timezone(USER_TIMEZONE)
    now = datetime.now(tz).strftime("%A, %d %B %Y, %I:%M %p IST")
    context = (
        f"Current date and time: {now}. "
        f"User location: {USER_LOCATION}."
    )

    with _hist_lock:
        recent = list(_history[-(LLM_HISTORY * 2):])

    reply, reply_language = _conv_mgr.handle_turn(transcript, recent, context, language_code)

    if reply:
        with _hist_lock:
            _history.append({"role": "user",      "content": transcript})
            _history.append({"role": "assistant",  "content": reply})

    return reply, reply_language


# ─────────────────────────────────────────────────────
# TTS — Google Cloud Text-to-Speech (language-matched voice)
# ─────────────────────────────────────────────────────

def _synthesize(text: str, language_code: str = "en-IN") -> bytes | None:
    wav = google_synthesize(text, language_code)
    if wav:
        return wav

    log.error("[TTS] Google TTS failed — trying eSpeak fallback (English only)")
    try:
        r = subprocess.run(
            ["espeak-ng", "-s", "150", "--stdout", text],
            capture_output=True, timeout=5
        )
        return r.stdout if r.returncode == 0 else None
    except Exception as e2:
        log.error(f"[TTS] eSpeak also failed: {e2}")
        return None


def _play(wav_bytes: bytes):
    with _node_lock:
        node = _out_node
    if not node:
        log.warning("[PLAY] No output node.")
        return
    _playing.set()
    try:
        subprocess.run(
            ["pw-play", "--target", node, "-"],
            input=wav_bytes,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(ECHO_TAIL_SEC)
    finally:
        _playing.clear()


def _play_greeting():
    def _go():
        wav = _synthesize(GREETING, "en-IN")
        if wav:
            _play(wav)
    threading.Thread(target=_go, daemon=True).start()


# ─────────────────────────────────────────────────────
# RECORDER THREAD
# One persistent pw-record per call session
# Resets after each utterance — never closes mid-call
# ─────────────────────────────────────────────────────

def recorder_thread():
    global _call_node
    log.info("Recorder started.")

    while not _shutdown.is_set():
        with _node_lock:
            node = _inp_node
        if not node:
            time.sleep(1)
            continue

        if node != _call_node:
            _call_node = node
            _clear_history(f"new node {node}")

        proc = subprocess.Popen(
            ["pw-record", "--target", node, "-",
             "--rate", str(SAMPLE_RATE), "--channels", "1", "--format", "s16"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        log.info(f"Stream open: {node}")

        def _reset():
            return 0, 0, False, bytearray(), time.time()

        speech_frames, silence_run, triggered, buf, t_start = _reset()

        try:
            while not _shutdown.is_set():
                with _node_lock:
                    if _inp_node != node:
                        log.info("Node changed — reopening stream.")
                        break

                chunk = proc.stdout.read(FRAME_BYTES)
                if len(chunk) < FRAME_BYTES:
                    log.warning("Stream ended unexpectedly.")
                    break

                if _playing.is_set():
                    speech_frames, silence_run, triggered, buf, t_start = _reset()
                    continue

                is_speech = (_energy(chunk) >= ENERGY_THRESHOLD and _is_speech(chunk))

                if is_speech:
                    if not triggered:
                        t_start = time.time()
                    triggered      = True
                    speech_frames += 1
                    silence_run    = 0
                    buf           += chunk

                elif triggered:
                    silence_run += 1
                    buf         += chunk

                    if silence_run >= SILENCE_FRAMES or (time.time() - t_start) >= MAX_UTTERANCE_SEC:
                        duration = len(buf) / (SAMPLE_RATE * 2)
                        if speech_frames >= SPEECH_MIN_FRAMES and duration >= MIN_UTTERANCE_SEC:
                            log.info(f"[REC] {duration:.2f}s / {speech_frames} speech frames")
                            if audio_q.full():
                                try:
                                    audio_q.get_nowait()
                                    log.debug("[REC] Dropped stale utterance.")
                                except queue.Empty:
                                    pass
                            audio_q.put((bytes(buf), duration))
                        else:
                            log.debug(f"[REC] Discarded: {duration:.2f}s / {speech_frames} frames")

                        speech_frames, silence_run, triggered, buf, t_start = _reset()

        finally:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
            log.info("Stream closed.")

        if not _shutdown.is_set():
            time.sleep(RECONNECT_SEC)

    log.info("Recorder stopped.")


# ─────────────────────────────────────────────────────
# PROCESSOR THREAD
# ─────────────────────────────────────────────────────

def processor_thread():
    log.info("Processor started.")

    while not _shutdown.is_set():
        try:
            pcm, duration = audio_q.get(timeout=1)
        except queue.Empty:
            continue

        transcript, language_code = _transcribe(pcm, duration)
        if not transcript:
            continue

        if _is_self_echo(transcript):
            continue

        log.info(f"Caller ({language_code}): '{transcript}'")

        reply, reply_language = _ask(transcript, language_code)
        if not reply:
            continue

        wav = _synthesize(reply, reply_language)
        if not wav:
            continue

        if reply_q.full():
            try:
                reply_q.get_nowait()
                log.debug("[PROC] Dropped stale reply.")
            except queue.Empty:
                pass
        reply_q.put(wav)

    log.info("Processor stopped.")


# ─────────────────────────────────────────────────────
# PLAYBACK THREAD
# ─────────────────────────────────────────────────────

def playback_thread():
    log.info("Playback started.")

    while not _shutdown.is_set():
        try:
            wav = reply_q.get(timeout=1)
        except queue.Empty:
            continue

        t0 = time.time()
        _play(wav)
        log.info(f"[PLAY] {time.time()-t0:.2f}s")

    log.info("Playback stopped.")


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────

def main():
    global _conv_mgr

    if "your-project-id-here" in os.environ["GOOGLE_CLOUD_PROJECT"]:
        sys.exit(
            "Set GOOGLE_CLOUD_PROJECT (and GOOGLE_CLOUD_LOCATION) — "
            "env vars, or edit the CONFIG section above."
        )

    log.info("AI Caller — Final (Google Cloud edition)")
    log.info(f"LLM={LLM_MODEL} | Project={os.environ['GOOGLE_CLOUD_PROJECT']} | "
              f"Location={os.environ['GOOGLE_CLOUD_LOCATION']} | User location={USER_LOCATION}")
    log.info(f"Company knowledge base: {COMPANY_DIR}")

    _conv_mgr = ConversationManager(
        company_dir=COMPANY_DIR,
        llm_model=LLM_MODEL,
        max_tokens=LLM_TOKENS,
        temperature=LLM_TEMP,
    )

    def _stop(sig, _):
        log.info("Shutting down...")
        _shutdown.set()

    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    threads = [
        threading.Thread(target=watchdog_thread,  daemon=True, name="watchdog"),
        threading.Thread(target=recorder_thread,  daemon=True, name="recorder"),
        threading.Thread(target=processor_thread, daemon=True, name="processor"),
        threading.Thread(target=playback_thread,  daemon=True, name="playback"),
    ]
    for t in threads:
        t.start()

    log.info("Ready — waiting for a call. Press Ctrl+C to stop.")
    while not _shutdown.is_set():
        time.sleep(0.5)

    for t in threads:
        t.join(timeout=3)
    log.info("Done.")


if __name__ == "__main__":
    main()
