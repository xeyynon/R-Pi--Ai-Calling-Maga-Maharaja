#!/usr/bin/env python3
"""
End-to-end verification of the stop-word fix, without a live phone call.

Synthesizes short stop-phrases via Google TTS, then replays them
through the EXACT same frame-by-frame onset/hangover logic the
recorder uses (same functions, same thresholds as currently deployed
in ai_caller_final3.py), to see whether they now get captured as real
utterances instead of being discarded — then runs the captured audio
through the real STT and the real _is_stop_request() logic.

Caveat: TTS-synthesized audio isn't identical to real human speech (no
real disfluencies/accent/background noise), so this doesn't replace a
live call test. But it DOES directly validate the specific mechanism
that was diagnosed: whether frame-count/duration thresholds now let a
short spoken word through.
"""
import array
import io
import os
import re
import sys
import wave

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "1078973938049")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "asia-south1")

sys.path.insert(0, "/home/alfaleus/projects/For Customers")

import webrtcvad
from companies.core.stt_google import transcribe
from companies.core.tts_google import synthesize as google_synthesize

# Exact current production values (ai_caller_final3.py, as deployed)
SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000) * 2
VAD_MODE = 3
ENERGY_THRESHOLD = 300
SPEECH_MIN_FRAMES = 10   # new value (was 20)
SILENCE_FRAMES = 16      # 500ms hangover / 30ms frames
MIN_UTTERANCE_SEC = 0.4  # new value (was 0.8)
STOP_CHECK_MIN_FRAMES = 4  # tentative-capture bar (new)

_vad = webrtcvad.Vad(VAD_MODE)


def _energy(chunk: bytes) -> float:
    samples = array.array("h")
    samples.frombytes(chunk)
    return sum(abs(s) for s in samples) / len(samples)


def _is_speech(chunk: bytes) -> bool:
    try:
        return _vad.is_speech(chunk, SAMPLE_RATE)
    except Exception:
        return False


# Reproduced from ai_caller_final3.py — kept in sync manually for this test
STOP_WORDS = {
    "stop", "hold", "pause", "chup",
    "ruk", "ruko", "rukja",
    "aagu", "aapu", "niluvu", "aagandi", "apu",
}
AMBIGUOUS_STOP_WORDS = {"bas", "wait"}
STOP_KEYWORDS = (
    "stop talking", "wait a", "hold on", "one second", "one minute",
    "give me a second", "give me a minute",
    "ruk ja", "thoda ruko", "ek minute", "ek second", "chup raho",
    "konchem aagu", "oka nimisham", "aagandi konchem",
)


def _is_stop_request(transcript: str) -> bool:
    words = re.findall(r"\w+", transcript.lower())
    if not words:
        return False
    if any(w in STOP_WORDS for w in words):
        return True
    if all(w in (STOP_WORDS | AMBIGUOUS_STOP_WORDS) for w in words):
        return True
    lower = transcript.lower()
    return any(k in lower for k in STOP_KEYWORDS)


def simulate_recorder(pcm: bytes):
    """
    Replays raw PCM through frame-by-frame onset/hangover logic
    identical to recorder_thread's inner loop, WITHOUT the _playing/
    _greeting_guard checks (not relevant here — this only tests
    whether a clip of this length/energy would be captured at all).

    Returns (captured: bool, speech_frames: int, duration_when_finalized: float)
    or (False, max_speech_frames_seen, None) if never finalized.
    """
    speech_frames = 0
    silence_run = 0
    triggered = False
    buf = bytearray()

    for i in range(0, len(pcm) - FRAME_BYTES + 1, FRAME_BYTES):
        chunk = pcm[i:i + FRAME_BYTES]
        is_sp = (_energy(chunk) >= ENERGY_THRESHOLD and _is_speech(chunk))

        if is_sp:
            triggered = True
            speech_frames += 1
            silence_run = 0
            buf += chunk
        elif triggered:
            silence_run += 1
            buf += chunk
            if silence_run >= SILENCE_FRAMES:
                duration = len(buf) / (SAMPLE_RATE * 2)
                return _classify(speech_frames, duration, buf)

    # Ran out of audio before hangover triggered — simulate end-of-clip
    # as if silence followed (same as what happens right after a real
    # utterance ends and the mic keeps recording quiet Bluetooth noise).
    if triggered:
        duration = len(buf) / (SAMPLE_RATE * 2)
        return _classify(speech_frames, duration, buf)
    return False, 0, 0.0, None, False


def _classify(speech_frames, duration, buf):
    if speech_frames >= SPEECH_MIN_FRAMES and duration >= MIN_UTTERANCE_SEC:
        return True, speech_frames, duration, bytes(buf), False
    elif speech_frames >= STOP_CHECK_MIN_FRAMES:
        return True, speech_frames, duration, bytes(buf), True  # tentative
    return False, speech_frames, duration, None, False


TEST_PHRASES = [
    "Stop",
    "Chup",
    "Can you stop",
    "Please stop",
    "Wait",
    "Bas",
    "Yes",  # real acknowledgment, NOT a stop-word - checking the tentative path's boundary
    "Ok",
]

print(f"Onset guard: SPEECH_MIN_FRAMES={SPEECH_MIN_FRAMES} ({SPEECH_MIN_FRAMES*FRAME_MS}ms), "
      f"MIN_UTTERANCE_SEC={MIN_UTTERANCE_SEC}s\n")

for phrase in TEST_PHRASES:
    print(f"{'='*60}\nPhrase: '{phrase}'")
    wav = google_synthesize(phrase, "en-IN")
    if not wav:
        print("  TTS FAILED")
        continue

    with wave.open(io.BytesIO(wav), "rb") as wf:
        pcm = wf.readframes(wf.getnframes())
        tts_duration = wf.getnframes() / wf.getframerate()
    print(f"  TTS audio duration: {tts_duration:.2f}s")

    captured, speech_frames, duration, captured_pcm, tentative = simulate_recorder(pcm)
    tag = " [TENTATIVE]" if tentative else ""
    print(f"  Onset guard result: captured={captured}{tag}, speech_frames={speech_frames} "
          f"({speech_frames*FRAME_MS}ms), utterance_len={duration:.2f}s")

    if not captured:
        print("  -> DISCARDED before reaching STT (same failure mode as before)")
        continue

    transcript, lang = transcribe(captured_pcm)
    print(f"  STT transcript: '{transcript}' (lang={lang})")

    if not transcript:
        print("  -> STT found no speech in the captured clip")
        if tentative:
            print("  -> tentative + empty transcript = silently discarded (correct)")
        continue

    is_stop = _is_stop_request(transcript)
    print(f"  _is_stop_request('{transcript}') = {is_stop}")

    if tentative and not is_stop:
        print("  -> tentative + not a stop-word = silently discarded (correct, no false turn)")
    elif is_stop:
        print("  -> CORRECTLY TRIGGERS STOP RESPONSE")
    else:
        print("  -> DOES NOT TRIGGER (check if expected)")

print(f"\n{'='*60}\nDone.")
