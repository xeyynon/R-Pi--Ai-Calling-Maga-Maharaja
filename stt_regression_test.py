"""
stt_regression_test.py — STT regression harness against real captured audio.

Built 2026-07-30 as part of a stabilization pass, directly targeting
the root cause behind every bug in that day's STT refactor: a new
capability (phrase hints, then real Sarvam streaming) had no automated
check and only got caught by a live phone call, sometimes after it had
already shipped. This harness costs nothing new to run — it replays
audio already captured in `/home/alfaleus/stt_dumps/` (via
AI_CALLER_DUMP_DIR) through the SAME code path a real call uses
(`companies.core.stt_provider.start_streaming_session`, frame-by-frame,
not a hand-rolled batch call), so it exercises whichever STT_PROVIDER
is currently active — the exact thing that broke before.

Reuses word_error_rate() from run_benchmark.py rather than
reimplementing it — same reasoning as this session's earlier WER/CER
dedup: one edit-distance implementation, not two.

GROUND_TRUTH below is deliberately small and conservative: only
utterances whose actual spoken content was cross-confirmed across
multiple real-call tests this session are included. Fabricating
ground truth for an uncertain utterance would make this tool actively
misleading — better a short, trustworthy list that grows over time
than a long, guessed one. NO_SPEECH_EXPECTED covers utterances known
to be silence/background noise, where the only correct answer is an
empty transcript — a non-empty result there is exactly the
hallucination failure mode found in production.

Usage:
    cd "/home/alfaleus/projects/For Customers"
    python3 stt_regression_test.py                       # uses current STT_PROVIDER
    STT_PROVIDER=google python3 stt_regression_test.py
    STT_PROVIDER=sarvam python3 stt_regression_test.py
    STT_PROVIDER=google STT_MODEL=chirp_2 STT_LOCATION=asia-southeast1 python3 stt_regression_test.py
    STT_PROVIDER=google STT_MODEL=chirp STT_LOCATION=asia-southeast1 python3 stt_regression_test.py
"""

import os
import sys
import wave
from pathlib import Path

sys.path.insert(0, ".")

# Same local_config.py fallback ai_caller_final3.py uses for
# SARVAM_API_KEY — without this, running this script standalone (not
# through the main app) leaves the key unset and every Sarvam call
# fails with "SARVAM_API_KEY is not set", which looks like a broken
# test run rather than a missing setup step.
try:
    from local_config import SARVAM_API_KEY as _LOCAL_SARVAM_KEY
except ImportError:
    _LOCAL_SARVAM_KEY = ""
if _LOCAL_SARVAM_KEY:
    os.environ.setdefault("SARVAM_API_KEY", _LOCAL_SARVAM_KEY)

from run_benchmark import word_error_rate, PASS_WER_THRESHOLD
from companies.core.stt_provider import start_streaming_session, STT_PROVIDER

DUMP_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/alfaleus/stt_dumps")
FRAME_BYTES = 960  # 30ms @ 16kHz s16 mono — matches recorder_thread's real frame size

# Cross-confirmed across multiple real-call tests this session — see
# module docstring for why this list is short on purpose.
GROUND_TRUTH = {
    "utt_120316_001.wav": "what is Maga Maharaja",
    "utt_121114_018.wav": "how much is the ghee",
    "utt_121138_019.wav": "how much is the ghee",
    "utt_121157_021.wav": "how much is the ghee",
    "utt_121208_022.wav": "delivery time",
}

NO_SPEECH_EXPECTED = {
    "utt_120342_004.wav",
    "utt_120955_008.wav",
    "utt_120957_009.wav",
    "utt_121004_011.wav",
    "utt_121015_013.wav",
    "utt_121148_020.wav",
    "utt_121227_023.wav",
}


def transcribe_via_pipeline(wav_path: Path) -> str:
    """Feeds a WAV through the real StreamingSession interface, frame
    by frame, exactly as recorder_thread does — not a hand-rolled
    batch call. This is what makes the test meaningful: it exercises
    the actual code path a live call uses, including whichever
    provider/config is currently active."""
    with wave.open(str(wav_path), "rb") as w:
        pcm = w.readframes(w.getnframes())
    session = start_streaming_session()
    for i in range(0, len(pcm), FRAME_BYTES):
        session.feed(pcm[i:i + FRAME_BYTES])
    transcript, _language_code = session.finish()
    return transcript


def main():
    wavs = sorted(DUMP_DIR.glob("*.wav"))
    if not wavs:
        print(f"No WAVs in {DUMP_DIR}")
        return 1

    print(f"STT regression test — provider={STT_PROVIDER}, {len(wavs)} utterance(s)\n")

    results = []
    for path in wavs:
        transcript = transcribe_via_pipeline(path)

        if path.name in NO_SPEECH_EXPECTED:
            passed = (transcript == "")
            detail = f"expected silence, got {transcript!r}" if not passed else "correctly silent"
            results.append((path.name, passed, detail))
        elif path.name in GROUND_TRUTH:
            expected = GROUND_TRUTH[path.name]
            wer = word_error_rate(expected.lower(), transcript.lower())
            passed = wer <= PASS_WER_THRESHOLD
            detail = f"WER={wer:.2f} expected={expected!r} got={transcript!r}"
            results.append((path.name, passed, detail))
        else:
            # No ground truth for this one — informational only, not
            # a pass/fail. Printed so a human can eyeball it and
            # potentially promote it to GROUND_TRUTH later.
            print(f"  ?    {path.name:28s} (no ground truth) got={transcript!r}")
            continue

        status = "PASS" if passed else "FAIL"
        print(f"  {status} {path.name:28s} {detail}")

    checked = [r for r in results]
    failed = [r for r in checked if not r[1]]
    print(f"\n{len(checked) - len(failed)}/{len(checked)} checked utterances passed"
          f" ({len(wavs) - len(checked)} had no ground truth, shown above but not scored)")
    if failed:
        print("FAILURES:")
        for name, _, detail in failed:
            print(f"  {name}: {detail}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
