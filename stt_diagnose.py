"""
stt_diagnose.py — offline STT root-cause harness.

Run on the Pi AFTER an instrumented call has dumped WAVs to a directory
(AI_CALLER_DUMP_DIR). For every dumped utterance it:

  1. Measures the real spectrum — the fraction of total energy that
     sits ABOVE 4 kHz. A CVSD (8 kHz narrowband) SCO link upsampled to
     16 kHz has ~0% energy there (a hard cliff); a true mSBC 16 kHz
     wideband link has meaningful energy above 4 kHz. This is the
     single measurement that decides "is the audio itself the problem."

  2. Re-submits the SAME bytes to Google STT v2 under several configs
     (model x language-set x declared-rate) and prints every transcript
     side by side, so the winning config is chosen from evidence, not
     guessed. Reusing recorded audio means this costs nothing extra to
     iterate and is perfectly reproducible.

Usage:
    GOOGLE_CLOUD_PROJECT=1078973938049 python3 stt_diagnose.py /home/alfaleus/stt_dumps
"""

import os
import sys
import wave
from pathlib import Path

import numpy as np
from google.cloud import speech_v2

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "1078973938049")
LOCATION = "global"
RECOGNIZER = f"projects/{PROJECT}/locations/{LOCATION}/recognizers/_"

# (label, model, language_codes, declared_rate_hz, downsample_to_8k)
# declared_rate_hz + downsample let us test the telephony model on
# genuinely-8kHz audio, the way it's meant to be used.
CONFIGS = [
    ("current (latest_long, 3-lang, 16k)", "latest_long", ["en-IN", "hi-IN", "te-IN"], 16000, False),
    ("latest_long EN-only 16k",            "latest_long", ["en-IN"],                    16000, False),
    ("telephony 3-lang 8k",                "telephony",   ["en-IN", "hi-IN", "te-IN"], 8000,  True),
    ("telephony EN-only 8k",               "telephony",   ["en-IN"],                    8000,  True),
    ("chirp_2 3-lang 16k",                 "chirp_2",     ["en-IN", "hi-IN", "te-IN"], 16000, False),
    ("chirp_2 EN-only 16k",                "chirp_2",     ["en-IN"],                    16000, False),
]

_client = speech_v2.SpeechClient()


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        pcm = w.readframes(w.getnframes())
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    return samples, rate


def hf_energy_fraction(samples, rate, cutoff=4000):
    """Fraction of spectral energy above `cutoff` Hz. Near 0 => narrowband."""
    if len(samples) < 256:
        return float("nan")
    windowed = samples * np.hanning(len(samples))
    spectrum = np.abs(np.fft.rfft(windowed)) ** 2
    freqs = np.fft.rfftfreq(len(samples), 1.0 / rate)
    total = spectrum.sum()
    if total <= 0:
        return float("nan")
    return spectrum[freqs >= cutoff].sum() / total


def downsample_16k_to_8k(samples):
    # crude decimation-by-2 after a light average — fine for STT input,
    # this is only to hand the telephony model an 8kHz stream.
    smoothed = (samples[:-1] + samples[1:]) / 2
    return smoothed[::2]


def to_pcm16(samples):
    return np.clip(samples, -32768, 32767).astype(np.int16).tobytes()


def transcribe(pcm, model, langs, rate):
    config = speech_v2.RecognitionConfig(
        explicit_decoding_config=speech_v2.ExplicitDecodingConfig(
            encoding=speech_v2.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=rate,
            audio_channel_count=1,
        ),
        language_codes=langs,
        model=model,
    )
    try:
        resp = _client.recognize(recognizer=RECOGNIZER, config=config, content=pcm)
    except Exception as e:
        return f"<ERROR: {str(e)[:80]}>"
    if not resp.results or not resp.results[0].alternatives:
        return "<no speech>"
    r = resp.results[0]
    return f"{r.alternatives[0].transcript.strip()!r} (lang={r.language_code})"


def main():
    dump_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/alfaleus/stt_dumps")
    wavs = sorted(dump_dir.glob("*.wav"))
    if not wavs:
        print(f"No WAVs in {dump_dir}")
        return

    print(f"Analyzing {len(wavs)} utterance(s) from {dump_dir}\n")
    for path in wavs:
        samples, rate = read_wav(path)
        dur = len(samples) / rate
        hf = hf_energy_fraction(samples, rate)
        print("=" * 78)
        print(f"{path.name}  ({dur:.2f}s @ {rate}Hz)  energy>4kHz = {hf*100:.1f}%"
              f"   {'<-- NARROWBAND (cliff at 4kHz)' if hf < 0.02 else '<-- has wideband content'}")
        print("-" * 78)
        pcm16 = to_pcm16(samples)
        pcm8 = to_pcm16(downsample_16k_to_8k(samples))
        for label, model, langs, decl_rate, down in CONFIGS:
            pcm = pcm8 if down else pcm16
            result = transcribe(pcm, model, langs, decl_rate)
            print(f"  {label:34s} {result}")
        print()


if __name__ == "__main__":
    main()
