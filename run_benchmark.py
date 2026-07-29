#!/usr/bin/env python3
"""
run_benchmark.py

TTS -> STT round-trip evaluation of the current production STT pipeline
(companies/core/stt_google.py, i.e. Google Cloud STT v2 as deployed).

Since there is no live caller available to record real audio for these
60 questions, each question's expected_text is synthesized to speech
with Google Cloud TTS (en-IN Chirp3-HD voice, chosen because all 60
transcripts are written in Roman script — there is no native Hindi/
Telugu script text to feed a hi-IN/te-IN voice) and the resulting audio
is fed back into the exact same transcribe() function the production
call pipeline uses. This measures the STT engine's raw recognition
ability under clean, noise-free, accent-neutral synthetic audio.

This is NOT equivalent to testing against real callers: it has no
telephone-line compression artifacts, no background noise, no genuine
Hindi/Telugu native-speaker phonetics or prosody, and no disfluencies.
Expect real accuracy on live calls to be measurably worse than these
numbers, especially on the code-switched Hindi/Telugu items.
"""
import io
import json
import os
import re
import statistics
import sys
import time
import wave
from difflib import SequenceMatcher

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "1078973938049")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "asia-south1")

sys.path.insert(0, "/home/alfaleus/projects/For Customers")

from companies.core import stt_google
from companies.core.tts_google import synthesize as google_synthesize
from google.cloud import speech_v2


def transcribe_with_confidence(pcm: bytes):
    """
    Calls Google STT v2 with the exact same client, recognizer resource,
    language_codes, and model as companies/core/stt_google.transcribe()
    (reusing its own helpers, not a re-specified config) — the only
    difference is this also returns the alternative's confidence score,
    which transcribe() computes internally but doesn't expose in its
    return signature. Production behavior is untouched; this is purely
    an additional read of the same response for benchmarking.
    """
    client = stt_google._get_client()
    recognizer = stt_google._get_recognizer()

    config = speech_v2.RecognitionConfig(
        explicit_decoding_config=speech_v2.ExplicitDecodingConfig(
            encoding=speech_v2.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=stt_google.SAMPLE_RATE,
            audio_channel_count=1,
        ),
        language_codes=stt_google.CANDIDATE_LANGUAGES,
        model=stt_google.STT_MODEL,
    )

    response = client.recognize(recognizer=recognizer, config=config, content=pcm)

    if not response.results:
        return "", "", None

    result = response.results[0]
    alt = result.alternatives[0]
    confidence = alt.confidence if alt.confidence else None
    return alt.transcript.strip(), result.language_code or "en-IN", confidence

DATASET_PATH = "/home/alfaleus/projects/For Customers/benchmark_dataset.json"
RESULTS_JSON_PATH = "/home/alfaleus/projects/For Customers/benchmark_results.json"
RESULTS_CSV_PATH = "/home/alfaleus/projects/For Customers/benchmark_results.csv"

TTS_VOICE_LANGUAGE = "en-IN"  # all 60 transcripts are Roman-script

# WER threshold below which a recognition counts as "Pass". 0.35 is a
# deliberately lenient bar for phone-quality code-switched speech —
# tune once real-call data exists.
PASS_WER_THRESHOLD = 0.35


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — so scoring
    isn't penalized for STT's auto-punctuation or casing choices."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Standard WER via word-level Levenshtein distance / len(reference words)."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m] / n


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Standard CER via character-level Levenshtein distance / len(reference chars)."""
    ref_chars = reference.replace(" ", "")
    hyp_chars = hypothesis.replace(" ", "")
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0

    n, m = len(ref_chars), len(hyp_chars)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m] / n


def similarity_score(reference: str, hypothesis: str) -> float:
    """difflib ratio, 0..1 — a coarser, non-edit-distance sanity metric."""
    return SequenceMatcher(None, reference, hypothesis).ratio()


def wav_bytes_to_pcm(wav_bytes: bytes) -> bytes:
    """Strips the WAV header to get raw PCM, matching what the real
    recorder thread hands to transcribe() in production."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.readframes(wf.getnframes())


def main():
    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    results = []
    print(f"Running benchmark: {len(dataset)} questions\n")

    for i, q in enumerate(dataset, 1):
        qid = q["id"]
        expected = q["expected_text"]
        lang = q["language"]
        category = q["category"]
        difficulty = q["difficulty"]

        print(f"[{i}/{len(dataset)}] {qid} ({lang}/{category}/{difficulty}): '{expected}'")

        # --- TTS synthesis ---
        t_tts0 = time.time()
        wav = google_synthesize(expected, TTS_VOICE_LANGUAGE)
        tts_elapsed = time.time() - t_tts0

        if not wav:
            print("  TTS FAILED — skipping")
            results.append({
                "id": qid, "language": lang, "category": category, "difficulty": difficulty,
                "expected_text": expected, "recognized_text": "", "detected_language": "",
                "confidence": None, "wer": 1.0, "cer": 1.0, "similarity": 0.0,
                "recognition_time_sec": None, "tts_time_sec": round(tts_elapsed, 3),
                "pass": False, "error": "TTS_FAILED",
            })
            continue

        with wave.open(io.BytesIO(wav), "rb") as wf:
            audio_duration = wf.getnframes() / wf.getframerate()

        pcm = wav_bytes_to_pcm(wav)

        # --- STT recognition (production's exact client/recognizer/config,
        # plus confidence which transcribe() doesn't expose) ---
        t_stt0 = time.time()
        try:
            recognized_text, detected_lang, confidence = transcribe_with_confidence(pcm)
            stt_elapsed = time.time() - t_stt0
            error = None
        except Exception as e:
            stt_elapsed = time.time() - t_stt0
            recognized_text, detected_lang, confidence = "", "ERROR", None
            error = str(e)

        norm_expected = normalize(expected)
        norm_recognized = normalize(recognized_text)

        wer = word_error_rate(norm_expected, norm_recognized)
        cer = character_error_rate(norm_expected, norm_recognized)
        sim = similarity_score(norm_expected, norm_recognized)
        passed = wer <= PASS_WER_THRESHOLD

        conf_str = f"{confidence:.3f}" if confidence is not None else "n/a"
        print(f"  Recognized: '{recognized_text}' (lang={detected_lang}, conf={conf_str})")
        print(f"  WER={wer:.3f} CER={cer:.3f} sim={sim:.3f} time={stt_elapsed:.2f}s "
              f"audio={audio_duration:.2f}s -> {'PASS' if passed else 'FAIL'}")

        results.append({
            "id": qid,
            "language": lang,
            "category": category,
            "difficulty": difficulty,
            "expected_text": expected,
            "recognized_text": recognized_text,
            "detected_language": detected_lang,
            "confidence": round(confidence, 4) if confidence is not None else None,
            "wer": round(wer, 4),
            "cer": round(cer, 4),
            "similarity": round(sim, 4),
            "audio_duration_sec": round(audio_duration, 3),
            "recognition_time_sec": round(stt_elapsed, 3),
            "tts_time_sec": round(tts_elapsed, 3),
            "pass": passed,
            "error": error,
        })
        print()

    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # --- CSV export ---
    import csv
    fieldnames = ["id", "language", "category", "difficulty", "expected_text",
                  "recognized_text", "detected_language", "confidence", "wer", "cer",
                  "similarity", "audio_duration_sec", "recognition_time_sec",
                  "tts_time_sec", "pass", "error"]
    with open(RESULTS_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to:\n  {RESULTS_JSON_PATH}\n  {RESULTS_CSV_PATH}")


if __name__ == "__main__":
    main()
