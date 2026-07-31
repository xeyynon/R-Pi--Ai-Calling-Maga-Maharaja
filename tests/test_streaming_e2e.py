#!/usr/bin/env python3
"""
End-to-end test of the streaming Gemini -> TTS -> validation pipeline,
without needing a live phone call. Exercises:
  1. conversation_manager.handle_turn_streaming() against the real
     knowledge base (sentence splitting on real Gemini output)
  2. Per-sentence validate_reply() gating (mirrors the fixed
     processor_thread logic in ai_caller_final3.py)
  3. _synthesize() per sentence + _concat_wav_chunks() producing a
     single valid, playable WAV
"""
import io
import os
import sys
import wave

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "1078973938049")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "asia-south1")

sys.path.insert(0, "/home/alfaleus/projects/For Customers")

from pathlib import Path
from companies.core.conversation_manager import ConversationManager
from companies.core.validator import validate_reply
from companies.core.tts_google import synthesize as google_synthesize

COMPANY_DIR = Path("/home/alfaleus/projects/For Customers/companies/maga_maharaja")

cm = ConversationManager(
    company_dir=COMPANY_DIR,
    llm_model="gemini-2.5-flash",
    max_tokens=150,
    temperature=0.5,
)

TEST_QUERIES = [
    ("What are your delivery charges?", "en-IN", True),
    ("Do you deliver Telugu snacks to Hyderabad?", "en-IN", False),
    ("Mujhe pickle chahiye, price kya hai?", "hi-IN", False),
]

context = "Current date and time: Tuesday, 28 July 2026, 03:30 PM IST. User location: India."


def concat_wav_chunks(chunks):
    if not chunks:
        return None
    if len(chunks) == 1:
        return chunks[0]
    frames = []
    params = None
    for chunk in chunks:
        with wave.open(io.BytesIO(chunk), "rb") as wf:
            if params is None:
                params = wf.getparams()
            frames.append(wf.readframes(wf.getnframes()))
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setparams(params)
        for f in frames:
            wf.writeframes(f)
    return out.getvalue()


for transcript, lang, is_first in TEST_QUERIES:
    print(f"\n{'='*70}")
    print(f"QUERY: '{transcript}' (lang={lang}, is_first_turn={is_first})")
    print('='*70)

    fallback_message = cm._fallback_for(lang)
    full_reply_parts = []
    wav_chunks = []
    n_sentences = 0

    for sentence, sent_lang in cm.handle_turn_streaming(transcript, [], context, lang):
        n_sentences += 1
        validated = validate_reply(sentence, fallback_message, is_first, language_code=lang)

        flag = ""
        if validated != sentence:
            flag = "  [VALIDATOR MODIFIED THIS SENTENCE]"

        print(f"  sentence {n_sentences}: '{sentence}'{flag}")
        if flag:
            print(f"    -> validated: '{validated}'")

        full_reply_parts.append(validated)

        wav = google_synthesize(validated, sent_lang)
        if wav:
            wav_chunks.append(wav)
            with wave.open(io.BytesIO(wav), "rb") as wf:
                dur = wf.getnframes() / wf.getframerate()
                print(f"    TTS ok: {len(wav)} bytes, {dur:.2f}s audio")
        else:
            print(f"    TTS FAILED for this sentence")

    full_reply = " ".join(full_reply_parts).strip()
    print(f"\n  FULL REPLY: '{full_reply}'")
    print(f"  Sentence count: {n_sentences}, WAV chunks: {len(wav_chunks)}")

    merged = concat_wav_chunks(wav_chunks)
    if merged:
        with wave.open(io.BytesIO(merged), "rb") as wf:
            total_dur = wf.getnframes() / wf.getframerate()
            print(f"  MERGED WAV: {len(merged)} bytes, {total_dur:.2f}s audio, "
                  f"channels={wf.getnchannels()}, rate={wf.getframerate()}, "
                  f"sampwidth={wf.getsampwidth()}")
            # Sanity: merged duration should roughly equal sum of chunk durations
            chunk_durs = []
            for c in wav_chunks:
                with wave.open(io.BytesIO(c), "rb") as wf2:
                    chunk_durs.append(wf2.getnframes() / wf2.getframerate())
            expected = sum(chunk_durs)
            diff = abs(total_dur - expected)
            status = "OK" if diff < 0.05 else "MISMATCH"
            print(f"  Sum of individual chunk durations: {expected:.2f}s -> {status} (diff={diff:.3f}s)")
    else:
        print("  MERGED WAV: None (no chunks)")

print("\n\nAll test queries completed.")
