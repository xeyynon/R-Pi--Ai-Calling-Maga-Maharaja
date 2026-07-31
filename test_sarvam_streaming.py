"""
Standalone experiment: does Sarvam's real streaming API actually let us
overlap recognition with capture, and how must audio chunks be framed?
Tests TWO framing strategies against the same real captured utterance
so we get a direct, honest answer instead of guessing from docs.
"""
import asyncio
import base64
import io
import os
import sys
import time
import wave

sys.path.insert(0, ".")
from local_config import SARVAM_API_KEY
from sarvamai import AsyncSarvamAI

WAV_PATH = sys.argv[1] if len(sys.argv) > 1 else "/home/alfaleus/stt_dumps/utt_120316_001.wav"
FRAME_MS = 30
SAMPLE_RATE = 16000
FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000) * 2


def load_pcm():
    with wave.open(WAV_PATH, "rb") as w:
        return w.readframes(w.getnframes())


def pcm_chunk_to_wav_bytes(pcm_chunk: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm_chunk)
    return buf.getvalue()


async def run_strategy(name, wrap_each_chunk_as_wav):
    print(f"\n=== Strategy: {name} ===")
    pcm = load_pcm()
    client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)
    t0 = time.time()
    final_transcript = None
    first_partial_time = None
    events = []

    try:
        async with client.speech_to_text_streaming.connect(
            model="saaras:v3",
            mode="transcribe",
            language_code="en-IN",
            high_vad_sensitivity=True,
            vad_signals=True,
        ) as ws:

            async def sender():
                for i in range(0, len(pcm), FRAME_BYTES):
                    chunk = pcm[i:i + FRAME_BYTES]
                    if len(chunk) < FRAME_BYTES:
                        break
                    audio_bytes = pcm_chunk_to_wav_bytes(chunk) if wrap_each_chunk_as_wav else chunk
                    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
                    await ws.transcribe(audio=audio_b64, encoding="audio/wav", sample_rate=SAMPLE_RATE)
                    await asyncio.sleep(FRAME_MS / 1000)  # simulate real-time arrival
                print(f"  [sender] finished feeding {len(pcm)} bytes at t={time.time()-t0:.2f}s")

            send_task = asyncio.create_task(sender())

            async def receiver():
                nonlocal final_transcript, first_partial_time
                async for message in ws:
                    elapsed = time.time() - t0
                    events.append((elapsed, message.type, str(message.data)[:120]))
                    if message.type == "data":
                        transcript = getattr(message.data, "transcript", None)
                        if transcript and first_partial_time is None:
                            first_partial_time = elapsed
                        if transcript:
                            final_transcript = transcript
                    if send_task.done():
                        break

            try:
                await asyncio.wait_for(receiver(), timeout=15)
            except asyncio.TimeoutError:
                print("  [receiver] timed out waiting for messages")

    except Exception as e:
        print(f"  ERROR: {e}")
        return

    print(f"  events received: {len(events)}")
    for elapsed, mtype, data in events[:10]:
        print(f"    t={elapsed:.2f}s type={mtype} data={data}")
    print(f"  first_partial_time={first_partial_time}")
    print(f"  final_transcript={final_transcript!r}")


async def main():
    await run_strategy("raw PCM chunks, no per-chunk WAV wrap", wrap_each_chunk_as_wav=False)
    await run_strategy("each chunk individually WAV-wrapped", wrap_each_chunk_as_wav=True)


asyncio.run(main())
