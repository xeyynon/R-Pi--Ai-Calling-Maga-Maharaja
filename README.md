# AI Caller — Maga Maharaja Foods

An AI phone-call agent for Maga Maharaja Foods, a Telugu/Hindi/English-speaking
home-food delivery business. Answers customer calls, speaks in whichever of the
three languages the caller uses, and answers questions from the business's
knowledge base via a Gemini-backed conversation manager.

Runs on a Raspberry Pi, connected to a phone over Bluetooth HFP/SCO (the phone
receives the actual call; the Pi captures/plays audio over that Bluetooth link
via PipeWire).

## Architecture

- **Telephony**: Bluetooth HFP/SCO via PipeWire (`pw-record`/`pw-play`), one
  call at a time. `watchdog_thread` detects call start/end by polling for the
  Bluetooth audio nodes.
- **STT**: Sarvam AI (`saaras:v3`, primary — purpose-built for Indian-language
  and code-switched speech) with Google Cloud STT v2 (`latest_long`, `global`)
  as a fallback/comparison provider. Selected via `STT_PROVIDER` env var,
  dispatched through `companies/core/stt_provider.py`.
- **LLM**: Gemini (`gemini-2.5-flash`) via Vertex AI, streamed
  sentence-by-sentence so TTS can start speaking before the full reply is
  generated (`companies/core/llm_gemini.py`, `conversation_manager.py`).
- **TTS**: Google Cloud Text-to-Speech, streaming synthesis
  (`companies/core/tts_google.py`).
- **Main process**: `ai_caller_final3.py` — five threads (recorder, processor,
  playback, watchdog, loopback guard) coordinating capture, VAD/endpointing,
  the STT→LLM→TTS pipeline, barge-in/interrupt handling, and call-boundary
  isolation.

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Copy `local_config.example.py` to `local_config.py` and fill in your own
   `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, and `SARVAM_API_KEY`.
   `local_config.py` is gitignored — never commit real credentials.
3. Business-specific content (system prompt, fallback messages, knowledge
   base) lives in `companies/maga_maharaja/config.yaml`.

## Running

```bash
cd "/home/alfaleus/projects/For Customers"
setsid nohup /usr/bin/python3 ai_caller_final3.py > /tmp/live_test.log 2>&1 < /dev/null &
disown
```

`setsid` fully detaches the process from your terminal session — without it,
some shells deliver signals to backgrounded jobs in ways that can kill the
process unexpectedly. Watch the log with `tail -f /tmp/live_test.log`.

To stop it cleanly: `pkill -f ai_caller_final3.py` (shuts down in under a
second — verified via a hardened shutdown path that guarantees termination
regardless of thread state).

## Configuration

Most tuning knobs (VAD thresholds, timeouts, STT provider/model/location,
concurrency limits) are environment variables — see the `CONFIG` section near
the top of `ai_caller_final3.py` for the full list with their defaults and the
reasoning behind each value.

## Testing

- `tests/stt_regression_test.py` — replays real captured WAVs (from
  `AI_CALLER_DUMP_DIR`) through the actual STT provider code path, scored by
  word error rate against a small hand-confirmed ground truth.
- `tests/test_stop_e2e.py` — verifies the stop-word capture/recognition path
  offline, without a live call.
- `tests/test_streaming_e2e.py` — verifies the streaming
  Gemini→TTS→validation pipeline offline, without a live call.
- `run_benchmark.py` / `analyze_benchmark.py` — STT accuracy/latency
  benchmark harness and its report generator.
- `stt_diagnose.py` — offline root-cause tool for STT issues: measures the
  real frequency spectrum of captured audio and re-submits it to Google STT
  under several configs side by side.

## Data

- `logs/vad_metrics.csv` — one row per utterance, VAD/endpointing metrics.
  Rotates automatically at 5MB.
- `/home/alfaleus/call_data/` (configurable via `CALL_DATA_DIR`) — one folder
  per call, containing each turn's audio (WAV) and a `manifest.jsonl` pairing
  it with the transcript, detected language, and the bot's reply. Intended as
  training data for a future in-house model. Always on by default
  (`CALL_DATA_ENABLED=0` to disable). Guarded against filling the disk: stops
  recording (call handling continues normally) if free space drops below
  500MB.

