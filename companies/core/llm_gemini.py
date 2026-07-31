"""
llm_gemini.py

Thin wrapper around Gemini (via Vertex AI) that exposes both blocking
and streaming chat-completion style functions.

ask()    — blocking, returns full reply (used by repeat/translate paths)
stream() — yields text chunks as they arrive from Gemini (used by the
           main conversation path for streaming Gemini→TTS)
"""

import logging
import os
import threading

from google import genai
from google.genai import types

from .retry import with_retry

log = logging.getLogger("llm_gemini")

REQUEST_TIMEOUT_MS = 25_000

_client = None
# 2026-07-31: without this, warmup() racing an early real call from
# another thread could construct two genai.Client instances — the
# second silently discarding the first's already-warmed HTTP
# connection pool. Low likelihood (warmup normally finishes before
# calls start) but cheap to close outright rather than rely on timing.
_client_lock = threading.Lock()


def _get_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                project = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id-here")
                location = os.environ.get("GOOGLE_CLOUD_LOCATION", "asia-south1")
                _client = genai.Client(vertexai=True, project=project, location=location)
    return _client


def warmup(model: str):
    """
    Forces client construction (credential loading) AND a real, minimal
    generate_content call at process boot — unlike the STT/TTS clients,
    google-genai's HTTP connection pool is only actually opened on the
    first real request, so just constructing the client isn't enough to
    warm it. A 1-token throwaway call is a small, one-time cost per
    process start, in exchange for the first real caller not paying for
    connection setup on their own turn.
    """
    client = _get_client()
    try:
        client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=[types.Part(text="hi")])],
            config=types.GenerateContentConfig(
                max_output_tokens=1,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
            ),
        )
        log.info("[GEMINI] Client warmed.")
    except Exception as e:
        log.warning(f"[GEMINI] Warmup call failed (non-fatal, first real call will just be cold): {e}")


def _split_system_and_turns(messages: list[dict]) -> tuple[str, list[types.Content]]:
    system_parts = []
    turns = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            turns.append(types.Content(role="user", parts=[types.Part(text=content)]))
        elif role == "assistant":
            turns.append(types.Content(role="model", parts=[types.Part(text=content)]))

    return "\n\n".join(system_parts), turns


def _make_config(max_tokens: int, temperature: float) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=None,  # set per-call
        max_output_tokens=max_tokens,
        temperature=temperature,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
    )


def ask(model: str, messages: list[dict], max_tokens: int, temperature: float) -> str:
    """Blocking call — returns full reply text."""
    client = _get_client()
    system_instruction, turns = _split_system_and_turns(messages)

    config = _make_config(max_tokens, temperature)
    config.system_instruction = system_instruction

    try:
        response = with_retry(lambda: client.models.generate_content(
            model=model,
            contents=turns,
            config=config,
        ))
        return (response.text or "").strip()
    except Exception as e:
        log.error(f"[GEMINI] {e}")
        return ""


def stream(model: str, messages: list[dict], max_tokens: int, temperature: float):
    """
    Streaming call — yields text chunks as Gemini generates them.
    Each chunk is a non-empty string fragment. Caller assembles the
    full reply by concatenating all yielded chunks.

    Retries once, but only if the failure happens before any chunk has
    been yielded. Once even one chunk has gone out, the caller (the
    processor thread) may already be synthesizing/queuing it for
    playback — restarting the call from scratch at that point would
    re-speak content the caller already heard, so a failure past that
    point is logged and simply ends the stream instead of retrying.
    """
    client = _get_client()
    system_instruction, turns = _split_system_and_turns(messages)

    config = _make_config(max_tokens, temperature)
    config.system_instruction = system_instruction

    attempts = 2
    for attempt in range(1, attempts + 1):
        yielded_any = False
        try:
            response_stream = client.models.generate_content_stream(
                model=model,
                contents=turns,
                config=config,
            )
            for chunk in response_stream:
                text = chunk.text
                if text:
                    yielded_any = True
                    yield text
            return
        except Exception as e:
            if yielded_any:
                log.error(f"[GEMINI-STREAM] failed after partial output: {e}")
                return
            log.warning(f"[GEMINI-STREAM] attempt {attempt}/{attempts} failed before any output: {e}")

    log.error("[GEMINI-STREAM] all attempts failed with no output")
