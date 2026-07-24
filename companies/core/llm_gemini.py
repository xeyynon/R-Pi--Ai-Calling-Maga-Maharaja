"""
llm_gemini.py

Thin wrapper around Gemini (via Vertex AI) that exposes a simple
chat-completion style function, so conversation_manager.py doesn't
need to know Gemini's message format differs from Groq/OpenAI-style
{"role", "content"} dicts.

Requires:
    pip install google-genai
    gcloud auth application-default login   (already done)
    gcloud config set project YOUR_PROJECT_ID
    gcloud auth application-default set-quota-project YOUR_PROJECT_ID

Set these two before first use (env vars or edit the constants below):
    GOOGLE_CLOUD_PROJECT   — your project ID (not the numeric project number)
    GOOGLE_CLOUD_LOCATION  — e.g. "asia-south1"
"""

import logging
import os

from google import genai
from google.genai import types

from .retry import with_retry

log = logging.getLogger("llm_gemini")

# Without an explicit deadline, a stalled network request can block
# this call forever without ever raising — with_retry() can't help
# against that since it only retries on an actual exception. Real
# calls finish in a few seconds; google-genai's HttpOptions.timeout is
# in MILLISECONDS (unlike the STT/TTS clients, which take seconds).
REQUEST_TIMEOUT_MS = 25_000

_client = None


def _get_client():
    global _client
    if _client is None:
        # Read lazily (not at import time) so it picks up whatever
        # ai_caller_final2.py has set via os.environ.setdefault(),
        # even though those calls run after this module is imported.
        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id-here")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "asia-south1")
        _client = genai.Client(vertexai=True, project=project, location=location)
    return _client


def _split_system_and_turns(messages: list[dict]) -> tuple[str, list[types.Content]]:
    """
    Converts Groq/OpenAI-style messages [{"role": "system"/"user"/"assistant",
    "content": "..."}] into Gemini's format: one combined system_instruction
    string, plus a list of Content turns (role "user" or "model").
    """
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


def ask(model: str, messages: list[dict], max_tokens: int, temperature: float) -> str:
    """
    Same shape of inputs as a Groq chat.completions.create() call would
    take, so this was a near drop-in replacement in conversation_manager.py.
    """
    client = _get_client()
    system_instruction, turns = _split_system_and_turns(messages)

    try:
        response = with_retry(lambda: client.models.generate_content(
            model=model,
            contents=turns,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                max_output_tokens=max_tokens,
                temperature=temperature,
                # Gemini 2.5 models "think" before answering by default,
                # spending output tokens on hidden reasoning. For short
                # knowledge-grounded phone replies that reasoning isn't
                # needed, and without disabling it, max_tokens can be
                # entirely consumed by thinking — leaving an empty reply
                # (finish_reason=MAX_TOKENS with zero visible text).
                # Disabling it also cuts latency since thinking happens
                # before the visible answer is generated.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
            ),
        ))
        return (response.text or "").strip()
    except Exception as e:
        log.error(f"[GEMINI] {e}")
        return ""
