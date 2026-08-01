"""
validator.py

Last checkpoint before a reply reaches TTS. Local string check, no API
call: if the reply is empty or contains signals that it drifted into
general-knowledge territory ("capital of", "as an AI", etc.), swap in
the fallback message instead of trusting the LLM to police itself.
"""

import logging
import re

log = logging.getLogger("validator")

# The opening greeting the LLM is instructed not to repeat past the
# first turn (see always_on.md). Despite that instruction, it's been
# observed repeating a "Hello! Welcome to Maga Maharaja..." style
# opener on nearly every reply, not just the first — this is the
# deterministic backstop: strip it if the LLM says it anyway on a
# non-first turn, rather than relying solely on prompt compliance.
_GREETING_PATTERN = re.compile(
    r"^\s*(hello|hi|namaste)[!,.]?\s*welcome to maga maharaja\b[^.!?]*[.!?]\s*",
    re.IGNORECASE,
)

# Telugu ("andi") and Hindi ("ji") respectful particles have been
# observed leaking into English replies, despite always_on.md
# explicitly saying not to use them in English. Same deterministic
# backstop pattern as the greeting strip above — don't rely solely on
# the LLM following the prompt instruction.
_WRONG_LANGUAGE_PARTICLE_PATTERN = re.compile(r"\s*\b(?:andi|ji)\b\s*,?", re.IGNORECASE)

# 2026-08-01: conversation_manager.py prepends "(Reply in Hindi.)" /
# "(Reply in English.)" / "(Reply in Telugu.)" to the USER turn as a
# language reminder for Gemini, with an explicit system-prompt
# instruction to never echo it. Despite that instruction, a real call
# showed Gemini echoing it verbatim as the start of its OWN reply
# ("(Reply in Hindi.) Agar aap 10 kilo..."), which then got spoken
# aloud by TTS. Same deterministic backstop pattern as the two above —
# strip it if the LLM says it anyway, rather than relying solely on
# prompt compliance.
# 2026-08-01, second pass: the original pattern required a literal
# "." (or nothing) right before the closing paren — "(Reply in
# Hindi:)" with a colon, which is a plausible Gemini punctuation
# variant of the same echo, would not have matched. Widened to accept
# any single terminal punctuation mark there, not just a period.
_LANGUAGE_REMINDER_ECHO_PATTERN = re.compile(
    r"^\s*\(\s*reply\s+in\s+(?:english|hindi|telugu)\s*[.:]?\s*\)\s*",
    re.IGNORECASE,
)

# Phrases that suggest the model answered something general-knowledge
# rather than staying inside the business scope, or was steered off
# its persona by something in the caller's speech. Extend as needed.
OFF_TOPIC_SIGNALS = [
    "capital of",
    "as an ai",
    "i don't have access to",
    "i'm not able to browse",
    "i'm gemini",
    "i am gemini",
    "large language model",
    "i'm an ai language model",
    "i don't have personal",
]


def validate_reply(
    reply: str,
    fallback_message: str,
    is_first_turn: bool = False,
    language_code: str = "",
) -> str:
    if not reply.strip():
        log.info("[VALIDATOR] empty reply -> using fallback message")
        return fallback_message

    stripped = _LANGUAGE_REMINDER_ECHO_PATTERN.sub("", reply, count=1)
    if stripped != reply:
        log.info("[VALIDATOR] stripped echoed language-reminder prefix from reply")
        reply = stripped.strip() or fallback_message

    reply_lower = reply.lower()
    for signal in OFF_TOPIC_SIGNALS:
        if signal in reply_lower:
            log.info(f"[VALIDATOR] reply tripped signal '{signal}' -> using fallback message")
            return fallback_message

    if not is_first_turn:
        stripped = _GREETING_PATTERN.sub("", reply, count=1)
        if stripped != reply:
            log.info("[VALIDATOR] stripped repeated opening greeting from non-first-turn reply")
            reply = stripped.strip() or fallback_message

    if language_code.lower().startswith("en"):
        cleaned = _WRONG_LANGUAGE_PARTICLE_PATTERN.sub("", reply)
        cleaned = re.sub(r"\s+([.,!?])", r"\1", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        if cleaned != reply.strip():
            log.info("[VALIDATOR] stripped Telugu/Hindi particles from English reply")
            reply = cleaned or reply

    return reply
