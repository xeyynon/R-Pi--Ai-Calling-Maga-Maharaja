"""
validator.py

Last checkpoint before a reply reaches TTS. Local string check, no API
call: if the reply is empty or contains signals that it drifted into
general-knowledge territory ("capital of", "as an AI", etc.), swap in
the fallback message instead of trusting the LLM to police itself.
"""

import logging

log = logging.getLogger("validator")

# Phrases that suggest the model answered something general-knowledge
# rather than staying inside the business scope. Extend as needed.
OFF_TOPIC_SIGNALS = [
    "capital of",
    "as an ai",
    "i don't have access to",
    "i'm not able to browse",
]


def validate_reply(reply: str, fallback_message: str) -> str:
    if not reply.strip():
        log.info("[VALIDATOR] empty reply -> using fallback message")
        return fallback_message

    reply_lower = reply.lower()
    for signal in OFF_TOPIC_SIGNALS:
        if signal in reply_lower:
            log.info(f"[VALIDATOR] reply tripped signal '{signal}' -> using fallback message")
            return fallback_message

    return reply
