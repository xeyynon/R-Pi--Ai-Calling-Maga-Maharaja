"""
conversation_manager.py

Orchestrates one reply turn:
  1. Load the always-on core knowledge (identity, language rules, bot
     rules, escalation triggers) — every turn, unconditionally.
  2. Retrieve the top-matching knowledge sections for the transcript
     via knowledge_retriever.get_relevant_context() (pure local
     keyword scoring, no API call).
  3. Assemble ONE system prompt (base prompt + language instruction +
     location/time context + core rules + retrieved knowledge) and
     make ONE call to llm_gemini.ask().
  4. Run the reply through validator.validate_reply() as a local
     safety-net check before it's spoken.

There used to be a separate LLM call here just to classify the
transcript into an intent category before retrieval, so the right
single knowledge file could be opened. That call was removed —
get_relevant_context() searches every topic file directly, so the
classification step was pure overhead. This halves the LLM calls per
turn with no loss in answer quality.
"""

import logging

import yaml

from .knowledge_retriever import get_core_context, get_relevant_context
from .llm_gemini import ask as gemini_ask
from .validator import validate_reply

log = logging.getLogger("conversation_manager")


class ConversationManager:
    def __init__(self, company_dir, llm_model: str, max_tokens: int, temperature: float):
        self.company_dir = company_dir
        self.llm_model = llm_model
        self.max_tokens = max_tokens
        self.temperature = temperature

        config_path = company_dir / "config.yaml"
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.base_prompt = self.config["system_prompt"]
        self.fallback_messages = {
            lang.lower(): msg for lang, msg in self.config["fallback_messages"].items()
        }

        self.core_context = get_core_context(company_dir)

    def _fallback_for(self, detected_language: str) -> str:
        key = (detected_language or "en-in").lower()
        return self.fallback_messages.get(key, self.fallback_messages["en-in"])

    def _build_system_prompt(self, context: str, knowledge: str, detected_language: str, is_first_turn: bool) -> str:
        parts = [self.base_prompt]

        if is_first_turn:
            parts.append("This is the first message of the call — the automatic greeting has already been spoken.")
        else:
            parts.append("This is a continuing conversation, not the start of the call — do not greet again.")

        if detected_language:
            parts.append(f"The caller's detected language is: {detected_language}.")

        if context:
            parts.append(context)

        if self.core_context:
            parts.append(self.core_context)

        if knowledge:
            parts.append("Relevant knowledge for this query:\n" + knowledge)
        else:
            log.debug("[KNOWLEDGE] no relevant sections retrieved for this turn")

        return "\n\n".join(parts)

    def handle_turn(self, transcript: str, recent_history: list[dict], context: str, detected_language: str):
        """
        Returns (reply_text, reply_language). reply_language mirrors
        detected_language so the caller can be answered in the
        language they used.
        """
        is_first_turn = not recent_history

        knowledge = get_relevant_context(self.company_dir, transcript)
        system_prompt = self._build_system_prompt(context, knowledge, detected_language, is_first_turn)

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(recent_history)
        messages.append({"role": "user", "content": transcript})

        reply = gemini_ask(
            model=self.llm_model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        reply = validate_reply(
            reply,
            self._fallback_for(detected_language),
            is_first_turn,
            language_code=detected_language or "",
        )

        return reply, detected_language or "en-IN"
