
"""
knowledge_retriever.py

Two layers of knowledge:

1. ALWAYS-ON CORE (knowledge/core/always_on.md) — identity, language
   switching rules, general bot behavior rules, escalation triggers.
   This is small enough and universally relevant enough that it gets
   loaded into EVERY reply, regardless of topic.

2. CROSS-FILE RETRIEVAL — every ## section across every topic file in
   knowledge/ is scored against the transcript by keyword overlap; the
   top-scoring sections (up to MAX_CONTEXT_CHARS) are returned. This
   replaces the old per-intent single-file lookup: instead of trusting
   an upstream classifier to pick the one right file, every topic is
   searched directly, so a query that touches two topics (e.g.
   "avakaya price and delivery time") can still pull sections from
   both products_pricing.md and delivery_shipping.md.

No vector DB here on purpose: on a Pi 4B, spinning up a
sentence-transformer model just to search a handful of small markdown
files is unnecessary weight. Section-level keyword matching is fast
and good enough for structured docs (pricing tables, FAQ entries). If
you later have dozens of long documents per topic, that's the point
where real embeddings would start to pay for themselves.
"""

import logging
import re
from pathlib import Path

log = logging.getLogger("knowledge_retriever")

# Topic files searched on every turn (all of them — no upstream
# classifier picks a subset anymore).
KNOWLEDGE_FILES = [
    "products_pricing.md",
    "ordering_payment.md",
    "delivery_shipping.md",
    "packaging.md",
    "refund_policy.md",
    "payment_issues.md",
    "faq.md",
    "recommendations.md",
]

MAX_CONTEXT_CHARS = 1200  # keep the LLM prompt small and fast
CORE_FILENAME = "core/always_on.md"


def _split_sections(text: str) -> list[str]:
    """Split a markdown doc into sections on ## headers (falls back to whole doc)."""
    sections = re.split(r"(?=^##\s+)", text, flags=re.MULTILINE)
    return [s.strip() for s in sections if s.strip()]


def _score_section(section: str, query_words: set[str]) -> int:
    """
    Word-boundary matching, not plain substring — otherwise a query
    word like "confirm" also matches unrelated occurrences of
    "confirmation" (boilerplate like "prepared after order
    confirmation" in product tables), diluting the score of the
    section the query is actually about.
    """
    section_lower = section.lower()
    return sum(
        1 for word in query_words
        if re.search(rf"\b{re.escape(word)}\b", section_lower)
    )


def get_core_context(company_dir: Path) -> str:
    """Always-on core knowledge — loaded in full on every turn."""
    core_path = company_dir / "knowledge" / CORE_FILENAME
    if not core_path.exists():
        log.warning(f"[KNOWLEDGE] missing always-on core file: {core_path}")
        return ""
    return core_path.read_text(encoding="utf-8")


def get_relevant_context(company_dir: Path, transcript: str) -> str:
    """
    Scores every section of every topic file in knowledge/ against the
    transcript and returns the top-matching sections, up to
    MAX_CONTEXT_CHARS. Returns "" if no file has any keyword overlap.
    """
    query_words = set(re.findall(r"\w+", transcript.lower()))
    if not query_words:
        return ""

    scored_sections = []  # list of (score, section_text)
    knowledge_dir = company_dir / "knowledge"

    for filename in KNOWLEDGE_FILES:
        file_path = knowledge_dir / filename
        if not file_path.exists():
            continue

        text = file_path.read_text(encoding="utf-8")
        for section in _split_sections(text):
            score = _score_section(section, query_words)
            if score > 0:
                scored_sections.append((score, section))

    if not scored_sections:
        log.debug(f"[KNOWLEDGE] no matching sections for: '{transcript}'")
        return ""

    scored_sections.sort(key=lambda item: item[0], reverse=True)

    context = ""
    for _, section in scored_sections:
        if len(context) + len(section) > MAX_CONTEXT_CHARS:
            break
        context += section + "\n\n"

    return context.strip() or scored_sections[0][1][:MAX_CONTEXT_CHARS]
