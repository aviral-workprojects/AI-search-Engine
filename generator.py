"""
Generator Module — Structured Answer Generation with Citations
Supports three answer modes: short / balanced / detailed
Includes query understanding and fact extraction helpers.
"""

import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# DECODING CONFIGS
# ─────────────────────────────────────────────────────────────

DECODING_CONFIGS = {
    "qa":            {"temperature": 0.20, "top_p": 0.85, "max_tokens": 600},
    "summarization": {"temperature": 0.30, "top_p": 0.90, "max_tokens": 450},
    "general":       {"temperature": 0.30, "top_p": 0.90, "max_tokens": 600},
}

# ─────────────────────────────────────────────────────────────
# ANSWER MODES
# ─────────────────────────────────────────────────────────────

ANSWER_MODES = {
    "short": {
        "label": "Short",
        "description": "Concise 2–3 sentence summary only",
        "max_tokens": 200,
    },
    "balanced": {
        "label": "Balanced",
        "description": "Summary + 2 key bullet points",
        "max_tokens": 400,
    },
    "detailed": {
        "label": "Detailed",
        "description": "Summary + 5 structured bullet points with citations",
        "max_tokens": 700,
    },
}


def _build_prompt(
    query: str,
    context: str,
    task_type: str = "qa",
    answer_mode: str = "balanced",
) -> str:
    """
    Build the LLM prompt based on task type and answer mode.

    answer_mode options:
        "short"    → brief summary only
        "balanced" → summary + 2 key points
        "detailed" → full structured report with 5 points
    """
    today = datetime.now().strftime("%B %d, %Y")

    # ── Shared rules block ──────────────────────────────────
    rules = """STRICT RULES:
1. Use ONLY the provided sources — never add outside knowledge.
2. Cite every factual claim with [1], [2], [3] … notation.
3. If a source does not cover a sub-topic, omit that section.
4. Preserve all numbers, dates, names, and statistics exactly.
5. Do NOT hallucinate. If uncertain, say so."""

    # ── Mode-specific output instructions ───────────────────
    if answer_mode == "short":
        output_format = f"""\
OUTPUT FORMAT — SHORT:
Write a single concise paragraph (2–4 sentences) that captures the most \
important fact or development. Start with "As of {today}, …"
End with a parenthetical citation list, e.g. [1][3]."""

    elif answer_mode == "balanced":
        output_format = f"""\
OUTPUT FORMAT — BALANCED:
**Summary** (2–3 sentences, start with "As of {today}, …")

**Key Points**
- [Point 1 with citation]
- [Point 2 with citation]"""

    else:  # detailed
        output_format = f"""\
OUTPUT FORMAT — DETAILED RESEARCH REPORT:

Start with: "As of {today}, …" (1–2 sentence overview)

**1. Latest Developments**
- 3–5 bullet points, each with a citation

**2. Key Facts & Figures**
- Numerical data, statistics, or named entities from the sources

**3. Context & Analysis**
- Background and significance (2–4 sentences, cite sources)

**4. Outlook / Implications**
- What happens next (1–3 sentences)

**5. Sources**
Numbered list matching the [1], [2]… citations used above."""

    return f"""You are an expert analyst synthesising information from multiple news sources.

{rules}

USER QUESTION:
{query}

SOURCES:
{context}

{output_format}

ANSWER:"""


# ─────────────────────────────────────────────────────────────
# LLM CALL WRAPPERS
# ─────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _call_groq(prompt: str, groq_client, config: Dict) -> str:
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=config["temperature"],
        top_p=config["top_p"],
        max_tokens=config["max_tokens"],
    )
    return response.choices[0].message.content.strip()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _call_gemini(prompt: str, gemini_model, config: Dict) -> str:
    generation_config = {
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "max_output_tokens": config["max_tokens"],
    }
    response = gemini_model.generate_content(prompt, generation_config=generation_config)
    return response.text.strip()


# ─────────────────────────────────────────────────────────────
# PUBLIC GENERATE FUNCTION
# ─────────────────────────────────────────────────────────────

def generate_answer(
    query: str,
    context: str,
    llm_choice: str,
    groq_client,
    gemini_model,
    task_type: str = "qa",
    answer_mode: str = "balanced",
) -> str:
    """
    Generate an answer with automatic LLM fallback.

    Args:
        query:        User question.
        context:      Formatted source context (from build_context_advanced).
        llm_choice:   "Groq" or "Gemini".
        groq_client:  Groq client instance (or None).
        gemini_model: Gemini model instance (or None).
        task_type:    "qa" | "summarization" | "general"
        answer_mode:  "short" | "balanced" | "detailed"

    Returns:
        Answer string with inline citations.
    """
    logger.info(f"Generating answer | LLM={llm_choice} | mode={answer_mode} | task={task_type}")

    # Merge task config with mode max_tokens (mode takes priority)
    config = dict(DECODING_CONFIGS.get(task_type, DECODING_CONFIGS["general"]))
    mode_tokens = ANSWER_MODES.get(answer_mode, ANSWER_MODES["balanced"])["max_tokens"]
    config["max_tokens"] = max(config["max_tokens"], mode_tokens)

    prompt = _build_prompt(query, context, task_type, answer_mode)

    # ── Primary LLM ─────────────────────────────────────────
    if llm_choice == "Groq" and groq_client:
        try:
            return _call_groq(prompt, groq_client, config)
        except Exception as e:
            logger.warning(f"Groq failed ({e}), trying Gemini fallback")
            if gemini_model:
                return _call_gemini(prompt, gemini_model, config)

    elif llm_choice == "Gemini" and gemini_model:
        try:
            return _call_gemini(prompt, gemini_model, config)
        except Exception as e:
            logger.warning(f"Gemini failed ({e})")

    return "❌ No LLM available. Please configure Groq or Gemini API keys in your .env file."


# ─────────────────────────────────────────────────────────────
# QUERY UNDERSTANDING
# ─────────────────────────────────────────────────────────────

def analyze_query(query: str) -> Dict:
    """
    Lightweight query understanding without an external model.
    Returns intent signals used to tune retrieval and generation.

    Returns:
        {
          "is_news":          bool,   # mentions temporal intent
          "needs_timeline":   bool,   # event/sequence likely
          "needs_numbers":    bool,   # numeric facts likely
          "needs_comparison": bool,   # vs / compare present
          "entities":         List[str],  # capitalised tokens
        }
    """
    q = query.lower()
    temporal_words = {"latest", "recent", "today", "now", "current", "2024", "2025", "2026"}
    timeline_words = {"timeline", "history", "development", "progress", "sequence", "when"}
    number_words   = {"price", "cost", "earnings", "revenue", "percent", "rate", "number",
                      "statistic", "data", "how many", "how much", "stock"}
    compare_words  = {"vs", "versus", "compare", "difference", "better", "worse", "between"}

    entities = [
        token for token in query.split()
        if token[0].isupper() and len(token) > 2 and token.lower() not in
        {"the", "a", "an", "in", "of", "for", "and", "or", "is", "are", "was"}
    ]

    return {
        "is_news":          bool(temporal_words & set(q.split())),
        "needs_timeline":   bool(timeline_words & set(q.split())),
        "needs_numbers":    any(w in q for w in number_words),
        "needs_comparison": any(w in q for w in compare_words),
        "entities":         entities[:8],
    }


# ─────────────────────────────────────────────────────────────
# FACT EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_facts(article_text: str) -> Dict:
    """
    Extract structured facts from a single article.
    Used to build richer context before LLM generation.

    Returns:
        {
          "numbers": List[str],   # numeric expressions
          "events":  List[str],   # first 5 sentences
          "dates":   List[str],   # date-like strings
        }
    """
    facts: Dict[str, List] = {"numbers": [], "events": [], "dates": []}

    # Numbers with optional commas / units
    facts["numbers"] = re.findall(r"\b\d[\d,]*(?:\.\d+)?(?:\s?[%$£€BMKbmk])?\b", article_text)[:15]

    # Dates (simple patterns)
    facts["dates"] = re.findall(
        r"\b(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{1,2}(?:,\s*\d{4})?|\b\d{4}\b",
        article_text,
    )[:10]

    # Leading sentences as "events"
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", article_text) if len(s.strip()) > 30]
    facts["events"] = sentences[:5]

    return facts


# ─────────────────────────────────────────────────────────────
# CITATION EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_citations(answer: str) -> List[int]:
    """Return sorted, deduplicated list of citation numbers from an answer."""
    return sorted(set(int(c) for c in re.findall(r"\[(\d+)\]", answer)))