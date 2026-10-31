"""Optional LLM layer. Used only when ANTHROPIC_API_KEY is set and TRIAGE_USE_LLM=1.

The LLM never decides the root cause: the tracer does that from graph structure. The LLM is
asked two narrow questions (which category, and how to word the fix) and its answer is
validated against the allowed categories before it is used. Any error falls back to rules.
"""
from __future__ import annotations

import json
import os

from .knowledge import CATEGORIES


def enabled() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY")) and os.getenv("TRIAGE_USE_LLM") == "1"


def _client():
    import anthropic  # imported lazily so the offline path has no dependency on it

    return anthropic.Anthropic()


def _model() -> str:
    return os.environ.get("TRIAGE_MODEL", "claude-sonnet-4-5")


def classify(evidence: str, precedents: list[dict]) -> str | None:
    prompt = (
        "You classify data pipeline failures. Allowed categories: "
        + ", ".join(CATEGORIES)
        + ".\nEvidence from the failing node and its upstream:\n"
        + evidence
        + "\nSimilar resolved cases:\n"
        + json.dumps(precedents, indent=1)
        + '\nReply with JSON only: {"category": "<one allowed category>"}'
    )
    try:
        msg = _client().messages.create(model=_model(), max_tokens=50, messages=[{"role": "user", "content": prompt}])
        cat = json.loads(msg.content[0].text).get("category")
        return cat if cat in CATEGORIES else None
    except Exception:
        return None


def word_fix(draft: str, evidence: str) -> str | None:
    prompt = (
        "Rewrite this proposed pipeline fix as two short imperative sentences for an on-call engineer. "
        "Do not add steps that are not in the draft.\nDraft: " + draft + "\nEvidence: " + evidence
    )
    try:
        msg = _client().messages.create(model=_model(), max_tokens=200, messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text.strip() or None
    except Exception:
        return None
