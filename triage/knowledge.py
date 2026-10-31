"""Failure classification, precedent search and fix drafting.

Everything here works offline with deterministic rules so the eval suite is reproducible
in CI. If ANTHROPIC_API_KEY is set, classification and fix wording can be delegated to
Claude (see llm.py); the rules stay as the fallback and as the baseline the eval compares to.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

CATEGORIES = ("schema_drift", "duplicate_merge", "missed_load", "permission", "unknown")

_RULES: list[tuple[str, re.Pattern]] = [
    ("permission", re.compile(r"insufficient privileges|access denied|not authori[sz]ed|permission denied", re.I)),
    ("schema_drift", re.compile(r"invalid identifier|column .* (not found|does not exist)|unknown column|renamed|dropped column|type mismatch|numeric value .* not recognized", re.I)),
    ("duplicate_merge", re.compile(r"duplicate|row count (doubled|2x)|merge .* (multiple|more than one)|non-?deterministic merge|unique key violat", re.I)),
    ("missed_load", re.compile(r"timed? ?out|timeout|not ready|skipped|did not run|no new files|throttl|connection reset", re.I)),
]


def classify(evidence_text: str, root_status: str | None) -> tuple[str, list[str]]:
    """Return (category, matched signals). Rule order encodes precedence."""
    signals: list[str] = []
    hit: str | None = None
    for cat, pat in _RULES:
        m = pat.search(evidence_text or "")
        if m:
            signals.append(f"{cat}: '{m.group(0)}'")
            hit = hit or cat
    if hit is None and root_status == "missing":
        hit, signals = "missed_load", ["status=missing"]
    return hit or "unknown", signals


# ---------------------------------------------------------------- precedent search

_TOKEN = re.compile(r"[a-z0-9_]+")
_STOP = {"the", "a", "an", "of", "to", "in", "on", "and", "for", "after", "with", "from", "is", "was", "be", "by"}


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


@dataclass
class Case:
    id: str
    category: str
    summary: str
    resolution: str


class CaseLibrary:
    """Tiny TF-IDF index over resolved cases. Swap for a vector store or Cortex Search in production."""

    def __init__(self, cases: list[Case], path: Path | None = None):
        self.cases = cases
        self.path = path
        self._reindex()

    @classmethod
    def load(cls, path: str | Path) -> "CaseLibrary":
        p = Path(path)
        raw = yaml.safe_load(p.read_text()) if p.exists() else []
        return cls([Case(**c) for c in (raw or [])], p)

    def _reindex(self) -> None:
        docs = [_tokens(f"{c.category} {c.summary}") for c in self.cases]
        df = Counter(t for d in docs for t in set(d))
        n = max(len(docs), 1)
        self._idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
        self._vecs = [self._vec(d) for d in docs]

    def _vec(self, toks: list[str]) -> dict[str, float]:
        tf = Counter(toks)
        v = {t: c * self._idf.get(t, 1.0) for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def search(self, query: str, k: int = 3) -> list[tuple[Case, float]]:
        q = self._vec(_tokens(query))
        scored = [(c, sum(q.get(t, 0.0) * w for t, w in v.items())) for c, v in zip(self.cases, self._vecs)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(c, round(s, 3)) for c, s in scored[:k] if s > 0]

    def add(self, case: Case, persist: bool = False) -> None:
        self.cases.append(case)
        self._reindex()
        if persist and self.path:
            self.path.write_text(yaml.safe_dump([c.__dict__ for c in self.cases], sort_keys=False))

    def next_id(self) -> str:
        nums = [int(m.group(1)) for c in self.cases if (m := re.search(r"(\d+)$", c.id))]
        return f"CASE-{(max(nums) + 1 if nums else 1):04d}"


# ---------------------------------------------------------------- fix drafting

_TEMPLATES = {
    "schema_drift": (
        "Update {target} to match the changed upstream contract{cause_clause}; add a backward-compatible alias, "
        "then re-run {target} and refresh {downstream}."
    ),
    "duplicate_merge": (
        "Tighten the MERGE key in {target} so versioned source rows match one target row; keep the latest version, "
        "remove duplicates written by the last run, then refresh {downstream}."
    ),
    "missed_load": (
        "Re-run {target} with a backfill from its last successful run, then trigger {downstream}. "
        "Transient failure: no code change."
    ),
    "permission": (
        "Restore the missing grant used by {target} for its service role, re-run {target}, then refresh {downstream}."
    ),
}

# How much a category's template can be trusted before evidence is counted. Deliberately conservative.
_BASE_CONFIDENCE = {"schema_drift": 0.6, "duplicate_merge": 0.55, "missed_load": 0.65, "permission": 0.6}


def draft_fix(category: str, target: str, cause: str | None, downstream: list[str]) -> str | None:
    tpl = _TEMPLATES.get(category)
    if not tpl:
        return None
    ds = ", ".join(downstream) if downstream else "dependent nodes"
    cause_clause = f" (cause: {cause})" if cause else ""
    return tpl.format(target=target, cause_clause=cause_clause, downstream=ds)


def confidence(category: str, ambiguous: bool, top_precedent: float, precedent_agrees: bool) -> float:
    """Heuristic, not a probability. Documented so a reviewer knows exactly what moves it."""
    if category not in _BASE_CONFIDENCE:
        return 0.0
    c = _BASE_CONFIDENCE[category]
    if precedent_agrees:
        c += 0.25 * min(top_precedent, 1.0)
    if ambiguous:
        c -= 0.3
    return round(max(0.0, min(c, 0.99)), 2)
