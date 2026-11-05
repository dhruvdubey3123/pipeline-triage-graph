"""The communicator: the LLM explains, the trace stays the facts.

This is how the original triage agent at Allegis divided the work: traversal of the dependency graph was
deterministic, a stored procedure, because a wrong root cause is expensive, and the LLM was used to
communicate what the traversal found. Here the tracer plays the stored procedure and this step plays the
LLM. It is given the findings, nothing else, and writes the briefing the person at the approval gate
reads: what they saw, what actually broke and why, what the plan does, and what the reviewer flagged.

A briefing is still a place a model can go wrong, so it is checked before anyone sees it:

  * it may only name tables and jobs the trace touched or the plan changes, so it cannot invent a step
    or point at a job nobody examined;
  * it has to name the root cause, so the one fact the reader needs is never paraphrased away;
  * it may not call the failure a different category from the one the findings settled on.

A briefing that fails any check is replaced by a template built from the same findings, and the reason
is recorded. The person always gets a briefing; only a faithful one comes from the model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .dag import PipelineDAG
from .investigator import ChatModel
from .knowledge import CATEGORIES

INSTRUCTIONS = (
    "You brief the on-call data engineer about a pipeline incident. You are given the findings of a "
    "deterministic trace of the pipeline. In three or four plain sentences, say what the person saw, which "
    "job actually broke and why, what the plan does, and anything the reviewer flagged. Use only the facts "
    "given. Name only tables and jobs that appear in the findings. Do not add steps and do not guess at "
    "causes the findings do not state. Plain text, no headings, no lists."
)
MAX_CHARS = 1500


@dataclass
class Briefing:
    text: str
    source: str                                   # "model" or "template"
    problems: list[str] = field(default_factory=list)
    model_text: str = ""                          # what the model wrote, kept even when it was not used


def facts(state: dict, dag: PipelineDAG) -> dict:
    """Everything the briefing may draw on, and nothing else."""
    inc = state.get("incident") or {}
    hops = state.get("hops") or []
    plan = state.get("plan") or []
    plan_nodes = [step.split(" ", 1)[1] for step in plan if " " in step]
    allowed = {h["node"] for h in hops} | set(plan_nodes)
    allowed |= {n for n in (state.get("fix_target"), state.get("upstream_cause"), inc.get("symptom_node")) if n}
    issues = state.get("review_issues") or []
    return {
        "incident": inc.get("id"),
        "what the person saw": f"{inc.get('symptom', '')} at {inc.get('symptom_node')}",
        "path walked upstream": [f"{h['node']}: " + ("healthy, ruled out" if h.get("ruled_out")
                                                     else f"{h['status']} ({h['detail']})") for h in hops],
        "root cause": state.get("fix_target"),
        "caused upstream by": state.get("upstream_cause"),
        "category": state.get("category"),
        "confidence": state.get("confidence"),
        "evidence": state.get("evidence"),
        "similar resolved cases": [f"{p['id']} ({p['category']}): {p['resolution']}"
                                   for p in (state.get("precedents") or [])[:2]],
        "plan": plan,
        "reviewer flagged": issues,
        "next": "waiting for a person to approve" if state.get("route") == "gate" else "escalated to a person",
        "_allowed_nodes": sorted(allowed),
    }


def template(f: dict) -> str:
    """The briefing built straight from the findings: always faithful, never fluent."""
    parts = [f"{f['incident']}: {f['what the person saw']}."]
    if f["root cause"]:
        cause = f" after an upstream change in {f['caused upstream by']}" if f["caused upstream by"] else ""
        parts.append(f"The trace found the failure at {f['root cause']}{cause} ({f['category']}, "
                     f"confidence {f['confidence']}).")
    else:
        parts.append("The trace found no unhealthy upstream node.")
    if f["plan"]:
        parts.append("Plan: " + ", then ".join(f["plan"]) + ".")
    if f["similar resolved cases"]:
        parts.append("Closest past case: " + f["similar resolved cases"][0] + ".")
    if f["reviewer flagged"]:
        parts.append("Flagged: " + "; ".join(f["reviewer flagged"]) + ".")
    parts.append(f["next"].capitalize() + ".")
    return " ".join(parts)


def _mentions(text: str, name: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text, flags=re.I) is not None


def check(text: str, f: dict, dag: PipelineDAG) -> list[str]:
    """Why a briefing cannot be shown as written. Empty when it is faithful to the findings."""
    if not text.strip():
        return ["empty"]
    problems = []
    if len(text) > MAX_CHARS:
        problems.append(f"longer than {MAX_CHARS} characters")
    allowed = set(f["_allowed_nodes"])
    for node in dag.nodes:
        if node not in allowed and _mentions(text, node):
            problems.append(f"names {node}, which the trace never touched and the plan does not change")
    if f["root cause"] and not _mentions(text, f["root cause"]):
        problems.append(f"does not name the root cause, {f['root cause']}")
    for cat in CATEGORIES:
        if cat in ("unknown", f["category"]):
            continue
        if _mentions(text, cat) or _mentions(text, cat.replace("_", " ")):
            problems.append(f"calls it {cat}; the findings say {f['category']}")
    return problems


def brief(state: dict, dag: PipelineDAG, model: ChatModel) -> Briefing:
    f = facts(state, dag)
    shown = {k: v for k, v in f.items() if not k.startswith("_")}
    lines = [f"{k}: {v}" for k, v in shown.items() if v not in (None, [], "")]
    try:
        msg = model.chat([{"role": "system", "content": INSTRUCTIONS},
                          {"role": "user", "content": "Findings:\n" + "\n".join(lines)}], [])
    except Exception as e:                       # no model server: the person still gets a briefing
        return Briefing(template(f), "template", [f"model unavailable ({type(e).__name__})"])
    text = re.sub(r"<think>.*?</think>", "", msg.get("content") or "", flags=re.S).strip()
    problems = check(text, f, dag)
    return Briefing(template(f) if problems else text, "template" if problems else "model", problems, text)
