"""The triage graph.

    intake -> [investigator] -> tracer -> analyst -> precedent -> classifier -> orchestrator -> reviewer
                                                                              |
                                              escalate <------ (blocked) -----+---- (passed) ---> [human gate]
                                                  |                                                  |
                                                  |                                approve -> apply -+
                                                  |                                reject ---------> escalate
                                                  v                                                  v
                                                learn <-------------------------------------------- learn -> END

The graph is compiled with interrupt_before=["gate"], so a run always pauses before any
change is made. A person resumes it with approve or reject (see TriageService.decide).

[investigator] is there only when a chat model is passed in. It is an LLM agent with read-only
tools that walks the dependency graph itself and submits a diagnosis; its root cause then drives
the plan, while the failure category stays with the rules.
The tracer still runs after it, as a cross-check: if the two disagree on the root cause, the
reviewer sends the incident to a person with both traces instead of to the approval gate.

communicator, after the reviewer, is there only when a chat model is passed in. It is the role the
LLM had in the original Allegis agent: the traversal stays deterministic, and the model writes the
briefing the person reads, checked against the trace before it is shown (see triage/communicator.py).
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from . import llm
from .communicator import brief
from .investigator import ChatModel, investigate
from .adapters import DataProbe, DryRunExecutor, Executor, FixtureProbe
from .dag import PipelineDAG, refresh_order, trace
from .knowledge import Case, CaseLibrary, classify, confidence, draft_fix


class TriageState(TypedDict, total=False):
    incident: dict
    log: Annotated[list[dict], operator.add]
    hops: list[dict]
    fix_target: str | None
    upstream_cause: str | None
    ambiguous: bool
    observations: dict[str, str]
    evidence: str
    precedents: list[dict]
    category: str
    signals: list[str]
    proposal: str | None
    plan: list[str]
    confidence: float
    review_issues: list[str]
    route: Literal["gate", "escalate"]
    human_decision: Literal["approve", "reject"] | None
    reviewer_name: str | None
    applied: list[str]
    outcome: Literal["resolved", "escalated"]
    learned_case: str | None
    investigation: dict            # the LLM investigator's calls and answer, when one is configured
    tracer_target: str | None      # what the rule-based tracer found, kept as a cross-check
    tracer_cause: str | None
    briefing: str                  # what the person at the gate reads, when a communicator is configured
    briefing_source: str           # "model" if it passed the faithfulness check, else "template"
    briefing_problems: list[str]


def _short(args: dict) -> str:
    return ", ".join(f"{k}={str(v)[:40]!r}" for k, v in (args or {}).items())


def _say(agent: str, msg: str) -> dict:
    return {"log": [{"agent": agent, "msg": msg}]}


def build_graph(
    dag: PipelineDAG,
    library: CaseLibrary,
    probe: DataProbe | None = None,
    executor: Executor | None = None,
    checkpointer: Any | None = None,
    min_confidence: float = 0.5,
    persist_learning: bool = False,
    investigator: ChatModel | None = None,
    investigator_steps: int = 14,
    communicator: ChatModel | None = None,
):
    """investigator: a chat model with tool calling. When set, an LLM agent walks the dependency graph
    first and its diagnosis drives the plan; the rule-based tracer still runs, as a cross-check, and a
    disagreement on the root cause goes to a person instead of to the approval gate."""
    probe = probe or FixtureProbe()
    executor = executor or DryRunExecutor()

    def intake(s: TriageState) -> dict:
        inc = s["incident"]
        if inc.get("symptom_node") not in dag.nodes:
            raise ValueError(f"symptom_node {inc.get('symptom_node')!r} is not in pipeline {dag.name}")
        return _say("intake", f"{inc.get('id', 'incident')}: {inc.get('symptom', '')} (at {inc['symptom_node']})")

    def investigator_node(s: TriageState) -> dict:
        inv = investigate(s["incident"], dag, library, investigator, max_steps=investigator_steps)
        lines = [{"agent": "investigator", "msg": f"{st['tool']}({_short(st['args'])}) -> "
                  + ("ok" if st["ok"] else st["result"])} for st in inv.steps]
        lines.append({"agent": "investigator", "msg": (
            f"diagnosis: fix {inv.fix_target} ({inv.category}), cause {inv.cause or 'none named'}: {inv.reason}"
            if inv.submitted else f"no diagnosis submitted after {len(inv.steps)} calls")})
        return {"investigation": vars(inv), "log": lines}

    def tracer(s: TriageState) -> dict:
        inc = s["incident"]
        res = trace(dag, inc.get("node_status", {}), inc["symptom_node"])
        hops = [h.__dict__ for h in res.hops]
        lines = [
            {"agent": "tracer", "msg": f"{h.node}: {'ruled out, healthy' if h.ruled_out else h.status} ({h.detail})"}
            for h in res.hops
        ]
        if res.fix_target:
            tail = f"root cause at {res.fix_target}" + (f", caused upstream by {res.upstream_cause}" if res.upstream_cause else "")
            if res.ambiguous:
                tail += f"; ambiguous between {res.candidates}"
        else:
            tail = "no unhealthy upstream node found"
        lines.append({"agent": "tracer", "msg": tail})
        inv = s.get("investigation")
        if inv is not None:
            agree = inv["submitted"] and inv["fix_target"] == res.fix_target
            lines.append({"agent": "tracer", "msg": "cross-check: " + (
                "agrees with the investigator" if agree else
                f"investigator says {inv['fix_target']}, tracer says {res.fix_target}")})
            return {"hops": hops, "fix_target": inv["fix_target"] if inv["submitted"] else None,
                    "upstream_cause": (inv["cause"] or None) if inv["submitted"] else None,
                    "ambiguous": res.ambiguous, "tracer_target": res.fix_target,
                    "tracer_cause": res.upstream_cause, "log": lines}
        return {"hops": hops, "fix_target": res.fix_target, "upstream_cause": res.upstream_cause,
                "ambiguous": res.ambiguous, "log": lines}

    def analyst(s: TriageState) -> dict:
        inc, obs = s["incident"], {}
        for nid in filter(None, [s.get("fix_target"), s.get("upstream_cause"), *dag.parents[inc["symptom_node"]]]):
            r = probe.check(nid, inc)
            if r:
                obs[nid] = r
        status = inc.get("node_status", {})
        parts = []
        for nid in filter(None, [s.get("fix_target"), s.get("upstream_cause")]):
            st = status.get(nid, {})
            parts += [st.get("error", ""), st.get("note", ""), " ".join(st.get("changes", []))]
        parts += list(obs.values()) + [inc.get("symptom", "")]
        evidence = " | ".join(p for p in parts if p)
        msg = "; ".join(f"{k}: {v}" for k, v in obs.items()) or "no data observations available"
        return {"observations": obs, "evidence": evidence, **_say("analyst", msg)}

    def precedent(s: TriageState) -> dict:
        hits = library.search(s.get("evidence", ""), k=3)
        prec = [{"id": c.id, "category": c.category, "summary": c.summary, "resolution": c.resolution, "score": sc}
                for c, sc in hits]
        msg = "; ".join(f"{p['id']} ({p['category']}, score {p['score']})" for p in prec) or "no similar resolved cases"
        return {"precedents": prec, **_say("precedent", msg)}

    def classifier(s: TriageState) -> dict:
        root_status = s["incident"].get("node_status", {}).get(s.get("fix_target") or "", {}).get("status")
        cat, signals = classify(s.get("evidence", ""), root_status)
        inv = s.get("investigation")
        if inv is not None and inv["submitted"] and inv["category"] != cat:
            # The category stays with the rules. The model has to pick one from a list, so on a failure
            # nobody has seen before it names a familiar one, and "unknown" is what sends an incident to
            # a person. Measured on incidents 41-60, that hid 4 of 7 escalations; rules 17/17 vs model 9/17 on category. Its category is kept
            # here as a second opinion instead.
            signals.append(f"investigator said {inv['category']}; rules said {cat}")
        if llm.enabled():
            llm_cat = llm.classify(s.get("evidence", ""), s.get("precedents", []))
            if llm_cat and llm_cat != cat:
                signals.append(f"llm suggested {llm_cat}; rules said {cat}")
                cat = llm_cat
        return {"category": cat, "signals": signals, **_say("classifier", f"{cat} ({', '.join(signals) or 'no signals'})")}

    def orchestrator(s: TriageState) -> dict:
        target = s.get("fix_target")
        if not target:
            return {"proposal": None, "plan": [], "confidence": 0.0, **_say("orchestrator", "nothing to fix: no root found")}
        downstream = [n for n in refresh_order(dag, target) if n != target]
        draft = draft_fix(s["category"], target, s.get("upstream_cause"), downstream)
        prec = s.get("precedents", [])
        top = prec[0]["score"] if prec else 0.0
        agrees = bool(prec) and prec[0]["category"] == s["category"]
        conf = confidence(s["category"], s.get("ambiguous", False), top, agrees)
        if draft and llm.enabled():
            draft = llm.word_fix(draft, s.get("evidence", "")) or draft
        plan = [f"fix {target}", *[f"refresh {n}" for n in downstream]] if draft else []
        return {"proposal": draft, "plan": plan, "confidence": conf,
                **_say("orchestrator", f"{draft or 'no template for this category'} (confidence {conf})")}

    def reviewer(s: TriageState) -> dict:
        issues: list[str] = []
        target = s.get("fix_target")
        inv = s.get("investigation")
        if inv is not None and not inv["submitted"]:
            issues.append("the investigator submitted no diagnosis")
        elif inv is not None and s.get("tracer_target") != target:
            issues.append(f"investigator and tracer disagree on the root cause ({target} vs "
                          f"{s.get('tracer_target')}); a person should decide from both traces")
        if not target:
            issues.append("no root cause")
        elif target in dag.protected:
            issues.append(f"{target} is protected; changes must go through its producing job")
        if s.get("category") == "unknown":
            issues.append("unrecognised failure category")
        if not s.get("proposal"):
            issues.append("no proposal")
        if s.get("ambiguous"):
            issues.append("trace ambiguous between multiple candidates")
        if s.get("confidence", 0.0) < min_confidence:
            issues.append(f"confidence {s.get('confidence', 0.0)} below threshold {min_confidence}")
        for step in s.get("plan", []):
            node = step.split(" ", 1)[1]
            if step.startswith("fix") and node != target:
                issues.append(f"plan modifies {node}, which is not the root")
        route = "escalate" if issues else "gate"
        msg = "passed, waiting for human approval" if not issues else "blocked: " + "; ".join(issues)
        return {"review_issues": issues, "route": route, **_say("reviewer", msg)}

    def communicator_node(s: TriageState) -> dict:
        b = brief(s, dag, communicator)
        note = ("briefing written by the model, checked against the trace" if b.source == "model" else
                "model briefing not used (" + "; ".join(b.problems) + "), template briefing instead")
        return {"briefing": b.text, "briefing_source": b.source, "briefing_problems": b.problems,
                **_say("communicator", note)}

    def gate(s: TriageState) -> dict:
        d = s.get("human_decision")
        if d not in ("approve", "reject"):
            raise ValueError("gate resumed without a human decision")
        who = s.get("reviewer_name") or "reviewer"
        return _say("human", f"{who} chose {d}")

    def apply(s: TriageState) -> dict:
        done = executor.apply(s.get("plan", []))
        return {"applied": done, "outcome": "resolved", **_say("apply", f"{len(done)} steps: " + "; ".join(done))}

    def escalate(s: TriageState) -> dict:
        owner = dag.nodes[s["fix_target"]].owner if s.get("fix_target") else "on-call"
        why = "; ".join(s.get("review_issues") or []) or "rejected by human reviewer"
        return {"outcome": "escalated", **_say("escalate", f"handed to {owner or 'on-call'} with full trace: {why}")}

    def learn(s: TriageState) -> dict:
        if s.get("category", "unknown") == "unknown" or not s.get("fix_target"):
            return {"learned_case": None, **_say("learn", "not added to case library (no confirmed category)")}
        cid = library.next_id()
        summary = f"{s['fix_target']} {s['category']}: {s.get('evidence', '')[:200]}"
        resolution = s.get("proposal") or ""
        if s.get("outcome") == "escalated":
            resolution = "escalated: " + resolution
        library.add(Case(cid, s["category"], summary, resolution), persist=persist_learning)
        return {"learned_case": cid, **_say("learn", f"added {cid} to case library")}

    g = StateGraph(TriageState)
    extra = [("investigator", investigator_node)] if investigator is not None else []
    if communicator is not None:
        extra.append(("communicator", communicator_node))
    for name, fn in [*extra, ("intake", intake), ("tracer", tracer), ("analyst", analyst), ("precedent", precedent),
                     ("classifier", classifier), ("orchestrator", orchestrator), ("reviewer", reviewer),
                     ("gate", gate), ("apply", apply), ("escalate", escalate), ("learn", learn)]:
        g.add_node(name, fn)
    g.add_edge(START, "intake")
    head = [("intake", "investigator"), ("investigator", "tracer")] if investigator is not None else [("intake", "tracer")]
    for a, b in [*head, ("tracer", "analyst"), ("analyst", "precedent"),
                 ("precedent", "classifier"), ("classifier", "orchestrator"), ("orchestrator", "reviewer")]:
        g.add_edge(a, b)
    decider = "reviewer"
    if communicator is not None:      # the briefing is written for whoever decides next: approver or owner
        g.add_edge("reviewer", "communicator")
        decider = "communicator"
    g.add_conditional_edges(decider, lambda s: s["route"], {"gate": "gate", "escalate": "escalate"})
    g.add_conditional_edges("gate", lambda s: "apply" if s.get("human_decision") == "approve" else "escalate",
                            {"apply": "apply", "escalate": "escalate"})
    g.add_edge("apply", "learn")
    g.add_edge("escalate", "learn")
    g.add_edge("learn", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver(), interrupt_before=["gate"])
