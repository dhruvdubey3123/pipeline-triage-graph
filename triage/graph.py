"""The triage graph.

    intake -> tracer -> analyst -> precedent -> classifier -> orchestrator -> reviewer
                                                                              |
                                              escalate <------ (blocked) -----+---- (passed) ---> [human gate]
                                                  |                                                  |
                                                  |                                approve -> apply -+
                                                  |                                reject ---------> escalate
                                                  v                                                  v
                                                learn <-------------------------------------------- learn -> END

The graph is compiled with interrupt_before=["gate"], so a run always pauses before any
change is made. A person resumes it with approve or reject (see TriageService.decide).
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from . import llm
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
):
    probe = probe or FixtureProbe()
    executor = executor or DryRunExecutor()

    def intake(s: TriageState) -> dict:
        inc = s["incident"]
        if inc.get("symptom_node") not in dag.nodes:
            raise ValueError(f"symptom_node {inc.get('symptom_node')!r} is not in pipeline {dag.name}")
        return _say("intake", f"{inc.get('id', 'incident')}: {inc.get('symptom', '')} (at {inc['symptom_node']})")

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
    for name, fn in [("intake", intake), ("tracer", tracer), ("analyst", analyst), ("precedent", precedent),
                     ("classifier", classifier), ("orchestrator", orchestrator), ("reviewer", reviewer),
                     ("gate", gate), ("apply", apply), ("escalate", escalate), ("learn", learn)]:
        g.add_node(name, fn)
    g.add_edge(START, "intake")
    for a, b in [("intake", "tracer"), ("tracer", "analyst"), ("analyst", "precedent"),
                 ("precedent", "classifier"), ("classifier", "orchestrator"), ("orchestrator", "reviewer")]:
        g.add_edge(a, b)
    g.add_conditional_edges("reviewer", lambda s: s["route"], {"gate": "gate", "escalate": "escalate"})
    g.add_conditional_edges("gate", lambda s: "apply" if s.get("human_decision") == "approve" else "escalate",
                            {"apply": "apply", "escalate": "escalate"})
    g.add_edge("apply", "learn")
    g.add_edge("escalate", "learn")
    g.add_edge("learn", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver(), interrupt_before=["gate"])
