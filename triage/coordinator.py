"""The coordinator: the LLM decides what else to look at; the rules have already decided.

In the original Allegis agent the LLM orchestrated: it chose which tools to call while the traversal of
the dependency graph stayed a deterministic stored procedure. This is that role, placed so it cannot
move a decision. It runs after the reviewer, when the root cause, category, plan and route are fixed,
and it gets read-only, deterministic tools to gather whatever the person deciding should also know:

  status(node)       what the incident reports about a node
  upstream(node)     everything a node depends on, with statuses
  downstream(node)   everything that reads from a node: the blast radius of the failure and of the fix
  probe(node)        a data check on a node, through the same DataProbe the analyst uses
  precedents(query)  resolved cases matching its own query, say the exact error text
  finish()           done looking

It chooses which to call, on what, in what order and when to stop. What it finds is recorded as extra
checks and handed to the communicator for the briefing. Nothing it does is read by the steps that
decide, which a test asserts on every labelled incident: an erratic coordinator leaves every decision
exactly as the rules made it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .adapters import DataProbe
from .dag import PipelineDAG
from .investigator import ChatModel, NodeArg, NoArgs, Query, ToolSpec, _loads, run_tool
from .investigator import tools as investigator_tools
from .knowledge import CaseLibrary

INSTRUCTIONS = (
    "You coordinate the follow-up checks on a pipeline incident that has already been diagnosed. The root cause, "
    "category and plan below are decided; you cannot change them. Decide what else the person approving or owning "
    "this incident should know before acting: what else is affected downstream, whether the data around the root "
    "cause looks as expected, whether past incidents with the same error went differently. Call the tools you need, "
    "one at a time, then call finish. Use at most five checks."
)
TERMINAL = "finish"


@dataclass
class Checks:
    calls: list[dict] = field(default_factory=list)   # tool, args, ok, result
    finished: bool = False

    def facts(self) -> list[str]:
        return [f"{c['tool']}({', '.join(f'{v}' for v in c['args'].values())}): {c['result']}"
                for c in self.calls if c["ok"] and c["tool"] != TERMINAL]

    def nodes(self, dag: PipelineDAG) -> set[str]:
        """Every node a check looked at or reported on, which a briefing may now mention as context."""
        seen = set()
        for c in self.calls:
            if not c["ok"]:
                continue
            text = " ".join(str(v) for v in c["args"].values()) + " " + c["result"]
            seen |= {n for n in dag.nodes if n in text}
        return seen


def tools(dag: PipelineDAG, library: CaseLibrary, incident: dict, probe: DataProbe) -> list[ToolSpec]:
    shared = {t.name: t for t in investigator_tools(dag, library, incident)}

    def known(node: str) -> None:
        if node not in dag.nodes:
            raise ValueError(f"'{node}' is not in this pipeline")

    def downstream(node: str) -> str:
        known(node)
        below = dag.downstream(node)
        reported = incident.get("node_status", {})
        return "\n".join(f"{n}: {reported.get(n, {}).get('status', 'nothing reported')}" for n in below) \
            or "nothing reads from it"

    def probe_node(node: str) -> str:
        known(node)
        return probe.check(node, incident) or "no data observation recorded for this node"

    def finish() -> str:
        return "done"

    return [
        shared["status"], shared["upstream"],
        ToolSpec("downstream", "Everything that reads from this node, with its reported status.", NodeArg, downstream),
        ToolSpec("probe", "Run the data check on one node.", NodeArg, probe_node),
        ToolSpec("precedents", "Resolved cases matching your query, with how each was fixed.", Query,
                 shared["precedents"].fn),
        ToolSpec(TERMINAL, "Stop: you have checked what the person should know.", NoArgs, finish),
    ]


def coordinate(state: dict, dag: PipelineDAG, library: CaseLibrary, probe: DataProbe, model: ChatModel,
               max_checks: int = 6) -> Checks:
    incident = state["incident"]
    specs = {t.name: t for t in tools(dag, library, incident, probe)}
    decided = (f"Incident {incident.get('id')}: {incident.get('symptom')} at {incident['symptom_node']}.\n"
               f"Root cause: {state.get('fix_target')}. Caused upstream by: {state.get('upstream_cause')}.\n"
               f"Category: {state.get('category')}. Plan: {', then '.join(state.get('plan') or []) or 'none'}.\n"
               f"Reviewer: {'; '.join(state.get('review_issues') or []) or 'passed'}. "
               f"Next: {'approval' if state.get('route') == 'gate' else 'escalation to a person'}.")
    messages = [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": decided}]
    out = Checks()
    try:
        for n in range(max_checks):
            msg = model.chat(messages, [t.schema() for t in specs.values()])
            calls = msg.get("tool_calls") or []
            if not calls:
                break
            call = calls[0]
            name, raw = call["function"]["name"], call["function"].get("arguments") or {}
            cid = call.get("id") or f"call_{n}"
            messages.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": cid, "type": "function", "function": {"name": name, "arguments": raw if isinstance(raw, str)
                                                             else json.dumps(raw)}}]})
            spec = specs.get(name)
            ok, text = run_tool(spec, raw) if spec else (False, f"ERROR: no tool named '{name}'")
            out.calls.append({"tool": name, "args": raw if isinstance(raw, dict) else _loads(raw), "ok": ok,
                              "result": text[:300]})
            messages.append({"role": "tool", "tool_call_id": cid, "content": text})
            if ok and name == TERMINAL:
                out.finished = True
                break
    except Exception as e:                      # no model server: no extra checks, the decision is unaffected
        out.calls.append({"tool": "-", "args": {}, "ok": False, "result": f"model unavailable ({type(e).__name__})"})
    return out
