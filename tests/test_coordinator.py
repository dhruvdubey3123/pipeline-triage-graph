"""The coordinator: the LLM chooses what else to check, and cannot move a decision.

The test that matters is the first one. It runs every labelled incident, clean and noisy, with a
coordinator that behaves erratically (random tools, nodes that do not exist, early stops, no stop) and
asserts each decision is exactly what the rules made without it.
"""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from triage.communicator import brief, check, facts
from triage.dag import PipelineDAG
from triage.knowledge import CaseLibrary
from triage.service import TriageService

ROOT = Path(__file__).resolve().parent.parent
DAG = PipelineDAG.from_yaml(ROOT / "examples/pipeline.yaml")
BASE = CaseLibrary.load(ROOT / "examples/case_library.yaml")
INCIDENT = json.loads((ROOT / "examples/incident_schema_drift.json").read_text())
DECISION = ("fix_target", "upstream_cause", "ambiguous", "category", "proposal", "plan", "confidence",
            "review_issues", "route")


def lib():
    return CaseLibrary(copy.deepcopy(BASE.cases))


class Erratic:
    """Picks its calls from a hash of the conversation, so each incident gets a different, repeatable mess."""

    model = "erratic"

    def chat(self, messages, tools):
        if not tools:
            return {"content": "brief"}
        seed = int(hashlib.sha256(json.dumps(messages).encode()).hexdigest(), 16)
        names = [t["function"]["name"] for t in tools]
        nodes = list(DAG.nodes) + ["not_a_node"]
        name = names[seed % len(names)]
        args = {} if name == "finish" else ({"query": "invalid identifier"} if name == "precedents"
                                            else {"node": nodes[(seed // 7) % len(nodes)]})
        return {"tool_calls": [{"id": f"x{seed % 997}", "type": "function",
                                "function": {"name": name, "arguments": json.dumps(args)}}]}


class Script:
    model = "script"

    def __init__(self, *calls, briefing=""):
        self.calls, self.briefing, self.seen = list(calls), briefing, []

    def chat(self, messages, tools):
        self.seen.append(messages)
        if not tools:
            return {"content": self.briefing}
        n = sum(1 for m in messages if m["role"] == "tool")
        if n < len(self.calls):
            name, args = self.calls[n]
            return {"tool_calls": [{"id": f"c{n}", "type": "function",
                                    "function": {"name": name, "arguments": json.dumps(args)}}]}
        return {"content": "done"}


class Down:
    def chat(self, messages, tools):
        raise ConnectionError("no server")


@pytest.mark.parametrize("data", ["incidents.jsonl", "incidents_noisy.jsonl"])
def test_an_erratic_coordinator_changes_no_decision_on_any_labelled_incident(data):
    rows = [json.loads(line) for line in (ROOT / "evals/data" / data).read_text().splitlines() if line.strip()]
    for row in rows:
        row.pop("label")
        plain = TriageService(DAG, lib()).open(copy.deepcopy(row))[1]
        steered = TriageService(DAG, lib(), coordinator=Erratic()).open(copy.deepcopy(row))[1]
        for k in DECISION:
            assert steered.get(k) == plain.get(k), (row["id"], k)
        assert steered["context_checks"]["calls"], row["id"]


def test_what_it_finds_reaches_the_briefing():
    model = Script(("downstream", {"node": "sp_fill_rate"}), ("probe", {"node": "load_timesheets"}), ("finish", {}),
                   briefing="sp_fill_rate failed after load_timesheets renamed a column; fix it and refresh fct_workforce "
                            "and rpt_workforce, which read from it.")
    v = TriageService(DAG, lib(), coordinator=model, communicator=model).open(INCIDENT)[1]
    tools_used = [c["tool"] for c in v["context_checks"]["calls"]]
    assert tools_used == ["downstream", "probe", "finish"]
    briefing_prompt = model.seen[-1][1]["content"]
    assert "downstream(sp_fill_rate)" in briefing_prompt and "probe(load_timesheets)" in briefing_prompt
    assert v["briefing_source"] == "model"


def test_context_cannot_turn_into_an_extra_step():
    state = TriageService(DAG, lib()).open(INCIDENT)[1] | {"incident": INCIDENT}
    f = facts(state, DAG)
    ok = "sp_fill_rate failed because load_timesheets renamed a column; the plan patches sp_fill_rate."
    assert check(ok, f, DAG) == []
    assert any("suggests acting on load_timesheets" in p
               for p in check(ok + " Also rerun load_timesheets to be safe.", f, DAG))


def test_no_model_server_means_no_extra_checks_and_the_same_decision():
    plain = TriageService(DAG, lib()).open(INCIDENT)[1]
    v = TriageService(DAG, lib(), coordinator=Down()).open(INCIDENT)[1]
    assert all(v[k] == plain[k] for k in DECISION)
    assert "model unavailable" in v["context_checks"]["calls"][0]["result"]


def test_it_cannot_reach_a_tool_that_writes():
    model = Script(("rerun_job", {"node": "sp_fill_rate"}), ("finish", {}))
    v = TriageService(DAG, lib(), coordinator=model).open(INCIDENT)[1]
    first = v["context_checks"]["calls"][0]
    assert not first["ok"] and "no tool named 'rerun_job'" in first["result"]


def test_the_eval_reports_what_it_chose():
    from evals.run_eval import evaluate
    res = evaluate(ROOT / "evals/data/incidents.jsonl", ROOT / "examples/pipeline.yaml",
                   ROOT / "examples/case_library.yaml", coordinator=Erratic(), limit=5)
    assert res["coordinator"]["mean extra checks per incident"] >= 0 and res["coordinator"]["checks it chose"]
    assert res["root_accuracy"] == 1.0
