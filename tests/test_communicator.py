"""The communicator, driven by a scripted model: what it may say, and what happens when it says more.

The traversal is deterministic; the model only writes the briefing. Every check here is a way a fluent
briefing could still mislead the person deciding: pointing at a job the trace never looked at, losing
the root cause in a paraphrase, or renaming the failure.
"""
import json
from pathlib import Path

from triage.communicator import brief, check, facts, template
from triage.dag import PipelineDAG
from triage.knowledge import CaseLibrary
from triage.service import TriageService

ROOT = Path(__file__).resolve().parent.parent
DAG = PipelineDAG.from_yaml(ROOT / "examples/pipeline.yaml")
INCIDENT = json.loads((ROOT / "examples/incident_schema_drift.json").read_text())   # sp_fill_rate, schema_drift

GOOD = ("The fill-rate report looked wrong because sp_fill_rate failed: load_timesheets renamed REQ_ID, so the "
        "procedure's reference broke. The plan patches sp_fill_rate, then refreshes fct_workforce and rpt_workforce. "
        "The reviewer flagged nothing; it is waiting for your approval.")


class Says:
    model = "scripted"

    def __init__(self, text):
        self.text, self.seen = text, None

    def chat(self, messages, tools):
        self.seen = (messages, tools)
        return {"content": f"<think>let me summarise</think>{self.text}"}


class Down:
    def chat(self, messages, tools):
        raise ConnectionError("no server")


def lib():
    return CaseLibrary.load(ROOT / "examples/case_library.yaml")


def run(text):
    return TriageService(DAG, lib(), communicator=Says(text)).open(INCIDENT)[1]


def state():
    return TriageService(DAG, lib()).open(INCIDENT)[1] | {"incident": INCIDENT}


def test_a_faithful_briefing_is_what_the_person_reads():
    v = run(GOOD)
    assert v["briefing_source"] == "model" and v["briefing"] == GOOD and v["briefing_problems"] == []
    assert v["waiting_for_human"] and any(line["agent"] == "communicator" for line in v["log"])


def test_the_traversal_is_unchanged_by_the_communicator():
    with_it, without = run(GOOD), TriageService(DAG, lib()).open(INCIDENT)[1]
    for k in ("fix_target", "upstream_cause", "category", "plan", "review_issues", "route"):
        assert with_it[k] == without[k]


def test_naming_a_job_the_trace_never_touched_falls_back_to_the_template():
    v = run(GOOD + " You should also rerun sp_vendor_rollup to be safe.")
    assert v["briefing_source"] == "template"
    assert any("sp_vendor_rollup" in p for p in v["briefing_problems"])
    assert "sp_vendor_rollup" not in v["briefing"]


def test_losing_the_root_cause_in_a_paraphrase_falls_back():
    v = run("A procedure upstream of the report failed after a column rename; the plan fixes it and refreshes the tables.")
    assert v["briefing_source"] == "template"
    assert any("does not name the root cause" in p for p in v["briefing_problems"])
    assert "sp_fill_rate" in v["briefing"]


def test_renaming_the_failure_falls_back():
    v = run(GOOD.replace("renamed REQ_ID", "loaded duplicate rows, a duplicate merge"))
    assert v["briefing_source"] == "template"
    assert any("calls it duplicate_merge" in p for p in v["briefing_problems"])


def test_no_model_server_still_gives_the_person_a_briefing():
    b = brief(state(), DAG, Down())
    assert b.source == "template" and "sp_fill_rate" in b.text and "model unavailable" in b.problems[0]


def test_the_model_is_given_the_findings_and_no_tools():
    model = Says(GOOD)
    brief(state(), DAG, model)
    messages, tools = model.seen
    assert tools == [] and "root cause: sp_fill_rate" in messages[1]["content"]
    assert "_allowed_nodes" not in messages[1]["content"]


def test_the_template_is_itself_faithful():
    f = facts(state(), DAG)
    assert check(template(f), f, DAG) == []


def test_an_escalated_incident_is_briefed_for_the_owner():
    status = {"rpt_workforce": {"status": "stale"}, "fct_workforce": {"status": "stale"}, "fct_spend": {"status": "stale"},
              "sp_fill_rate": {"status": "failed", "error": "invalid identifier"},
              "sp_vendor_rollup": {"status": "failed", "error": "Duplicate row detected"}}
    inc = {"id": "AMB-1", "symptom_node": "rpt_workforce", "symptom": "report wrong", "node_status": status}
    v = TriageService(DAG, lib(), communicator=Says("x")).open(inc)[1]
    assert not v["waiting_for_human"] and "escalated to a person" in v["briefing"].lower()


def test_the_eval_reports_how_often_the_model_briefing_was_used():
    from evals.run_eval import evaluate
    res = evaluate(ROOT / "evals/data/incidents.jsonl", ROOT / "examples/pipeline.yaml",
                   ROOT / "examples/case_library.yaml", communicator=Says("Nothing to see."), limit=4)
    c = res["communicator"]
    assert c["briefings written by the model that passed the check"].endswith("/4")
    assert c["fell back to the template"] >= 1 and c["why it fell back"]
