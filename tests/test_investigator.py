"""The LLM investigator, driven by a scripted chat model so CI needs no model and no network.

What is worth asserting is the shape around the model rather than the model: it can only read, a
malformed answer goes back to it as an error, the first valid answer stands, and when it and the
tracer disagree on the root cause the incident goes to a person instead of to the approval gate.
How good a real model is lives in evals/run_eval.py --investigator, not here.
"""
import json
from pathlib import Path

from triage.dag import PipelineDAG
from triage.investigator import TERMINAL, investigate, tools
from triage.knowledge import CaseLibrary
from triage.service import TriageService

ROOT = Path(__file__).resolve().parent.parent
DAG = PipelineDAG.from_yaml(ROOT / "examples/pipeline.yaml")
INCIDENT = json.loads((ROOT / "examples/incident_schema_drift.json").read_text())   # sp_fill_rate, cause load_timesheets


def lib():
    return CaseLibrary.load(ROOT / "examples/case_library.yaml")


class Scripted:
    """Answers by how many tool results it has seen, so the same conversation gets the same reply."""

    model = "scripted"

    def __init__(self, *calls, final="done"):
        self.calls, self.final, self.seen_tools = list(calls), final, []

    def chat(self, messages, tools):
        self.seen_tools = [t["function"]["name"] for t in tools]
        n = sum(1 for m in messages if m["role"] == "tool")
        if n < len(self.calls):
            name, args = self.calls[n]
            return {"tool_calls": [{"id": f"c{n}", "type": "function",
                                    "function": {"name": name, "arguments": json.dumps(args)}}]}
        return {"content": f"<think>hm</think>{self.final}"}


RIGHT = {"category": "schema_drift", "fix_target": "sp_fill_rate", "cause": "load_timesheets",
         "reason": "load_timesheets changed a column that sp_fill_rate reads; patch the procedure."}
WRONG = {**RIGHT, "fix_target": "load_timesheets", "cause": ""}


def svc(model):
    return TriageService(DAG, lib(), investigator=model)


def agents(view):
    return [line["agent"] for line in view["log"]]


def test_it_can_only_read():
    names = {t.name for t in tools(DAG, lib(), INCIDENT)}
    assert names == {"pipeline", "status", "upstream", "precedents", TERMINAL}


def test_a_right_diagnosis_drives_the_plan_and_waits_for_a_person():
    _, v = svc(Scripted(("pipeline", {}), ("upstream", {"node": "rpt_workforce"}), (TERMINAL, RIGHT))).open(INCIDENT)
    assert v["waiting_for_human"] and v["fix_target"] == "sp_fill_rate"
    assert v["plan"][0] == "fix sp_fill_rate"
    assert "investigator" in agents(v)
    assert any("agrees with the investigator" in line["msg"] for line in v["log"])


def test_a_malformed_answer_goes_back_and_the_first_valid_one_stands():
    """Qwen3 8B, told to submit once, submitted the right node and later talked itself into a wrong one."""
    inv = investigate(INCIDENT, DAG, lib(), Scripted(
        (TERMINAL, {**RIGHT, "category": "schema-drift"}),          # not a category
        (TERMINAL, {**RIGHT, "fix_target": "rpt_workforce"}),       # protected, if the pipeline says so
        (TERMINAL, RIGHT),
        (TERMINAL, WRONG)))                                         # never reached
    assert inv.submitted and inv.fix_target == "sp_fill_rate"
    assert "is not a category" in inv.steps[0]["result"]
    assert inv.steps[-1]["tool"] == TERMINAL and inv.steps[-1]["ok"]
    assert all(s["args"].get("fix_target") != "load_timesheets" for s in inv.steps)


def test_when_it_and_the_tracer_disagree_a_person_decides_from_both():
    _, v = svc(Scripted((TERMINAL, WRONG))).open(INCIDENT)
    assert not v["waiting_for_human"] and v["outcome"] == "escalated"
    assert v["tracer_target"] == "sp_fill_rate" and v["fix_target"] == "load_timesheets"
    assert any("disagree on the root cause" in i for i in v["review_issues"])


def test_no_answer_is_escalated_not_guessed():
    _, v = svc(Scripted(("pipeline", {}), final="I am not sure.")).open(INCIDENT)
    assert not v["waiting_for_human"]
    assert "the investigator submitted no diagnosis" in v["review_issues"]


def test_an_unknown_tool_or_node_is_an_error_it_can_read():
    inv = investigate(INCIDENT, DAG, lib(), Scripted(
        ("rerun_job", {"node": "sp_fill_rate"}), ("status", {"node": "nope"}), (TERMINAL, RIGHT)))
    assert "no tool named 'rerun_job'" in inv.steps[0]["result"]
    assert "not in this pipeline" in inv.steps[1]["result"]
    assert inv.submitted


def test_the_category_stays_with_the_rules_and_the_models_is_a_second_opinion():
    _, v = svc(Scripted((TERMINAL, {**RIGHT, "category": "missed_load"}))).open(INCIDENT)
    assert v["category"] == "schema_drift"
    assert any("investigator said missed_load" in sig for sig in v["signals"])


def test_a_failure_nobody_recognises_reaches_a_person_even_when_the_model_names_one():
    """Regression from a real Qwen3 8B run: on incidents the rules call 'unknown', the model picked a
    familiar category and 4 of 7 escalations reached the approval gate instead of a person."""
    rows = [json.loads(line) for line in (ROOT / "evals/data/incidents.jsonl").read_text().splitlines()]
    inc = next(r for r in rows if r["id"] == "SYN-043")
    assert inc.pop("label")["category"] == "unknown"
    _, rules_view = TriageService(DAG, lib()).open(inc)
    model = Scripted((TERMINAL, {"category": "schema_drift", "fix_target": rules_view["fix_target"],
                                 "reason": "looks like schema drift"}))
    _, v = svc(model).open(inc)
    assert v["fix_target"] == rules_view["fix_target"]             # it found the right place
    assert not v["waiting_for_human"]                                # and still goes to a person
    assert "unrecognised failure category" in v["review_issues"]


def test_without_a_model_the_graph_is_the_rule_based_one():
    _, v = TriageService(DAG, lib()).open(INCIDENT)
    assert "investigator" not in agents(v) and "investigation" not in v
    assert v["fix_target"] == "sp_fill_rate"


def test_the_eval_scores_the_investigator_next_to_the_tracer():
    from evals.run_eval import evaluate
    res = evaluate(ROOT / "evals/data/incidents.jsonl", ROOT / "examples/pipeline.yaml",
                   ROOT / "examples/case_library.yaml", investigator=Scripted((TERMINAL, RIGHT)), limit=5)
    r = res["investigator"]
    assert r["incidents"] == "1-5" and r["submitted a diagnosis"] == "5/5"
    assert r["tracer root cause correct, same incidents"] == "5/5"
    assert "cross-checked" in res["mode"]


def test_the_http_client_speaks_the_openai_tool_calling_shape(monkeypatch):
    import httpx

    from triage.investigator import OpenAICompatChat
    seen = {}

    class R:
        status_code, headers = 200, {}

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def post(url, json=None, timeout=None, headers=None):
        seen.update(url=url, body=json)
        return R()

    monkeypatch.setattr(httpx, "post", post)
    msg = OpenAICompatChat(model="qwen3:8b").chat([{"role": "user", "content": "hi"}],
                                                  [t.schema() for t in tools(DAG, lib(), INCIDENT)])
    assert msg == {"content": "ok"}
    assert seen["url"] == "http://localhost:11434/v1/chat/completions"
    assert seen["body"]["model"] == "qwen3:8b" and seen["body"]["temperature"] == 0.0
    assert {t["function"]["name"] for t in seen["body"]["tools"]} >= {TERMINAL, "upstream"}


def test_a_model_that_times_out_sends_the_incident_to_a_person():
    """A real run on a laptop hit a five-minute timeout and crashed the whole graph. A model that cannot
    answer has not diagnosed anything: that is a reason to escalate, not to fail."""
    import httpx

    class TimesOut:
        model = "slow"

        def chat(self, messages, tools):
            raise httpx.ReadTimeout("timed out")

    _, v = svc(TimesOut()).open(INCIDENT)
    assert not v["waiting_for_human"] and v["outcome"] == "escalated"
    assert "the investigator submitted no diagnosis" in v["review_issues"]
    assert "model unavailable (ReadTimeout)" in v["investigation"]["steps"][0]["result"]


def test_one_broken_incident_does_not_lose_the_eval(monkeypatch):
    from evals.run_eval import evaluate
    real_open = TriageService.open

    def flaky(self, incident):
        if incident["id"] == "SYN-002":
            raise RuntimeError("boom")
        return real_open(self, incident)

    monkeypatch.setattr(TriageService, "open", flaky)
    res = evaluate(ROOT / "evals/data/incidents.jsonl", ROOT / "examples/pipeline.yaml",
                   ROOT / "examples/case_library.yaml", limit=4)
    assert res["incidents"] == 4 and res["root_accuracy"] == 0.75
    assert res["errored"] == [{"id": "SYN-002", "error": "RuntimeError: boom"}]


def test_the_request_timeout_is_configurable(monkeypatch):
    from triage.investigator import OpenAICompatChat
    monkeypatch.setenv("TRIAGE_TIMEOUT", "900")
    assert OpenAICompatChat().timeout == 900.0


def test_a_rate_limited_free_tier_is_waited_out_not_counted_as_no_answer(monkeypatch):
    import httpx

    import triage.investigator as inv
    from triage.investigator import OpenAICompatChat
    replies = [429, 503, 200]

    class R:
        def __init__(self, code):
            self.status_code, self.headers = code, {"retry-after": "0"}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError("err", request=None, response=None)

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(httpx, "post", lambda *a, **k: R(replies.pop(0)))
    monkeypatch.setattr(inv.time, "sleep", lambda s: None)
    assert OpenAICompatChat().chat([{"role": "user", "content": "hi"}], []) == {"content": "ok"}
