import json
from pathlib import Path

import pytest

from triage.dag import PipelineDAG, refresh_order, trace
from triage.knowledge import CaseLibrary, classify
from triage.service import TriageService

ROOT = Path(__file__).resolve().parent.parent
DAG = PipelineDAG.from_yaml(ROOT / "examples/pipeline.yaml")
INCIDENT = json.loads((ROOT / "examples/incident_schema_drift.json").read_text())


def lib():
    return CaseLibrary.load(ROOT / "examples/case_library.yaml")


def test_tracer_finds_failed_procedure_and_upstream_change():
    res = trace(DAG, INCIDENT["node_status"], "rpt_workforce")
    assert res.fix_target == "sp_fill_rate"
    assert res.upstream_cause == "load_timesheets"
    assert not res.ambiguous
    ruled_out = {h.node for h in res.hops if h.ruled_out}
    assert "sp_merge_cases" in ruled_out  # healthy branch checked but not expanded


def test_tracer_flags_two_independent_failures_as_ambiguous():
    status = {
        "rpt_workforce": {"status": "stale"}, "fct_workforce": {"status": "stale"}, "fct_spend": {"status": "stale"},
        "sp_fill_rate": {"status": "failed", "error": "invalid identifier"},
        "sp_vendor_rollup": {"status": "failed", "error": "Duplicate row detected"},
    }
    res = trace(DAG, status, "rpt_workforce")
    assert res.ambiguous
    assert set(res.candidates) == {"sp_fill_rate", "sp_vendor_rollup"}


def test_stale_red_herring_loses_to_real_failure():
    status = {
        "rpt_workforce": {"status": "stale"}, "fct_workforce": {"status": "stale"},
        "load_cases": {"status": "missing", "error": "Job timed out"},
        "sp_merge_cases": {"status": "stale"}, "sp_fill_rate": {"status": "stale"},
        "fct_spend": {"status": "stale", "note": "refresh ran late"},
    }
    res = trace(DAG, status, "rpt_workforce")
    assert res.fix_target == "load_cases" and not res.ambiguous


def test_refresh_order_is_topological():
    order = refresh_order(DAG, "load_cases")
    assert order.index("load_cases") < order.index("sp_fill_rate") < order.index("fct_workforce") < order.index("rpt_workforce")


def test_cycle_rejected():
    with pytest.raises(ValueError):
        PipelineDAG.from_dict({"nodes": [{"id": "a", "kind": "x"}, {"id": "b", "kind": "x"}], "edges": [["a", "b"], ["b", "a"]]})


@pytest.mark.parametrize("text,cat", [
    ("SQL compilation error: invalid identifier 'X'", "schema_drift"),
    ("Duplicate row detected during DML action", "duplicate_merge"),
    ("Job timed out after 3600s", "missed_load"),
    ("Insufficient privileges to operate on schema", "permission"),
    ("Process exited with code 137", "unknown"),
])
def test_classifier_rules(text, cat):
    assert classify(text, None)[0] == cat


def test_precedent_search_ranks_matching_case_first():
    hits = lib().search("procedure failed invalid identifier after column renamed")
    assert hits and hits[0][0].category == "schema_drift"


def test_graph_pauses_at_gate_then_applies_on_approval():
    svc = TriageService(DAG, lib())
    run_id, v = svc.open(dict(INCIDENT))
    assert v["waiting_for_human"] is True
    assert v["fix_target"] == "sp_fill_rate" and v["category"] == "schema_drift"
    assert v["plan"][0] == "fix sp_fill_rate"
    assert "applied" not in v  # nothing changes before a person decides
    done = svc.decide(run_id, approve=True, reviewer="test")
    assert done["outcome"] == "resolved"
    assert all(step.startswith("[dry-run]") for step in done["applied"])
    assert done["learned_case"] == "CASE-0005"


def test_reject_escalates_without_applying():
    svc = TriageService(DAG, lib())
    run_id, _ = svc.open(dict(INCIDENT, id="INC-reject"))
    done = svc.decide(run_id, approve=False)
    assert done["outcome"] == "escalated" and "applied" not in done


def test_reviewer_blocks_unknown_category():
    inc = dict(INCIDENT, id="INC-unknown")
    inc["node_status"] = dict(INCIDENT["node_status"], sp_fill_rate={"status": "failed", "error": "Process exited with code 137"},
                              load_timesheets={"status": "ok"})
    svc = TriageService(DAG, lib())
    _, v = svc.open(inc)
    assert v["waiting_for_human"] is False and v["outcome"] == "escalated"
    assert any("unrecognised" in i for i in v["review_issues"])


def test_decide_twice_is_rejected():
    svc = TriageService(DAG, lib())
    run_id, _ = svc.open(dict(INCIDENT, id="INC-twice"))
    svc.decide(run_id, approve=True)
    with pytest.raises(ValueError):
        svc.decide(run_id, approve=True)


def test_api_roundtrip():
    from fastapi.testclient import TestClient
    from triage.api import app

    c = TestClient(app)
    assert c.get("/healthz").json()["ok"]
    r = c.post("/incidents", json=dict(INCIDENT, id="INC-api"))
    assert r.status_code == 200 and r.json()["waiting_for_human"]
    r = c.post("/incidents/INC-api/decision", json={"approve": True, "reviewer": "api-test"})
    assert r.json()["outcome"] == "resolved"
    assert c.post("/incidents", json={"symptom_node": "nope"}).status_code == 422


def test_mcp_server_exposes_tools():
    import asyncio
    from triage.mcp_server import server

    names = {t.name for t in asyncio.run(server.list_tools())}
    assert {"get_upstream", "get_downstream", "trace_failure", "search_precedents", "list_nodes"} <= names
