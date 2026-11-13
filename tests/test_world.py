"""Incidents where a SQL fix is not always the answer: generated faults, bad source data, mistaken complaints."""
import json

import pytest

from triage.warehouse import world as world_eval
from triage.warehouse import as_dag, fix, mutants
from triage.warehouse.breaks import BREAKS
from triage.warehouse.build import Warehouse
from triage.warehouse.pipeline import BY_ID, REPORTS
from triage.warehouse.probe import observe_sources
from triage.warehouse.sourcefaults import SOURCE_FAULTS


class Scripted:
    model = "scripted"

    def __init__(self, calls, then="I cannot tell."):
        self.calls, self.then, self.tools = list(calls), then, []

    def chat(self, messages, tools):
        self.tools = [t["function"]["name"] for t in tools]
        if not self.calls:
            return {"content": self.then}
        name, args = self.calls.pop(0)
        return {"tool_calls": [{"id": f"c{len(self.calls)}", "function": {"name": name, "arguments": json.dumps(args)}}]}


@pytest.fixture(scope="module")
def wh():
    w = Warehouse().build()
    yield w
    w.close()


@pytest.fixture(scope="module")
def all_cases(wh):
    return world_eval.cases(wh)


def play(wh, case, calls, **kw):
    inc = world_eval.stage(wh, case)
    model = Scripted(calls)
    try:
        inv, sql, session = fix.draft_checked(wh, as_dag(), inc, model, open_world=True, **kw)
        kept = sql is not None and fix.apply_and_verify(wh, inv.fix_target, sql, inc["symptom_node"])["kept"]
        return world_eval.result(case, sql, kept, session.closed, inv.answer), session, model, inv
    finally:
        world_eval.restore(wh)


FIRST = [("where_checks_fail", {}), ("check_sources", {})]


# ------------------------------------------------------------------ generated faults

def test_generated_faults_are_real_incidents_and_the_same_every_time(wh, all_cases):
    gen = [c for c in all_cases if c["kind"] == "sql"]
    assert len(gen) >= 25 and len({c["id"] for c in gen}) == len(gen)
    assert [c["id"] for c in gen] == [m["id"] for m in mutants.generate(wh)]
    hand = {b.broken_sql() for b in BREAKS}
    for c in gen:
        assert c["sql"] != BY_ID[c["node"]].sql and c["sql"] not in hand
    assert {c["split"] for c in gen} == {"evolve", "held_in", "ood"}
    assert all(c["split"] == "ood" for c in gen if c["id"].startswith(("gen_null_default", "gen_aggregate")))


def test_every_generated_fault_changes_a_report_and_reverts_cleanly(wh, all_cases):
    for c in [c for c in all_cases if c["kind"] == "sql"][::4]:
        inc = world_eval.stage(wh, c)
        assert inc["symptom_node"] in REPORTS and inc["symptom_node"] in wh.affected()
        world_eval.restore(wh)
    assert not wh.affected() and not wh.changed_procedures()


# ------------------------------------------------------------------ bad source data

def test_a_source_fault_breaks_a_report_with_every_procedure_untouched(wh):
    for f in SOURCE_FAULTS:
        wh.corrupt_source(f.table, f.statements)
        wh.run_cycle()
        assert any(r in wh.affected() for r in REPORTS), f.id
        assert not wh.changed_procedures() and wh.corrupted_sources() == [f.table]
        assert (f.table in observe_sources(wh)) == f.visible, f.id
        assert wh.reload_sources() == [f.table]
        wh.run_cycle()
        assert not wh.affected() and not observe_sources(wh)


def test_naming_the_bad_source_is_right_and_changes_no_sql(wh, all_cases):
    case = next(c for c in all_cases if c["id"] == "source.orders_loaded_twice")
    got, session, _, _ = play(wh, case, FIRST + [("close_without_fix", {"kind": "source_data", "source_table": "raw.orders",
                                                                         "reason": "orders has duplicate keys"})])
    assert got == "named the bad source" and session.closed["source_table"] == "orders"
    wrong, _, _, _ = play(wh, case, FIRST + [("close_without_fix", {"kind": "source_data", "source_table": "customer", "reason": "x"})])
    assert wrong == "blamed the wrong source"


def test_a_sql_workaround_for_bad_data_is_counted_as_a_workaround_not_a_fix(wh, all_cases):
    case = next(c for c in all_cases if c["id"] == "source.orders_loaded_twice")
    dedup = BY_ID["stg_orders"].sql.replace("select ", "select distinct ", 1)
    got, session, _, _ = play(wh, case, FIRST + [("check_node", {"node": "stg_orders"}),
                                                 ("try_fix", {"fix_target": "stg_orders", "sql": dedup, "reason": "dedupe"}),
                                                 ("submit_diagnosis", {"fix_target": "stg_orders", "sql": dedup, "reason": "dedupe"})])
    assert session.verified and got == "worked around bad data in SQL" and got != world_eval.RIGHT["source"]


def test_an_incident_cannot_be_closed_before_the_checks_are_read(wh, all_cases):
    case = next(c for c in all_cases if c["kind"] == "none")
    close = ("close_without_fix", {"kind": "nothing_wrong", "reason": "looks fine"})
    got, session, _, _ = play(wh, case, [close])
    assert got == "handed to a person" and session.closed is None and len(session.refusals) == 1
    got, session, _, _ = play(wh, case, FIRST + [close])
    assert got == "said nothing was wrong" and got == world_eval.RIGHT["none"]


def test_closing_a_real_sql_fault_as_nothing_wrong_is_scored_as_wrong(wh, all_cases):
    case = next(c for c in all_cases if c["kind"] == "sql" and c["silent"])
    got, _, _, _ = play(wh, case, FIRST + [("close_without_fix", {"kind": "nothing_wrong", "reason": "all checks pass"})])
    assert got == "said nothing was wrong" and got != world_eval.RIGHT["sql"]
    node, good = case["node"], BY_ID[case["node"]].sql
    got, _, _, _ = play(wh, case, FIRST + [("check_node", {"node": node}), ("try_fix", {"fix_target": node, "sql": good, "reason": "r"}),
                                           ("submit_diagnosis", {"fix_target": node, "sql": good, "reason": "r"})])
    assert got == "fixed"


def test_a_fix_cannot_read_the_clean_copy_of_a_source(wh):
    with pytest.raises(ValueError, match="cannot read"):
        fix.validate(wh, as_dag(), "stg_orders", "select * from clean.orders")


# ------------------------------------------------------------------ the deploy history

def test_a_deploy_is_in_the_history_and_a_rehearsal_is_not(wh, all_cases):
    case = next(c for c in all_cases if c["kind"] == "sql")
    world_eval.stage(wh, case)
    try:
        assert [d["node"] for d in wh.deploys_since_good()] == [case["node"]]
        fix.rehearse(wh, case["node"], BY_ID[case["node"]].sql, REPORTS[0])
        assert [d["node"] for d in wh.deploys_since_good()] == [case["node"]]
    finally:
        world_eval.restore(wh)
    wh.mark_good()
    assert wh.deploys_since_good() == []


def test_the_history_tool_is_its_own_arm_and_shows_the_change_as_a_diff(wh, all_cases):
    case = next(c for c in all_cases if c["id"].startswith("gen_join_kind"))
    _, session, model, inv = play(wh, case, [("recent_changes", {})], history=True)
    assert "recent_changes" in model.tools and session.asked_changes
    shown = inv.steps[0]["result"]
    assert case["node"] in shown and "-" in shown and "+" in shown and "left join" in shown
    _, _, model, inv = play(wh, case, [("recent_changes", {})])
    assert "recent_changes" not in model.tools and not inv.steps[0]["ok"]


def test_with_bad_data_the_history_is_empty(wh, all_cases):
    case = next(c for c in all_cases if c["kind"] == "source")
    _, _, _, inv = play(wh, case, [("recent_changes", {})], history=True)
    assert inv.steps[0]["result"].startswith("no procedure has been deployed")


def test_nights_are_the_same_for_the_same_seed_and_not_all_alike(all_cases):
    a, b = world_eval.schedule(all_cases, 40, seed=3), world_eval.schedule(all_cases, 40, seed=3)
    assert a == b and a != world_eval.schedule(all_cases, 40, seed=4)
    kinds = {c["kind"] if c else "quiet" for c in a}
    assert kinds == {"quiet", "sql", "source", "none"}


def test_reloading_a_source_that_was_fine_changes_nothing(wh):
    f = SOURCE_FAULTS[0]
    wh.corrupt_source(f.table, f.statements)
    assert wh.reload_sources(only="customer") == [] and wh.corrupted_sources() == [f.table]
    assert wh.reload_sources(only=f.table) == [f.table]
    wh.run_cycle()
    assert not wh.affected()
