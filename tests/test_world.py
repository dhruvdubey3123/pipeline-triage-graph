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


def test_a_fix_cannot_read_anything_that_holds_the_answer(wh):
    for sql in ("select * from clean.orders", 'select * from "reference"."rpt_regional_summary"', "select * from main.golden",
                "select sql from procedures", "select * from deploys", "select * from good_run", "select * from run_log",
                "select * from source_log", "select * from information_schema.tables", "select * from duckdb_tables()"):
        with pytest.raises(ValueError, match="cannot read"):
            fix.validate(wh, as_dag(), "stg_orders", sql)
    assert fix.validate(wh, as_dag(), "stg_orders", BY_ID["stg_orders"].sql)


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


# ------------------------------------------------------------------ what a first run with a model exposed

def test_the_model_is_never_shown_the_case_id_or_the_label(wh, all_cases):
    """The first sample was run with ids like source.cost_in_cents in the task text. The model read
    the answer off them. Nothing the model receives may contain a case id, its kind or its label."""
    from evals.warehouse_eval import each_break

    class Listen:
        model, said = "listen", ""

        def chat(self, messages, tools):
            Listen.said += json.dumps(messages) + json.dumps(tools)
            return {"content": "UNSURE"}

    class Pokes:
        """Calls every tool that reads, on every node, so that what the tools return is checked too."""
        model = "pokes"

        def __init__(self, report):
            from triage.warehouse.pipeline import NODES
            self.calls = ([("where_checks_fail", {}), ("check_sources", {}), ("pipeline", {})]
                          + [(tool, {"node": x.id}) for x in NODES for tool in ("status", "upstream", "profile", "check_node")])

        def chat(self, messages, tools):
            Listen.said += json.dumps(messages[-1])
            if not self.calls:
                return {"content": "UNSURE"}
            name, args = self.calls.pop(0)
            return {"tool_calls": [{"id": "c", "function": {"name": name, "arguments": json.dumps(args)}}]}

    for n, c in enumerate(all_cases[::3], 1):
        seen = world_eval.as_seen(world_eval.stage(wh, c), n)
        Listen.said = ""
        fix.draft_checked(wh, as_dag(), seen, Pokes(seen["symptom_node"]), open_world=True, max_steps=80)
        fix.draft_checked(wh, as_dag(), seen, Listen(), open_world=True, history=True)
        world_eval.restore(wh)
        assert c["id"] not in Listen.said and c["id"].split(".")[1] not in Listen.said and (c["kind"] == "none" or c["what"] not in Listen.said), c["id"]
        assert "gen_" not in Listen.said and "mistaken_" not in Listen.said
        assert set(seen) == {"id", "symptom_node", "symptom", "node_status"} and seen["id"] == f"INC-{n:03d}"
    for n, inc in enumerate(each_break(wh), 1):
        seen, Listen.said = world_eval.as_seen(inc, n), ""
        fix.draft_checked(wh, as_dag(), seen, Listen())
        fix.draft(wh, as_dag(), seen, Listen())
        assert inc["id"] not in Listen.said and inc["label"]["family"] not in Listen.said and "label" not in seen, inc["id"]


def test_sql_that_changes_no_table_is_not_accepted_as_a_fix(wh, all_cases):
    """What the model did on a mistaken complaint: rewrote a correct formula into a different one that
    happens to give the same numbers on this data. It passed, because the warehouse was right before."""
    case = next(c for c in all_cases if c["id"] == "none.mistaken_2")
    node = "rpt_regional_summary"
    same = BY_ID[node].sql.replace("(select sum(margin) / sum(revenue) from mart_supplier_performance)",
                                   "(select sum(margin) / (select sum(revenue) from mart_customer_revenue) from mart_supplier_performance)")
    assert same != BY_ID[node].sql
    got, session, _, inv = play(wh, case, FIRST + [("check_node", {"node": node}), ("try_fix", {"fix_target": node, "sql": same, "reason": "r"}),
                                                   ("submit_diagnosis", {"fix_target": node, "sql": same, "reason": "r"})])
    assert inv.steps[3]["result"].startswith("not a fix") and session.no_ops == 1 and not session.verified
    assert got == "handed to a person" and not inv.steps[4]["ok"]


def test_an_answer_in_text_gets_one_reminder_to_use_a_tool(wh, all_cases):
    case = next(c for c in all_cases if c["kind"] == "none")
    inc = world_eval.stage(wh, case)

    class Talks:
        model, turns = "talks", 0

        def chat(self, messages, tools):
            Talks.turns += 1
            if Talks.turns <= 2:
                return {"tool_calls": [{"id": f"c{Talks.turns}", "function": {"name": ("where_checks_fail", "check_sources")[Talks.turns - 1], "arguments": "{}"}}]}
            if Talks.turns == 3:
                return {"content": 'I would close it: {"kind": "nothing_wrong"}'}
            assert "text ends nothing" in messages[-1]["content"]
            return {"tool_calls": [{"id": "c4", "function": {"name": "close_without_fix", "arguments": json.dumps({"kind": "nothing_wrong", "reason": "all correct"})}}]}

    inv, sql, session = fix.draft_checked(wh, as_dag(), inc, Talks(), open_world=True)
    world_eval.restore(wh)
    assert inv.nudged and session.closed["kind"] == "nothing_wrong" and Talks.turns == 4
