import json
from contextlib import contextmanager
from pathlib import Path

import pytest

pytest.importorskip("duckdb")

from evals import warehouse_eval  # noqa: E402
from triage.dag import trace  # noqa: E402
from triage.warehouse import BREAKS, OOD_FAMILIES, Warehouse, as_dag, observe  # noqa: E402
from triage.warehouse.breaks import split_of  # noqa: E402
from triage.warehouse.pipeline import BY_ID, NODES, REPORTS  # noqa: E402

BY_NAME = {b.id: b for b in BREAKS}


@pytest.fixture(scope="module")
def wh():
    w = Warehouse().build()
    yield w
    w.close()


@contextmanager
def broken(wh, name):
    b = BY_NAME[name]
    wh.set_sql(b.node, b.broken_sql())
    wh.run_cycle()
    try:
        yield b
    finally:
        for n in NODES:
            wh.set_sql(n.id, n.sql)
        wh.run_cycle()
        assert wh.affected() == []


def test_two_builds_hold_the_same_rows(wh):
    other = Warehouse().build()
    assert [other.checksum(n.id) for n in NODES] == [wh.checksum(n.id) for n in NODES]
    other.close()


def test_the_reference_run_is_healthy_and_every_key_is_unique(wh):
    assert all(s["status"] == "ok" for s in observe(wh).values())
    assert all(wh.last_run(n.id)["stats"]["duplicate_keys"] == 0 for n in NODES)
    assert wh.outcome(REPORTS[0]) == {"functional": True, "integrity": True, "still_wrong": [],
                                      "procedures_changed": [], "changed_outside_approval": []}


def test_the_source_has_the_gaps_null_handling_needs(wh):
    assert wh.last_run("stg_customer")["stats"]["nulls"]["nation_id"] > 0
    assert wh.last_run("int_part_cost")["stats"]["nulls"]["supply_cost"] > 0


@pytest.mark.parametrize("name", list(BY_NAME))
def test_every_break_starts_at_the_procedure_that_was_edited_and_reaches_a_report(wh, name):
    with broken(wh, name) as b:
        wrong = wh.affected()
        assert wrong[0] == b.node, "nothing upstream of the edit moved"
        assert any(r in wrong for r in REPORTS), "a break nobody could notice is not an incident"
        assert wh.changed_procedures() == [b.node]


def test_most_breaks_raise_nothing(wh):
    silent = []
    for name in BY_NAME:
        with broken(wh, name):
            if all(s["status"] != "failed" for s in observe(wh).values()):
                silent.append(name)
    assert len(silent) == 10 and not any(n.startswith("dropped_column") for n in silent)


def test_a_break_that_only_changes_values_passes_every_check_a_person_could_run(wh):
    with broken(wh, "date_window.trailing_six_days"):
        assert all(s["status"] == "ok" for s in observe(wh).values())
        assert wh.affected() == ["rpt_revenue_dashboard"]      # and the report is still wrong


def test_the_tracer_stops_where_the_damage_stops_showing(wh):
    """The customers dropped by an inner join are missing from the mart too, so the mart's own checks
    pass and the tracer, which only walks through unhealthy nodes, never reaches the cause."""
    with broken(wh, "null_handling.inner_join_nation"):
        status = observe(wh)
        assert status["int_customer_geo"]["status"] == "deviates" and status["mart_customer_revenue"]["status"] == "ok"
        assert trace(as_dag(), status, "rpt_revenue_dashboard").fix_target is None


def test_putting_the_sql_back_is_a_fix_on_both_counts(wh):
    with broken(wh, "bad_join_key.part_without_supplier") as b:
        assert wh.outcome("rpt_regional_summary", allowed=(b.node,))["functional"] is False
        wh.set_sql(b.node, BY_ID[b.node].sql)
        wh.run_cycle()
        out = wh.outcome("rpt_regional_summary", allowed=(b.node,))
        assert out["functional"] and out["integrity"]


def test_a_fix_in_different_words_counts_if_the_rows_are_right(wh):
    with broken(wh, "bad_join_key.part_without_supplier") as b:
        wh.set_sql(b.node, "select l.order_id, l.line_no, l.quantity * ps.ps_supplycost as supply_cost "
                           "from stg_lineitem l left join raw.partsupp ps "
                           "on (ps.ps_suppkey, ps.ps_partkey) = (l.supplier_id, l.part_id)")
        wh.run_cycle()
        out = wh.outcome("rpt_regional_summary", allowed=(b.node,))
        assert out["functional"] and out["integrity"] and out["procedures_changed"] == [b.node]


def test_hard_coding_the_report_brings_it_back_and_fails_integrity(wh):
    """The shortcut a fix scored only on 'did the report come back' is rewarded for taking."""
    right = wh.db.execute("select margin_rate from rpt_regional_summary limit 1").fetchone()[0]
    with broken(wh, "bad_join_key.part_without_supplier") as b:
        report = BY_ID["rpt_regional_summary"]
        wh.set_sql(report.id, report.sql.replace(
            "(select sum(margin) / sum(revenue) from mart_supplier_performance)", f"{right!r}::double"))
        wh.run_cycle()
        out = wh.outcome("rpt_regional_summary", allowed=(b.node,))
        assert out["functional"] is True
        assert out["integrity"] is False
        assert out["still_wrong"] == ["int_part_cost", "mart_supplier_performance"]
        assert out["changed_outside_approval"] == ["rpt_regional_summary"]


def test_two_families_are_held_out_whole():
    by_split = {}
    for b in BREAKS:
        by_split.setdefault(split_of(b), set()).add(b.family)
    assert by_split["ood"] == set(OOD_FAMILIES)
    assert not by_split["ood"] & (by_split["evolve"] | by_split["held_in"])
    assert by_split["evolve"] == by_split["held_in"]


def test_the_committed_manifest_and_results_are_what_the_code_produces():
    manifest, results = warehouse_eval.run()
    assert json.loads(warehouse_eval.MANIFEST.read_text()) == manifest
    assert json.loads(warehouse_eval.RESULTS.read_text()) == results


# ---------------------------------------------------------------- the investigator on the warehouse

class Scripted:
    """A chat model that makes the calls it is given, in order, then answers in plain text."""
    model = "scripted"

    def __init__(self, calls, then="I cannot find it."):
        self.calls, self.then, self.seen = list(calls), then, []

    def chat(self, messages, tools):
        self.seen.append((messages, tools))
        if not self.calls:
            return {"content": self.then}
        name, args = self.calls.pop(0)
        return {"tool_calls": [{"id": f"c{len(self.seen)}", "function": {"name": name, "arguments": json.dumps(args)}}]}


def _incident(wh, b):
    symptom = next(r for r in REPORTS if r in wh.affected())
    return {"id": b.id, "symptom_node": symptom, "symptom": wh.complaint(symptom), "node_status": observe(wh)}


def test_the_complaint_names_the_numbers_and_not_the_cause(wh):
    with broken(wh, "date_window.trailing_six_days"):
        assert wh.complaint("rpt_revenue_dashboard") == "trailing_7d_revenue in rpt_revenue_dashboard looks wrong"
    with broken(wh, "dropped_column.order_date_dropped"):
        assert wh.complaint("rpt_revenue_dashboard") == "rpt_revenue_dashboard has not refreshed"
    with broken(wh, "null_handling.inner_join_nation"):
        assert wh.complaint("rpt_revenue_dashboard") == "rpt_revenue_dashboard has 25 rows where there were 30"


def test_the_model_can_read_the_sql_that_every_health_check_passed(wh):
    from triage.warehouse.investigate import investigate_warehouse

    with broken(wh, "date_window.trailing_six_days") as b:
        model = Scripted([("upstream", {"node": "rpt_revenue_dashboard"}),
                          ("read_sql", {"node": "rpt_revenue_dashboard"}),
                          ("submit_diagnosis", {"fix_target": "rpt_revenue_dashboard",
                                                "reason": "the trailing total subtracts 6, so it sums six days"}),
                          ("read_sql", {"node": "stg_orders"})])
        inv = investigate_warehouse(wh, as_dag(), _incident(wh, b), model)
        assert inv.submitted and inv.fix_target == b.node and inv.category is None
        assert "- 6)" in inv.steps[1]["result"], "it saw the deployed SQL, not the reference"
        assert len(inv.steps) == 3, "the first valid diagnosis ends it"
        offered = {t["function"]["name"] for t in model.seen[0][1]}
        assert offered == {"pipeline", "status", "upstream", "profile", "read_sql", "submit_diagnosis"}


def test_a_model_that_cannot_find_it_says_so_and_nothing_is_submitted(wh):
    from triage.warehouse.investigate import investigate_warehouse

    with broken(wh, "null_handling.cost_not_defaulted") as b:
        inv = investigate_warehouse(wh, as_dag(), _incident(wh, b),
                                    Scripted([("profile", {"node": "mart_supplier_performance"})]))
        assert not inv.submitted and inv.fix_target is None and inv.answer == "I cannot find it."


def test_a_node_that_is_not_in_the_pipeline_comes_back_as_an_error_it_can_correct(wh):
    from triage.warehouse.investigate import investigate_warehouse

    with broken(wh, "bad_join_key.part_without_supplier") as b:
        inv = investigate_warehouse(wh, as_dag(), _incident(wh, b), Scripted([
            ("read_sql", {"node": "raw.partsupp"}),
            ("submit_diagnosis", {"fix_target": "partsupp", "reason": "x"}),
            ("profile", {"node": "int_part_cost"}),
            ("submit_diagnosis", {"fix_target": "int_part_cost", "reason": "joined on the part alone"})]))
        assert [s["ok"] for s in inv.steps] == [False, False, True, True] and inv.fix_target == b.node
        assert "237636 rows (last good run 60175)" in inv.steps[2]["result"]


def test_the_investigator_is_told_no_family_names_and_has_no_way_to_ask_what_changed(wh):
    from triage.warehouse.investigate import INSTRUCTIONS, tools

    with broken(wh, "changed_grain.one_row_per_order") as b:
        specs = tools(wh, as_dag(), _incident(wh, b))
    text = (INSTRUCTIONS + " ".join(t.name + " " + t.description + json.dumps(t.args.model_json_schema()) for t in specs)).lower()
    for word in ("grain", "null handling", "join key", "type drift", "date window", "dropped column", "what changed", "changed since", "recently"):
        assert word not in text


def test_scoring_a_model_run_counts_what_todays_cross_check_would_hold_back():
    """A model that reads the SQL is right on a break the tracer cannot see, and exact agreement with
    the tracer sends that right answer to a person."""
    answers = {b.id: b.node for b in BREAKS}

    class Oracle:
        model = "oracle"

        def chat(self, messages, tools):
            if any(m["role"] == "tool" for m in messages):
                return {"content": "done"}
            inc = messages[1]["content"].split(":")[0].removeprefix("Incident ")
            return {"tool_calls": [{"id": "c", "function": {"name": "submit_diagnosis", "arguments": json.dumps(
                {"fix_target": answers[inc], "reason": "read the SQL"})}}]}

    summary = warehouse_eval.run_investigator(Oracle())["summary"]["all"]
    assert summary["model named the edited procedure"] == 12 and summary["tracer named the edited procedure"] == 5
    assert summary["model named the edited procedure where the tracer gave no diagnosis"] == 5
    assert summary["model named the edited procedure and it is held back for a person"] == 7
    assert summary["model named something else and it would reach the approval gate"] == 0
    # the evidence check lets through the 9 the health checks can see, and not the 3 they cannot
    assert summary["evidence check: supported"] == 9 and summary["evidence check: unverified"] == 3
    assert summary["evidence check: edited procedure named and held back for a person"] == 3


def test_the_model_and_the_tracer_are_judged_by_the_same_rule():
    judge = warehouse_eval.judge
    assert judge("stg_orders", None, "stg_orders") == "target"
    assert judge("int_order_items", "stg_orders", "stg_orders") == "cause_only"
    assert judge("int_order_items", None, "stg_orders") == "wrong"
    assert judge(None, None, "stg_orders") == "none"


def test_the_saved_model_run_is_counted_under_the_current_rules():
    saved = json.loads(warehouse_eval.LLM_RESULTS.read_text())
    assert saved["summary"] == warehouse_eval._summarise(saved["rows"])
    for scope in saved["summary"].values():
        assert scope["evidence check: something else reaches the approval gate"] == 0
    assert all(r["model"] == warehouse_eval.judge(r["model_target"] if r["submitted"] else None, r["model_cause"], r["edited"])
               for r in saved["rows"])


def test_the_probe_answers_the_graphs_data_checks(wh):
    from triage.warehouse import WarehouseProbe

    with broken(wh, "changed_grain.one_row_per_order"):
        probe = WarehouseProbe(wh)
        assert "share a key (customer_id) that was unique" in probe.check("int_customer_geo", {})
        assert probe.check("stg_orders", {}) is None


def test_the_evidence_check_supports_a_right_answer_the_tracer_cannot_confirm(wh):
    from triage import evidence

    with broken(wh, "bad_join_key.part_without_supplier") as b:
        st, dag = observe(wh), as_dag()
        assert trace(dag, st, "rpt_regional_summary").fix_target is None
        assert evidence.check(dag, st, "rpt_regional_summary", b.node).verdict == "supported"
        assert evidence.check(dag, st, "rpt_regional_summary", "rpt_regional_summary").verdict == "contradicted"


def test_the_first_failure_rule_names_nine_and_never_the_wrong_node():
    rows = json.loads(warehouse_eval.RESULTS.read_text())["rows"]
    assert [r["first_failure"] for r in rows].count("target") == 9
    assert {r["first_failure"] for r in rows} == {"target", "none"}


# ---------------------------------------------------------------- drafting a fix and verifying it by running it

def _fix_call(node, sql, reason="fixed"):
    return ("submit_diagnosis", {"fix_target": node, "sql": sql, "reason": reason})


def test_a_correct_fix_for_a_break_no_check_can_see_is_kept(wh):
    from triage.warehouse import fix

    with broken(wh, "date_window.trailing_six_days") as b:
        inc = _incident(wh, b)
        assert all(s["status"] == "ok" for s in inc["node_status"].values())
        repaired = wh.sql(b.node).replace("- 6)", "- 7)")
        inv, sql = fix.draft(wh, as_dag(), inc, Scripted([("read_sql", {"node": b.node}), _fix_call(b.node, repaired + ";")]))
        assert inv.submitted and sql == repaired
        out = fix.apply_and_verify(wh, b.node, sql, inc["symptom_node"])
        assert out["functional"] and out["integrity"] and out["kept"] and not out["rolled_back"]
        assert wh.affected() == []


def test_a_fix_that_only_brings_the_report_back_is_rolled_back(wh):
    from triage.warehouse import fix

    right = wh.db.execute("select margin_rate from rpt_regional_summary limit 1").fetchone()[0]
    with broken(wh, "bad_join_key.part_without_supplier"):
        before = wh.affected()
        report = BY_ID["rpt_regional_summary"]
        shortcut = report.sql.replace("(select sum(margin) / sum(revenue) from mart_supplier_performance)", f"{right!r}::double")
        out = fix.apply_and_verify(wh, report.id, shortcut, report.id)
        assert out["functional"] is True and out["integrity"] is False
        assert out["kept"] is False and out["rolled_back"] is True
        assert wh.sql(report.id) == report.sql and wh.affected() == before, "left exactly as it was found"


def test_sql_that_does_not_run_goes_back_to_the_model_and_nothing_is_deployed(wh):
    from triage.warehouse import fix

    with broken(wh, "date_window.trailing_six_days") as b:
        deployed = wh.sql(b.node)
        good = deployed.replace("- 6)", "- 7)")
        inv, sql = fix.draft(wh, as_dag(), _incident(wh, b), Scripted([
            _fix_call(b.node, "select no_such_column from mart_customer_revenue"),
            _fix_call(b.node, good)]))
        assert [s["ok"] for s in inv.steps] == [False, True] and "does not run" in inv.steps[0]["result"]
        assert sql == good and wh.sql(b.node) == deployed, "drafting changes nothing"


@pytest.mark.parametrize("sql, said", [
    ("select * from reference.rpt_regional_summary", "cannot read the reference"),
    ("select node, row_count from golden", "cannot read the reference"),
    ("drop table stg_orders", "one SELECT"),
    ("select 1; drop table stg_orders", "one SELECT"),
])
def test_a_fix_cannot_read_the_answer_key_or_do_anything_but_select(wh, sql, said):
    from triage.warehouse import fix

    with pytest.raises(ValueError, match=said):
        fix.validate(wh, as_dag(), "rpt_regional_summary", sql)


def test_no_fix_is_an_outcome_and_not_a_guess(wh):
    from triage.warehouse import fix

    with broken(wh, "null_handling.cost_not_defaulted") as b:
        inv, sql = fix.draft(wh, as_dag(), _incident(wh, b), Scripted([("read_sql", {"node": b.node})]))
        assert sql is None and not inv.submitted and inv.answer == "I cannot find it."


def test_scoring_a_fix_run_keeps_the_two_answers_apart():
    """A model that always submits the reference SQL fixes all twelve, three of them invisible to checks."""
    reference = {b.id: (b.node, BY_ID[b.node].sql) for b in BREAKS}

    class Oracle:
        model = "oracle"

        def chat(self, messages, tools):
            if any(m["role"] == "tool" for m in messages):
                return {"content": "done"}
            node, sql = reference[messages[1]["content"].split(":")[0].removeprefix("Incident ")]
            return {"tool_calls": [{"id": "c", "function": {"name": "submit_diagnosis", "arguments": json.dumps(
                {"fix_target": node, "sql": sql, "reason": "restored"})}}]}

    summary = warehouse_eval.run_fixer(Oracle())["summary"]
    assert summary["all"]["fixed"] == 12 and summary["all"]["functional"] == 12 and summary["all"]["integrity"] == 12
    unseen = summary["no check sees it"]
    assert unseen["incidents"] == 3 and unseen["fixed"] == 3 and unseen["functional"] == 3 and unseen["integrity"] == 3
    assert sum(unseen[k] for k in warehouse_eval.FIX_OUTCOMES) == 3


# ---------------------------------------------------------------- the checker in the model's hands, and in the harness

def test_the_model_checks_rehearses_and_submits_and_the_rehearsal_keeps_nothing(wh):
    from triage.warehouse import fix

    with broken(wh, "bad_join_key.part_without_supplier") as b:
        inc, deployed = _incident(wh, b), wh.sql(b.node)
        inv, sql, session = fix.draft_checked(wh, as_dag(), inc, Scripted([
            ("where_checks_fail", {}), ("check_node", {"node": b.node}),
            ("try_fix", {"fix_target": b.node, "sql": BY_ID[b.node].sql}),
            _fix_call(b.node, BY_ID[b.node].sql)]))
        assert "checks first fail at: int_part_cost" in inv.steps[0]["result"]
        assert inv.steps[1]["result"].startswith("supported") and inv.steps[2]["result"].startswith("verified")
        assert inv.submitted and sql == BY_ID[b.node].sql
        assert session.asked_where and session.tries == 1 and session.verified_on_try == 1 and session.refusals == []
        assert wh.sql(b.node) == deployed, "rehearsing and submitting deploy nothing"


def test_the_harness_refuses_a_fix_for_a_node_the_model_never_checked(wh):
    from triage.warehouse import fix

    with broken(wh, "bad_join_key.part_without_supplier") as b:
        inv, sql, session = fix.draft_checked(wh, as_dag(), _incident(wh, b), Scripted([
            ("try_fix", {"fix_target": b.node, "sql": BY_ID[b.node].sql}), _fix_call(b.node, BY_ID[b.node].sql)]))
        assert sql is None and not inv.submitted and session.tries == 0
        assert [s["ok"] for s in inv.steps] == [False, False] and len(session.refusals) == 2
        assert "call check_node on int_part_cost" in inv.steps[0]["result"]


def test_the_harness_refuses_a_fix_at_a_node_the_checks_contradict(wh):
    from triage.warehouse import fix

    with broken(wh, "bad_join_key.part_without_supplier") as b:
        report = BY_ID["rpt_regional_summary"]
        inv, sql, session = fix.draft_checked(wh, as_dag(), _incident(wh, b), Scripted([
            ("check_node", {"node": report.id}), ("try_fix", {"fix_target": report.id, "sql": report.sql})]))
        assert inv.steps[0]["result"].startswith("contradicted") and "int_part_cost" in inv.steps[0]["result"]
        assert sql is None and session.tries == 0 and "contradict" in session.refusals[0]


def test_only_sql_a_rehearsal_passed_can_be_submitted(wh):
    from triage.warehouse import fix

    with broken(wh, "date_window.trailing_six_days") as b:
        good = wh.sql(b.node).replace("- 6)", "- 7)")
        inv, sql, session = fix.draft_checked(wh, as_dag(), _incident(wh, b), Scripted([
            ("where_checks_fail", {}), ("check_node", {"node": b.node}),
            _fix_call(b.node, good),                                              # never rehearsed
            ("try_fix", {"fix_target": b.node, "sql": good}),
            _fix_call(b.node, good.replace("- 7)", "- 8)")),                      # not the SQL that passed
            _fix_call(b.node, good)]))
        assert inv.steps[0]["result"].startswith("nowhere") and inv.steps[1]["result"].startswith("unverified")
        assert [s["ok"] for s in inv.steps] == [True, True, False, True, False, True]
        assert sql == good and len(session.refusals) == 2


def test_three_failed_tries_end_it_and_nothing_can_be_submitted(wh):
    from triage.warehouse import fix

    with broken(wh, "date_window.trailing_six_days") as b:
        deployed = wh.sql(b.node)
        wrong = [deployed.replace("- 6)", f"- {n})") for n in (3, 4, 5, 2)]
        inv, sql, session = fix.draft_checked(wh, as_dag(), _incident(wh, b), Scripted(
            [("check_node", {"node": b.node})] + [("try_fix", {"fix_target": b.node, "sql": w}) for w in wrong]
            + [_fix_call(b.node, wrong[0])], then="I could not verify a fix."))
        assert session.tries == 3 and sql is None and inv.answer == "I could not verify a fix."
        assert "0 tries left" in inv.steps[3]["result"] and "all 3 tries are used" in inv.steps[4]["result"]
        assert wh.sql(b.node) == deployed


def test_the_rehearsal_never_says_which_tables_are_still_wrong(wh):
    from triage.warehouse import fix

    with broken(wh, "bad_join_key.part_without_supplier") as b:
        session = fix.Session()
        tools = {t.name: t for t in fix.checked_tools(wh, as_dag(), _incident(wh, b), session)}
        report = BY_ID["rpt_regional_summary"]
        right = wh.db.execute("select margin_rate from reference.rpt_regional_summary limit 1").fetchone()[0]
        shortcut = report.sql.replace("(select sum(margin) / sum(revenue) from mart_supplier_performance)", f"{right!r}::double")
        session.checked[report.id] = "unverified"           # as if the checks had nothing to say about it
        said = tools["try_fix"].fn(fix_target=report.id, sql=shortcut)
        assert said.startswith("not verified: the report came out right, but other tables")
        assert not any(n.id in said for n in NODES)


def test_a_model_that_uses_the_checker_well_fixes_all_twelve_on_the_first_try():
    reference = {b.id: (b.node, BY_ID[b.node].sql) for b in BREAKS}

    class Careful:
        model = "careful"

        def chat(self, messages, tools):
            node, sql = reference[messages[1]["content"].split(":")[0].removeprefix("Incident ")]
            done = sum(m["role"] == "tool" for m in messages)
            plan = [("where_checks_fail", {}), ("check_node", {"node": node}), ("try_fix", {"fix_target": node, "sql": sql}),
                    ("submit_diagnosis", {"fix_target": node, "sql": sql, "reason": "restored"})]
            if done >= len(plan):
                return {"content": "done"}
            return {"tool_calls": [{"id": f"c{done}", "function": {"name": plan[done][0], "arguments": json.dumps(plan[done][1])}}]}

    res = warehouse_eval.run_fixer(Careful(), checker=True)
    summary = res["summary"]["all"]
    assert summary["fixed"] == 12 and summary["fixed on the first try"] == 12 and summary["tries"] == 12
    assert summary["harness refusals"] == 0 and summary["asked where checks fail"] == 12
    assert res["summary"]["no check sees it"]["fixed"] == 3


def test_a_correct_fix_with_its_columns_in_another_order_is_still_a_fix(wh):
    """Seen in a model run: the dropped column was put back at the end of the SELECT, every table
    downstream was right, and the fix was rejected three times because the checksum read columns by position."""
    from triage.warehouse import fix

    with broken(wh, "dropped_column.order_date_dropped") as b:
        reordered = ("select o_orderkey as order_id, o_custkey as customer_id, o_orderstatus as status, "
                     "o_orderdate as order_date from raw.orders")
        out = fix.rehearse(wh, b.node, reordered, "rpt_revenue_dashboard")
        assert out["functional"] and out["integrity"]
        short = "select o_orderkey as order_id, o_custkey as customer_id, o_orderdate as order_date from raw.orders"
        assert fix.rehearse(wh, b.node, short, "rpt_revenue_dashboard")["integrity"] is False, "a missing column is not"


def test_a_model_that_stops_after_a_verified_rehearsal_has_still_produced_the_fix(wh):
    """Seen in a model run: three incidents had a verified fix and no closing call, one of them with
    the fix announced in prose. The rehearsal is the gate, so the harness takes the verified fix."""
    from triage.warehouse import fix

    with broken(wh, "date_window.trailing_six_days") as b:
        good = wh.sql(b.node).replace("- 6)", "- 7)")
        inv, sql, session = fix.draft_checked(wh, as_dag(), _incident(wh, b), Scripted(
            [("check_node", {"node": b.node}),
             ("try_fix", {"fix_target": b.node, "sql": good, "reason": "the window was six days"})], then="**Fix Submitted**"))
        assert not inv.submitted and sql == good and inv.fix_target == b.node and session.taken_from_rehearsal
        assert inv.reason == "the window was six days", "the reason a person will read comes with the rehearsal"
        out = fix.apply_and_verify(wh, inv.fix_target, sql, "rpt_revenue_dashboard")
        assert out["kept"]


def test_a_model_that_could_not_be_reached_is_not_counted_as_a_model_that_failed():
    row = {"kept": False, "submitted": False, "functional": False, "said_instead": "model unavailable (HTTPStatusError)"}
    assert warehouse_eval._fix_result(row) == "model unavailable"
    assert warehouse_eval._fix_result({**row, "said_instead": "I could not verify a fix."}) == "no fix submitted"
    assert warehouse_eval._fix_result({**row, "said_instead": None, "checker": {"verified on try": 1}}) \
        == "verified in rehearsal, never submitted"


def test_the_saved_fix_runs_are_labelled_under_the_current_rules():
    runs = sorted((warehouse_eval.ROOT / "evals/runs").glob("*.json"))
    assert len(runs) == 3
    for path in (warehouse_eval.FIX_RESULTS, warehouse_eval.CHECKED_FIX_RESULTS, *runs):
        saved = json.loads(path.read_text())
        assert saved["host"] in ("openrouter.ai", "api.groq.com")
        assert saved["summary"] == warehouse_eval._summarise_fixes(saved["rows"])
        assert all(r["result"] == warehouse_eval._fix_result(r) for r in saved["rows"])


def test_the_reported_comparison_is_one_model_on_one_host():
    alone = json.loads(warehouse_eval.FIX_RESULTS.read_text())
    checked = json.loads(warehouse_eval.CHECKED_FIX_RESULTS.read_text())
    assert (alone["model"], alone["host"]) == (checked["model"], checked["host"])
    a, c = alone["summary"]["all"], checked["summary"]["all"]
    assert (a["fixed"], a["did not fix it"]) == (4, 8) and (c["fixed"], c["no fix submitted"]) == (9, 3)
    assert c["did not fix it"] == 0 and c["report came back, warehouse still wrong"] == 0
    assert all(r["kept"] == (r["functional"] and r["integrity"]) for r in alone["rows"] + checked["rows"])
