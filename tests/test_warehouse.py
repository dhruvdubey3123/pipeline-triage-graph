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
