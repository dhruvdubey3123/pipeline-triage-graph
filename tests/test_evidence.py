from triage import evidence
from triage.dag import PipelineDAG

#   a -> b -> d (report)        c -> d
DAG = PipelineDAG.from_dict({"name": "t", "nodes": [{"id": n, "kind": "report" if n == "d" else "procedure", "owner": "x"}
                                                   for n in "abcde"],
                             "edges": [["a", "b"], ["b", "d"], ["c", "d"]]})


def status(**bad):
    return {n: {"status": bad.get(n, "ok")} for n in "abcde"}


def verdict(st, target, cause=None):
    return evidence.check(DAG, st, "d", target, cause).verdict


def test_supported_when_the_checks_first_fail_at_the_node_blamed():
    st = status(a="changed", b="changed", d="changed")
    assert verdict(st, "a") == "supported" and evidence.origin(DAG, st, "d") == "a"


def test_supported_across_a_gap_the_tracer_cannot_walk():
    st = status(a="deviates", d="deviates")          # b looks healthy in between
    assert verdict(st, "a") == "supported"


def test_stopping_partway_down_an_unhealthy_chain_is_contradicted():
    st = status(a="changed", b="changed", d="changed")
    ev = evidence.check(DAG, st, "d", "b")
    assert ev.verdict == "contradicted" and "upstream of b, at a" in ev.why


def test_blaming_a_healthy_node_downstream_of_the_failure_is_contradicted():
    assert verdict(status(a="deviates"), "d") == "contradicted"
    assert verdict(status(a="deviates"), "c") == "contradicted"       # a different branch cannot explain it


def test_a_node_that_is_not_upstream_of_the_report_is_contradicted():
    assert verdict(status(a="failed"), "e") == "contradicted"
    assert verdict(status(a="failed"), "nowhere") == "contradicted"


def test_when_every_check_passes_nothing_is_supported_and_nothing_is_contradicted():
    assert verdict(status(), "d") == "unverified" and evidence.origin(DAG, status(), "d") is None


def test_the_errored_node_with_its_upstream_cause_is_supported_only_if_the_cause_is_where_it_starts():
    st = status(a="changed", b="failed", d="stale")
    assert verdict(st, "b", cause="a") == "supported"
    assert verdict(st, "b") == "contradicted"                         # b alone: a is already failing above it
    assert verdict(st, "b", cause="c") == "contradicted"              # c is not upstream of b
    assert verdict(status(a="changed"), "b", cause="a") == "contradicted"   # b is healthy, nothing broke it


def test_two_places_where_checks_first_fail_leave_it_to_a_person():
    st = status(a="deviates", c="deviates")
    assert verdict(st, "a") == "unverified" and evidence.origin(DAG, st, "d") is None


def test_known_limit_a_break_invisible_at_its_source_supports_the_node_below_it():
    """a is where the SQL is wrong, but only b's checks fail. Blaming b is supported, and it is the
    wrong node. Supported means the first visible damage is there, not that the SQL there is wrong."""
    st = status(b="deviates")
    assert verdict(st, "b") == "supported"
    assert verdict(st, "a") == "unverified"


def test_no_diagnosis_is_unverified():
    assert verdict(status(a="failed"), None) == "unverified"
