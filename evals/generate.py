"""Generate a labeled, synthetic incident set from a pipeline DAG.

Each incident injects one known fault (or, sometimes, two unrelated ones), marks everything
downstream stale, and adds noise. Labels record the true fix target, upstream cause, category
and whether the right action is to propose a fix or escalate.

These are synthetic. They test the triage logic for regressions; they say nothing about how
often real pipelines fail in each way.

    python -m evals.generate --n 100 --seed 7
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from triage.dag import PipelineDAG

ROOT = Path(__file__).resolve().parent.parent

ERRORS = {
    "missed_load": [
        "Job timed out after 3600s waiting on source extract",
        "Source API throttled requests; extract aborted",
        "Load skipped: no new files found in stage",
        "connection reset by peer during COPY INTO",
    ],
    "schema_drift": [
        "SQL compilation error: invalid identifier '{col}'",
        "SQL compilation error: column '{col}' does not exist",
        "Numeric value 'N/A' is not recognized in column {col}",
    ],
    "duplicate_merge": [
        "Duplicate row detected during DML action",
        "MERGE matched more than one source row for a target row",
        "Unique key violation on target table",
    ],
    "permission": [
        "Insufficient privileges to operate on schema ANALYTICS",
        "SQL access control error: access denied for role SVC_ETL",
    ],
    "unknown": [
        "Internal error: unexpected exception (code 300002)",
        "Process exited with code 137",
        "Unhandled exception in task runtime",
    ],
}
CHANGES = {
    "schema_drift": ["column {col} renamed to {col}_ID", "column {col} dropped", "type of {col} changed NUMBER to VARCHAR"],
    "duplicate_merge": ["source now sends versioned history rows", "source re-sent the previous day's file"],
}
COLS = ["REQ_ID", "CLIENT_KEY", "RATE_TYPE", "VENDOR_CODE", "START_DT"]


def _inject(dag: PipelineDAG, rng: random.Random, status: dict) -> dict:
    procs = [n for n, v in dag.nodes.items() if v.kind == "procedure"]
    loads = [n for n, v in dag.nodes.items() if v.kind == "load"]
    cat = rng.choices(["schema_drift", "duplicate_merge", "missed_load", "permission", "unknown"],
                      weights=[30, 20, 25, 10, 15])[0]
    col = rng.choice(COLS)
    cause = None
    if cat == "missed_load":
        target = rng.choice(loads)
        status[target] = {"status": "missing", "error": rng.choice(ERRORS[cat])}
    else:
        target = rng.choice(procs)
        status[target] = {"status": "failed", "error": rng.choice(ERRORS[cat]).format(col=col)}
        if cat in CHANGES and rng.random() < 0.75:
            cause = rng.choice(dag.parents[target])
            status[cause] = {"status": "changed", "changes": [rng.choice(CHANGES[cat]).format(col=col)]}
    for d in dag.downstream(target):
        status.setdefault(d, {"status": "stale", "note": "not refreshed since last successful run"})
    return {"target": target, "cause": cause, "category": cat}


def make_incident(dag: PipelineDAG, rng: random.Random, idx: int, symptom: str, noisy: float = 0.0) -> dict:
    status: dict = {}
    fault = _inject(dag, rng, status)
    if rng.random() < noisy:
        # misleading secondary message from a different failure mode, as real logs often have
        other = rng.choice([c for c in ERRORS if c not in (fault["category"], "unknown")])
        status[fault["target"]]["error"] = rng.choice(ERRORS[other]).format(col=rng.choice(COLS)) + "; then: " + status[fault["target"]]["error"]
    targets = [fault["target"]]
    ambiguous = False

    r = rng.random()
    if r < 0.08:
        # second, independent real failure on another branch: the right move is to escalate
        others = [n for n, v in dag.nodes.items() if v.kind == "procedure" and n not in targets
                  and n not in status and fault["target"] not in dag.upstream(n)]
        if others:
            o = rng.choice(others)
            status[o] = {"status": "failed", "error": rng.choice(ERRORS["duplicate_merge"])}
            for d in dag.downstream(o):
                status.setdefault(d, {"status": "stale"})
            targets.append(o)
            ambiguous = True
    elif r < 0.25:
        # red herring: a table on another branch is late but nothing upstream of it failed
        quiet = [n for n, v in dag.nodes.items() if v.kind == "table" and n not in status]
        if quiet:
            status[rng.choice(quiet)] = {"status": "stale", "note": "refresh ran late"}

    # Healthy nodes the probe reported on explicitly; the rest are simply absent.
    for n in dag.nodes:
        if n not in status and rng.random() < 0.5:
            status[n] = {"status": "ok"}

    expected_action = "escalate" if (ambiguous or fault["category"] == "unknown") else "propose"
    return {
        "id": f"SYN-{idx:03d}",
        "symptom_node": symptom,
        "symptom": "report numbers look wrong",
        "node_status": status,
        "label": {"fix_targets": targets, "cause": fault["cause"], "category": fault["category"],
                  "expected_action": expected_action, "ambiguous": ambiguous},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--pipeline", default=str(ROOT / "examples/pipeline.yaml"))
    ap.add_argument("--out", default=str(ROOT / "evals/data/incidents.jsonl"))
    ap.add_argument("--noisy", type=float, default=0.0, help="fraction of incidents with a misleading extra error line")
    a = ap.parse_args()
    dag = PipelineDAG.from_yaml(a.pipeline)
    report = next(n for n, v in dag.nodes.items() if v.kind == "report")
    rng = random.Random(a.seed)
    rows = [make_incident(dag, rng, i + 1, report, a.noisy) for i in range(a.n)]
    Path(a.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote {len(rows)} incidents to {a.out}")


if __name__ == "__main__":
    main()
