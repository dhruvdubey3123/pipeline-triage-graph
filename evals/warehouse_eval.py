"""Break the warehouse twelve ways and see what the rule-based tracer makes of each.

    python -m evals.warehouse_eval            # rebuild, write the manifest and the results
    python -m evals.warehouse_eval --check    # rebuild and fail if either file would change (CI)

For every break: build nothing new, edit one procedure, run the nightly cycle, read node health off
the warehouse with the probe, and hand the tracer the first report that came out wrong. Then put the
procedure back and confirm every table is the reference again.

The tracer's answer lands in one of four places:

    target      it named the procedure that was edited
    cause_only  it named the procedure that errored and gave the edited one as its upstream cause
    none        it found no root cause, so the incident goes to a person
    wrong       it named something else

Twelve incidents is a regression set, not a benchmark. It is here so the numbers have to be earned
against SQL that runs, and so a change to the tracer or the probe shows up as a diff.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from triage.dag import trace
from triage.warehouse import BREAKS, Warehouse, as_dag, observe
from triage.warehouse.breaks import split_of
from triage.warehouse.pipeline import BY_ID, REPORTS

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "evals/data/warehouse_incidents.json"
RESULTS = ROOT / "evals/warehouse_results.json"


def incidents(wh: Warehouse) -> list[dict]:
    """Every break as an incident the triage graph can read, with the label kept apart from it."""
    out = []
    for b in BREAKS:
        wh.set_sql(b.node, b.broken_sql())
        wh.run_cycle()
        wrong, status = wh.affected(), observe(wh)
        symptom = next((r for r in REPORTS if r in wrong), None)
        out.append({
            "id": b.id, "symptom_node": symptom, "symptom": "report numbers look wrong",
            "node_status": status,
            "label": {"fix_targets": [b.node], "family": b.family, "split": split_of(b), "what": b.what,
                      "diff": {"find": b.find, "replace": b.replace},
                      "silent": all(s["status"] != "failed" for s in status.values()),
                      "probe_sees_it_at_the_source": status[b.node]["status"] != "ok",
                      "tables_actually_wrong": wrong},
        })
        wh.set_sql(b.node, BY_ID[b.node].sql)
        wh.run_cycle()
        if wh.affected():
            raise RuntimeError(f"{b.id}: reverting the break did not restore {wh.affected()}")
    return out


def verdict(inc: dict, dag) -> str:
    if inc["symptom_node"] is None:
        return "no_symptom"
    res, edited = trace(dag, inc["node_status"], inc["symptom_node"]), inc["label"]["fix_targets"][0]
    if res.fix_target == edited:
        return "target"
    if res.upstream_cause == edited:
        return "cause_only"
    return "none" if res.fix_target is None else "wrong"


def run() -> tuple[list[dict], dict]:
    wh = Warehouse().build()
    dag = as_dag()
    incs = incidents(wh)
    wh.close()
    rows = [{"id": i["id"], "split": i["label"]["split"], "edited": i["label"]["fix_targets"][0],
             "symptom_node": i["symptom_node"], "silent": i["label"]["silent"],
             "probe_sees_it_at_the_source": i["label"]["probe_sees_it_at_the_source"], "tracer": verdict(i, dag)}
            for i in incs]
    tally: dict[str, dict[str, int]] = {}
    for r in rows:
        for key in (r["split"], "all"):
            t = tally.setdefault(key, {"incidents": 0, "target": 0, "cause_only": 0, "none": 0, "wrong": 0})
            t["incidents"] += 1
            t[r["tracer"]] = t.get(r["tracer"], 0) + 1
    manifest = [{"id": i["id"], "symptom_node": i["symptom_node"],
                 "node_status": {n: s["status"] for n, s in i["node_status"].items()}, **i["label"]} for i in incs]
    results = {"scale_factor": 0.01, "incidents": len(rows), "silent": sum(r["silent"] for r in rows),
               "invisible_to_every_probe_check": sum(not r["probe_sees_it_at_the_source"] for r in rows),
               "mode": "rule-based tracer on node health read from the warehouse",
               "by_split": tally, "rows": rows}
    return manifest, results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if the committed manifest or results would change")
    a = ap.parse_args()
    manifest, results = run()
    texts = {MANIFEST: json.dumps(manifest, indent=1) + "\n", RESULTS: json.dumps(results, indent=2) + "\n"}
    if a.check:
        stale = [p.name for p, t in texts.items() if not p.exists() or p.read_text() != t]
        if stale:
            print("WAREHOUSE EVAL DRIFTED: " + ", ".join(stale) + " would change. Rerun without --check and review the diff.")
            sys.exit(1)
        print("warehouse eval matches the committed results")
    else:
        for p, t in texts.items():
            p.write_text(t)
    print(json.dumps(results["by_split"], indent=2))


if __name__ == "__main__":
    main()
