"""Break the warehouse twelve ways and see what the rule-based tracer makes of each.

    python -m evals.warehouse_eval            # rebuild, write the manifest and the results
    python -m evals.warehouse_eval --check    # rebuild and fail if either file would change (CI)
    python -m evals.warehouse_eval --investigator
                                              # the LLM investigator on the same incidents, with read-only
                                              # warehouse tools; writes evals/warehouse_results_llm.json

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
LLM_RESULTS = ROOT / "evals/warehouse_results_llm.json"


def each_break(wh: Warehouse):
    """Yield every break as an incident while the warehouse is still in the broken state, then put the
    procedure back and confirm every table is the reference again. The label is kept apart from what
    the triage reads."""
    for b in BREAKS:
        wh.set_sql(b.node, b.broken_sql())
        wh.run_cycle()
        wrong, status = wh.affected(), observe(wh)
        symptom = next((r for r in REPORTS if r in wrong), None)
        yield {
            "id": b.id, "symptom_node": symptom,
            "symptom": wh.complaint(symptom) if symptom else "nothing visible in a report",
            "node_status": status,
            "label": {"fix_targets": [b.node], "family": b.family, "split": split_of(b), "what": b.what,
                      "diff": {"find": b.find, "replace": b.replace},
                      "silent": all(s["status"] != "failed" for s in status.values()),
                      "probe_sees_it_at_the_source": status[b.node]["status"] != "ok",
                      "tables_actually_wrong": wrong},
        }
        wh.set_sql(b.node, BY_ID[b.node].sql)
        wh.run_cycle()
        if wh.affected():
            raise RuntimeError(f"{b.id}: reverting the break did not restore {wh.affected()}")


def incidents(wh: Warehouse) -> list[dict]:
    return list(each_break(wh))


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
    manifest = [{"id": i["id"], "symptom_node": i["symptom_node"], "symptom": i["symptom"],
                 "node_status": {n: s["status"] for n, s in i["node_status"].items()}, **i["label"]} for i in incs]
    results = {"scale_factor": 0.01, "incidents": len(rows), "silent": sum(r["silent"] for r in rows),
               "invisible_to_every_probe_check": sum(not r["probe_sees_it_at_the_source"] for r in rows),
               "mode": "rule-based tracer on node health read from the warehouse",
               "by_split": tally, "rows": rows}
    return manifest, results


def run_investigator(model, only: list[str] | None = None, out: Path | None = None, max_steps: int = 18) -> dict:
    """The LLM investigator on the same incidents, next to the tracer.

    Under the cross-check as it stands today a diagnosis reaches the approval gate only when it names
    the node the tracer named. So each incident also records what that rule would do with the model's
    answer: a right answer the tracer cannot confirm is held back for a person, and that count is the
    cost of the rule on a warehouse where the tracer has blind spots.
    """
    from triage.warehouse.investigate import investigate_warehouse

    wh, dag, rows = Warehouse().build(), as_dag(), []
    res = {"model": getattr(model, "model", "?"), "mode": "LLM investigator with read-only warehouse tools, next to the tracer",
           "incidents": 0, "rows": rows}
    for inc in each_break(wh):
        if (only and inc["id"] not in only) or inc["symptom_node"] is None:
            continue
        print(f"  {inc['id']} ...", file=sys.stderr, flush=True)
        inv = investigate_warehouse(wh, dag, inc, model, max_steps=max_steps)
        edited, tr = inc["label"]["fix_targets"][0], trace(dag, inc["node_status"], inc["symptom_node"])
        right = bool(inv.submitted and inv.fix_target == edited)
        agrees = bool(inv.submitted and inv.fix_target == tr.fix_target)
        rows.append({
            "id": inc["id"], "split": inc["label"]["split"], "edited": edited, "symptom": inc["symptom"],
            "submitted": inv.submitted, "model_target": inv.fix_target, "model_cause": inv.cause or None,
            "model_right": right, "reason": inv.reason, "said_instead": inv.answer or None,
            "tracer": verdict(inc, dag), "tracer_target": tr.fix_target,
            "under_todays_cross_check": "approval gate" if agrees else "a person",
            "tool_calls": len(inv.steps),
            "calls": [f"{s['tool']}({', '.join(str(v) for v in s['args'].values())[:60]})" + ("" if s["ok"] else " -> error")
                      for s in inv.steps],
        })
        res["incidents"] = len(rows)
        res["summary"] = _summarise(rows)
        if out:
            out.write_text(json.dumps(res, indent=2) + "\n")       # after every incident: a rate limit loses one, not all
        print(f"    model: {inv.fix_target if inv.submitted else 'no diagnosis'} "
              f"({'right' if right else 'wrong' if inv.submitted else 'abstained'}), tracer: {rows[-1]['tracer']}",
              file=sys.stderr, flush=True)
    wh.close()
    return res


def _summarise(rows: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for r in rows:
        for key in (r["split"], "all"):
            t = out.setdefault(key, {"incidents": 0, "model right": 0, "model wrong": 0, "model gave no diagnosis": 0,
                                     "tracer named the edited procedure": 0,
                                     "model right where the tracer found nothing": 0,
                                     "wrong diagnoses that would reach the approval gate": 0,
                                     "right diagnoses held back because the tracer could not confirm them": 0})
            t["incidents"] += 1
            t["model right"] += r["model_right"]
            t["model wrong"] += r["submitted"] and not r["model_right"]
            t["model gave no diagnosis"] += not r["submitted"]
            t["tracer named the edited procedure"] += r["tracer"] == "target"
            t["model right where the tracer found nothing"] += r["model_right"] and r["tracer"] == "none"
            gate = r["under_todays_cross_check"] == "approval gate"
            t["wrong diagnoses that would reach the approval gate"] += gate and not r["model_right"]
            t["right diagnoses held back because the tracer could not confirm them"] += r["model_right"] and not gate
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if the committed manifest or results would change")
    ap.add_argument("--investigator", action="store_true",
                    help="run the LLM investigator on the warehouse (TRIAGE_MODEL / TRIAGE_BASE_URL / TRIAGE_API_KEY)")
    ap.add_argument("--only", nargs="*", help="incident ids, to rerun some without repeating the slow ones")
    a = ap.parse_args()
    if a.investigator:
        from triage.investigator import OpenAICompatChat
        res = run_investigator(OpenAICompatChat(), a.only, LLM_RESULTS)
        print(json.dumps(res.get("summary", {}), indent=2))
        print(f"{res['incidents']} incidents with {res['model']}; see {LLM_RESULTS}")
        return
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
