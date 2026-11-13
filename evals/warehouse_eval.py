"""Break the warehouse twelve ways and see what the rule-based tracer makes of each.

    python -m evals.warehouse_eval            # rebuild, write the manifest and the results
    python -m evals.warehouse_eval --check    # rebuild and fail if either file would change (CI)
    python -m evals.warehouse_eval --investigator
                                              # the LLM investigator on the same incidents, with read-only
                                              # warehouse tools; writes evals/warehouse_results_llm.json
    python -m evals.warehouse_eval --fixer    # the model drafts SQL for each break; a fix is kept only if
                                              # the warehouse is right again; writes warehouse_fix_results_llm.json
    python -m evals.warehouse_eval --fixer --checker
                                              # the same, with the checker and a rehearsal as tools the model
                                              # has to use; writes warehouse_fix_checked_results_llm.json

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

from triage import evidence
from triage.dag import trace
from triage.warehouse import BREAKS, Warehouse, as_dag, observe
from triage.warehouse.breaks import split_of
from triage.warehouse.pipeline import BY_ID, REPORTS

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "evals/data/warehouse_incidents.json"
RESULTS = ROOT / "evals/warehouse_results.json"
LLM_RESULTS = ROOT / "evals/warehouse_results_llm.json"
FIX_RESULTS = ROOT / "evals/warehouse_fix_results_llm.json"
CHECKED_FIX_RESULTS = ROOT / "evals/warehouse_fix_checked_results_llm.json"
HISTORY_FIX_RESULTS = ROOT / "evals/warehouse_fix_history_results_llm.json"


def each_break(wh: Warehouse):
    """Yield every break as an incident while the warehouse is still in the broken state, then put the
    procedure back and confirm every table is the reference again. The label is kept apart from what
    the triage reads."""
    for b in BREAKS:
        wh.mark_good()                  # the deploy history a tool may read starts here
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


def judge(target: str | None, cause: str | None, edited: str) -> str:
    """One rule for the tracer and the model alike: target, cause_only, none or wrong."""
    if target is None:
        return "none"
    if target == edited:
        return "target"
    return "cause_only" if cause == edited else "wrong"


def verdict(inc: dict, dag) -> str:
    if inc["symptom_node"] is None:
        return "no_symptom"
    res = trace(dag, inc["node_status"], inc["symptom_node"])
    return judge(res.fix_target, res.upstream_cause, inc["label"]["fix_targets"][0])


def run() -> tuple[list[dict], dict]:
    wh = Warehouse().build()
    dag = as_dag()
    incs = incidents(wh)
    wh.close()
    rows = [{"id": i["id"], "split": i["label"]["split"], "edited": i["label"]["fix_targets"][0],
             "symptom_node": i["symptom_node"], "silent": i["label"]["silent"],
             "probe_sees_it_at_the_source": i["label"]["probe_sees_it_at_the_source"], "tracer": verdict(i, dag),
             "first_failure": judge(evidence.origin(dag, i["node_status"], i["symptom_node"]), None,
                                    i["label"]["fix_targets"][0])}
            for i in incs]
    tally: dict[str, dict[str, int]] = {}
    by_rule: dict[str, dict[str, int]] = {}
    for r in rows:
        for key in (r["split"], "all"):
            t = tally.setdefault(key, {"incidents": 0, "target": 0, "cause_only": 0, "none": 0, "wrong": 0})
            t["incidents"] += 1
            t[r["tracer"]] = t.get(r["tracer"], 0) + 1
            ff = by_rule.setdefault(key, {"incidents": 0, "target": 0, "none": 0, "wrong": 0})
            ff["incidents"] += 1
            ff[r["first_failure"]] += 1
    manifest = [{"id": i["id"], "symptom_node": i["symptom_node"], "symptom": i["symptom"],
                 "node_status": {n: s["status"] for n, s in i["node_status"].items()}, **i["label"]} for i in incs]
    results = {"scale_factor": 0.01, "incidents": len(rows), "silent": sum(r["silent"] for r in rows),
               "invisible_to_every_probe_check": sum(not r["probe_sees_it_at_the_source"] for r in rows),
               "mode": "rule-based tracer on node health read from the warehouse",
               "by_split": tally,
               "first_failure_rule": "name the node where the health checks first fail, if there is exactly one",
               "first_failure_by_split": by_rule, "rows": rows}
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
            "model_right": right, "model": judge(inv.fix_target if inv.submitted else None, inv.cause, edited),
            "reason": inv.reason, "said_instead": inv.answer or None,
            "tracer": verdict(inc, dag), "tracer_target": tr.fix_target,
            "under_todays_cross_check": "approval gate" if agrees else "a person",
            **_evidence(dag, inc, inv.fix_target if inv.submitted else None, inv.cause),
            "tool_calls": len(inv.steps),
            "calls": [f"{s['tool']}({', '.join(f'{k}={v}' for k, v in s['args'].items())[:70]})" + ("" if s["ok"] else " -> error")
                      for s in inv.steps],
        })
        res["incidents"] = len(rows)
        res["summary"] = _summarise(rows)
        if out:
            out.write_text(json.dumps(res, indent=2) + "\n")       # after every incident: a rate limit loses one, not all
        print(f"    model: {inv.fix_target if inv.submitted else 'no diagnosis'} ({rows[-1]['model']}), "
              f"tracer: {rows[-1]['tracer']}",
              file=sys.stderr, flush=True)
    wh.close()
    return res


def _evidence(dag, inc: dict, target: str | None, cause: str | None) -> dict:
    ev = evidence.check(dag, inc["node_status"], inc["symptom_node"], target, cause)
    return {"evidence": ev.verdict, "evidence_why": ev.why,
            "under_the_evidence_check": "approval gate" if ev.verdict == evidence.SUPPORTED else "a person",
            "first_failure": evidence.origin(dag, inc["node_status"], inc["symptom_node"])}


LABELS = {"target": "named the edited procedure", "cause_only": "named the procedure that errored, edited one as cause",
          "none": "gave no diagnosis", "wrong": "named something else"}


def _summarise(rows: list[dict]) -> dict:
    """The model and the tracer are counted by the same four outcomes. A diagnosis is 'sound' when it
    points at the edited procedure, as the target or as the stated cause of the one that errored."""
    out: dict[str, dict] = {}
    for r in rows:
        for key in (r["split"], "all"):
            t = out.setdefault(key, {"incidents": 0, **{f"model {v}": 0 for v in LABELS.values()},
                                     **{f"tracer {v}": 0 for v in LABELS.values()},
                                     "model named the edited procedure where the tracer gave no diagnosis": 0,
                                     "either one named the edited procedure": 0,
                                     "model named something else and it would reach the approval gate": 0,
                                     "model named the edited procedure and it is held back for a person": 0,
                                     "evidence check: supported": 0, "evidence check: contradicted": 0,
                                     "evidence check: unverified": 0,
                                     "evidence check: something else reaches the approval gate": 0,
                                     "evidence check: edited procedure named and held back for a person": 0,
                                     "first-failure rule alone named the edited procedure": 0})
            t["incidents"] += 1
            t[f"model {LABELS[r['model']]}"] += 1
            t[f"tracer {LABELS[r['tracer']]}"] += 1
            gate = r["under_todays_cross_check"] == "approval gate"
            t["model named the edited procedure where the tracer gave no diagnosis"] += r["model"] == "target" and r["tracer"] == "none"
            t["either one named the edited procedure"] += "target" in (r["model"], r["tracer"])
            t["model named something else and it would reach the approval gate"] += gate and r["model"] == "wrong"
            t["model named the edited procedure and it is held back for a person"] += r["model"] == "target" and not gate
            ev_gate = r["under_the_evidence_check"] == "approval gate"
            t[f"evidence check: {r['evidence']}"] += 1
            t["evidence check: something else reaches the approval gate"] += ev_gate and r["model"] == "wrong"
            t["evidence check: edited procedure named and held back for a person"] += r["model"] == "target" and not ev_gate
            t["first-failure rule alone named the edited procedure"] += r["first_failure"] == r["edited"]
    return out


def rescore(path: Path) -> dict:
    """Recount a finished model run under the current rules without calling the model again. Only the
    derived fields change: what the model answered and what it called are left as they were recorded."""
    res = json.loads(path.read_text())
    wh, dag = Warehouse().build(), as_dag()
    state = {i["id"]: i for i in each_break(wh)}             # node health is rebuilt, the same every time
    wh.close()
    for r in res["rows"]:
        target = r["model_target"] if r["submitted"] else None
        r["model"] = judge(target, r["model_cause"], r["edited"])
        r.update(_evidence(dag, state[r["id"]], target, r["model_cause"]))
    res["summary"] = _summarise(res["rows"])
    path.write_text(json.dumps(res, indent=2) + "\n")
    return res


def run_fixer(model, only: list[str] | None = None, out: Path | None = None, max_steps: int | None = None,
              checker: bool = False, history: bool = False) -> dict:
    """The model drafts SQL for each break, and the fix is scored by running it.

    A fix counts only when the report is right again and so is every other table, with one procedure
    changed. Anything else is rolled back. The model never sees the reference it is scored against.
    """
    from triage.warehouse import fix

    wh, dag, rows = Warehouse().build(), as_dag(), []
    res = {"model": getattr(model, "model", "?"),
           "host": getattr(model, "base_url", "").split("//")[-1].split("/")[0] or "?",   # the same weights behave a little differently per host
           "mode": ("model drafts SQL with the checker as tools it must use; only a rehearsed fix can be submitted"
                    + ("; it can also read what was deployed since the last good run" if history else "")
                    if checker else "model drafts SQL; kept only if the warehouse is right again"),
           "incidents": 0, "rows": rows}
    for inc in each_break(wh):
        if (only and inc["id"] not in only) or inc["symptom_node"] is None:
            continue
        print(f"  {inc['id']} ...", file=sys.stderr, flush=True)
        session = None
        if checker:
            inv, sql, session = fix.draft_checked(wh, dag, inc, model, max_steps=max_steps or 26, history=history)
        else:
            inv, sql = fix.draft(wh, dag, inc, model, max_steps=max_steps or 20)
        edited = inc["label"]["fix_targets"][0]
        row = {"id": inc["id"], "split": inc["label"]["split"], "edited": edited, "symptom": inc["symptom"],
               "checks_see_it": inc["label"]["probe_sees_it_at_the_source"],
               "submitted": sql is not None, "node": inv.fix_target if sql else None, "sql": sql,
               "reason": inv.reason, "said_instead": inv.answer or None,
               "evidence": evidence.check(dag, inc["node_status"], inc["symptom_node"], inv.fix_target if sql else None).verdict,
               "functional": False, "integrity": False, "kept": False, "still_wrong": None, "changed_outside_approval": None,
               "tool_calls": len(inv.steps),
               "calls": [f"{s['tool']}({', '.join(f'{k}={v}' for k, v in s['args'].items())[:70]})"
                         + ("" if s["ok"] else f" -> {s['result'][:110]}") for s in inv.steps]}
        if session is not None:
            row["checker"] = {"asked where checks fail": session.asked_where, "nodes checked": session.checked,
                              "tries": session.tries, "verified on try": session.verified_on_try,
                              "taken from the rehearsal": session.taken_from_rehearsal,
                              "harness refusals": session.refusals}
        if sql is not None:
            scored = fix.apply_and_verify(wh, inv.fix_target, sql, inc["symptom_node"])
            row.update({k: scored[k] for k in ("functional", "integrity", "kept", "still_wrong", "changed_outside_approval")})
        row["result"] = _fix_result(row)
        rows.append(row)
        if only and out and out.exists():                       # a partial rerun replaces only the rows it reran
            order = [b.id for b in BREAKS]
            kept = [r for r in json.loads(out.read_text())["rows"] if r["id"] not in {x["id"] for x in rows}]
            res["rows"] = sorted(kept + rows, key=lambda r: order.index(r["id"]))
        res["incidents"], res["summary"] = len(res["rows"]), _summarise_fixes(res["rows"])
        if out:
            out.write_text(json.dumps(res, indent=2) + "\n")
        print(f"    {row['node'] or 'no fix'}: {row['result']}", file=sys.stderr, flush=True)
    wh.close()
    return res


FIX_OUTCOMES = ("fixed", "report came back, warehouse still wrong", "did not fix it", "no fix submitted",
                "verified in rehearsal, never submitted", "model unavailable")


def _fix_result(row: dict) -> str:
    """`model unavailable` is the provider cutting the run off, not the model failing: those incidents
    are counted apart and are not evidence about the model. `verified in rehearsal, never submitted`
    only occurs in runs saved before the harness took a verified fix on the model's behalf."""
    if row["kept"]:
        return "fixed"
    if row["submitted"]:
        return "report came back, warehouse still wrong" if row["functional"] else "did not fix it"
    if (row.get("said_instead") or "").startswith("model unavailable"):
        return "model unavailable"
    if (row.get("checker") or {}).get("verified on try"):
        return "verified in rehearsal, never submitted"
    return "no fix submitted"


def rescore_fixes(path: Path) -> dict:
    """Relabel a saved fix run under the current rules. Nothing the model did is changed."""
    res = json.loads(path.read_text())
    for r in res["rows"]:
        r["result"] = _fix_result(r)
    res["summary"] = _summarise_fixes(res["rows"])
    path.write_text(json.dumps(res, indent=2) + "\n")
    return res


def _summarise_fixes(rows: list[dict]) -> dict:
    """Functional and integrity are counted apart, and so are the breaks no health check can see."""
    out: dict[str, dict] = {}
    for r in rows:
        for key in (r["split"], "checks see it" if r["checks_see_it"] else "no check sees it", "all"):
            t = out.setdefault(key, {"incidents": 0, **{k: 0 for k in FIX_OUTCOMES}, "functional": 0, "integrity": 0})
            t["incidents"] += 1
            t[r["result"]] += 1
            t["functional"] += r["functional"]
            t["integrity"] += r["integrity"]
            if "checker" in r:
                c = r["checker"]
                for k in ("asked where checks fail", "fixed on the first try", "fixed after a failed try", "tries",
                          "harness refusals", "incidents where the harness refused something"):
                    t.setdefault(k, 0)
                t.setdefault("taken from the rehearsal by the harness", 0)
                t["taken from the rehearsal by the harness"] += bool(c.get("taken from the rehearsal"))
                t["asked where checks fail"] += c["asked where checks fail"]
                t["fixed on the first try"] += r["kept"] and c["verified on try"] == 1
                t["fixed after a failed try"] += r["kept"] and (c["verified on try"] or 0) > 1
                t["tries"] += c["tries"]
                t["harness refusals"] += len(c["harness refusals"])
                t["incidents where the harness refused something"] += bool(c["harness refusals"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if the committed manifest or results would change")
    ap.add_argument("--investigator", action="store_true",
                    help="run the LLM investigator on the warehouse (TRIAGE_MODEL / TRIAGE_BASE_URL / TRIAGE_API_KEY)")
    ap.add_argument("--fixer", action="store_true",
                    help="have the model draft SQL for each break and score it by running it (same TRIAGE_* variables)")
    ap.add_argument("--checker", action="store_true",
                    help="with --fixer: give the model the checker and the rehearsal as tools it has to use")
    ap.add_argument("--history", action="store_true",
                    help="with --fixer --checker: also give the model the deploy history. Saved apart, as its own arm")
    ap.add_argument("--only", nargs="*", help="incident ids, to rerun some without repeating the slow ones")
    ap.add_argument("--rescore", action="store_true", help="recount the saved model run under the current rules")
    a = ap.parse_args()
    if a.rescore:
        print(json.dumps(rescore(LLM_RESULTS)["summary"]["all"], indent=2))
        for path in (FIX_RESULTS, CHECKED_FIX_RESULTS):
            if path.exists():
                print(path.name, json.dumps(rescore_fixes(path)["summary"]["all"], indent=2))
        return
    if a.fixer:
        from triage.investigator import OpenAICompatChat
        if a.history and not a.checker:
            sys.exit("--history is an arm of the checker run: use --fixer --checker --history")
        path = HISTORY_FIX_RESULTS if a.history else CHECKED_FIX_RESULTS if a.checker else FIX_RESULTS
        res = run_fixer(OpenAICompatChat(), a.only, path, checker=a.checker, history=a.history)
        print(json.dumps(res.get("summary", {}).get("all", {}), indent=2))
        print(f"{res['incidents']} incidents with {res['model']}; see {path}")
        return
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
