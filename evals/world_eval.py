"""Incidents where the answer is not always a SQL fix.

    python -m evals.world_eval                 # write the case list (no model)
    python -m evals.world_eval --check         # CI: the committed case list is what the code produces
    python -m evals.world_eval --model         # run the model on every case (TRIAGE_* variables)
    python -m evals.world_eval --model --history     # the same, with the deploy history as a tool
    python -m evals.world_eval --model --sample 12 --seed 1    # a spread of cases, for a cheap first look

The twelve hand-written breaks all have the same shape: one procedure is wrong, find it, fix it. An
agent can score well on that by always proposing a fix. Three kinds of case are here, mixed:

    sql      a generated edit to one procedure (triage/warehouse/mutants.py). Right answer: a fix that verifies.
    source   every procedure is correct and a source table arrived bad. Right answer: no SQL change,
             and the name of the bad table.
    none     nothing is wrong and the person is mistaken. Right answer: say so, change nothing.

Each kind has its own way of going wrong, and they are counted apart and never averaged: a fix
pushed onto bad data, a real fault closed as "nothing wrong", a correct procedure rewritten.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from triage.warehouse import as_dag
from triage.warehouse.build import Warehouse
from triage.warehouse.probe import observe_sources
from triage.warehouse.world import RIGHT, cases, restore, result, stage

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "evals/data/world_cases.json"
RESULTS = ROOT / "evals/world_results_llm.json"
HISTORY_RESULTS = ROOT / "evals/world_history_results_llm.json"

def run_model(model, out: Path, history: bool = False, only: list[str] | None = None, sample: int | None = None,
              seed: int = 0, max_steps: int = 30) -> dict:
    from triage.warehouse import fix

    wh, dag = Warehouse().build(), as_dag()
    todo = cases(wh)
    if sample:                      # a spread across the three kinds, the same one for the same seed
        rnd, picked = random.Random(seed), []
        for kind, share in (("sql", sample - min(sample // 3, 5) - min(sample // 6, 4)), ("source", min(sample // 3, 5)),
                            ("none", min(sample // 6, 4))):
            pool = [c for c in todo if c["kind"] == kind]
            picked += rnd.sample(pool, min(share, len(pool)))
        todo = [c for c in todo if c in picked]
    if only:
        todo = [c for c in todo if c["id"] in only]
    res = {"model": getattr(model, "model", "?"), "host": getattr(model, "base_url", "").split("//")[-1].split("/")[0] or "?",
           "mode": "open world: a fix, a bad source, or nothing wrong" + ("; with the deploy history as a tool" if history else ""),
           "sample": sample, "seed": seed if sample else None, "rows": []}
    for c in todo:
        print(f"  {c['id']} ...", file=sys.stderr, flush=True)
        inc = stage(wh, c)
        inv, sql, session = fix.draft_checked(wh, dag, inc, model, max_steps=max_steps, history=history, open_world=True)
        kept = False
        if sql is not None:
            kept = fix.apply_and_verify(wh, inv.fix_target, sql, inc["symptom_node"])["kept"]
        got = result(c, sql, kept, session.closed, inv.answer)
        res["rows"].append({
            "id": c["id"], "kind": c["kind"], "split": c["split"], "what": c["what"], "symptom": inc["symptom"],
            "checks_fail_somewhere": any(s["status"] != "ok" for s in inc["node_status"].values()),
            "source_checks_fail": sorted(observe_sources(wh)), "result": got, "right": got == RIGHT[c["kind"]],
            "node": inv.fix_target if sql else None, "sql": sql, "closed": session.closed, "reason": inv.reason,
            "said_instead": inv.answer or None, "tries": session.tries, "harness_refusals": session.refusals,
            "read_the_history": session.asked_changes if history else None, "tool_calls": len(inv.steps),
            "calls": [f"{s['tool']}({', '.join(f'{k}={v}' for k, v in s['args'].items())[:70]})"
                      + ("" if s["ok"] else f" -> {s['result'][:110]}") for s in inv.steps]})
        restore(wh)
        res["summary"] = summarise(res["rows"])
        out.write_text(json.dumps(res, indent=2) + "\n")
        print(f"    {got}", file=sys.stderr, flush=True)
    wh.close()
    return res


def summarise(rows: list[dict]) -> dict:
    """Per kind, the count of each result. Nothing is averaged across kinds: being right about a SQL
    fault and being right that nothing is wrong are different abilities, and so are the ways of failing."""
    out = {}
    for kind in ("sql", "source", "none"):
        mine = [r for r in rows if r["kind"] == kind and r["result"] != "model unavailable"]
        if not mine:
            continue
        counts: dict[str, int] = {}
        for r in mine:
            counts[r["result"]] = counts.get(r["result"], 0) + 1
        out[kind] = {"cases": len(mine), "right": sum(r["right"] for r in mine), "right_answer": RIGHT[kind],
                     "results": dict(sorted(counts.items(), key=lambda kv: -kv[1]))}
        if kind == "sql":
            for split in ("evolve", "held_in", "ood"):
                part = [r for r in mine if r["split"] == split]
                if part:
                    out[kind][split] = f"{sum(r['right'] for r in part)} of {len(part)}"
    unavailable = sum(r["result"] == "model unavailable" for r in rows)
    if unavailable:
        out["model unavailable, not counted"] = unavailable
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--model", action="store_true", help="run the model (TRIAGE_MODEL / TRIAGE_BASE_URL / TRIAGE_API_KEY)")
    ap.add_argument("--history", action="store_true", help="also give the model the deploy history; saved as its own arm")
    ap.add_argument("--sample", type=int, help="run a spread of this many cases instead of all of them")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if a.model:
        from triage.investigator import OpenAICompatChat
        path = HISTORY_RESULTS if a.history else RESULTS
        res = run_model(OpenAICompatChat(), path, a.history, a.only, a.sample, a.seed)
        print(json.dumps(res.get("summary", {}), indent=2))
        print(f"{len(res['rows'])} cases with {res['model']}; see {path}")
        return
    wh = Warehouse().build()
    text = json.dumps(cases(wh), indent=1) + "\n"
    wh.close()
    if a.check:
        if not CASES.exists() or CASES.read_text() != text:
            sys.exit("WORLD CASES DRIFTED: evals/data/world_cases.json would change. Rerun without --check and review the diff.")
        print("world cases match the committed list")
        return
    CASES.write_text(text)
    rows = json.loads(text)
    print({k: sum(r["kind"] == k for r in rows) for k in ("sql", "source", "none")})


if __name__ == "__main__":
    main()
