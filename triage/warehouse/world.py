"""The cases an on-call agent actually meets, and a way to deal them out one night at a time.

Three kinds, mixed (see evals/world_eval.py for how they are scored):

    sql      a generated edit to one procedure. Right answer: a fix that verifies.
    source   every procedure is correct and a source table arrived bad. Right answer: name the table.
    none     nothing is wrong and the person is mistaken. Right answer: say so, change nothing.

`schedule` turns them into nights. Most nights something happens, some nights nothing does, and
whoever is on call is not told which kind tonight is.
"""
from __future__ import annotations

import random

from .build import Warehouse
from .mutants import generate
from .pipeline import BY_ID, REPORTS
from .probe import observe
from .sourcefaults import SOURCE_FAULTS

MISTAKEN = (("rpt_revenue_dashboard", "revenue in rpt_revenue_dashboard looks too low this week"),
            ("rpt_regional_summary", "margin_rate in rpt_regional_summary looks wrong"),
            ("rpt_revenue_dashboard", "trailing_7d_revenue in rpt_revenue_dashboard looks wrong"),
            ("rpt_regional_summary", "revenue_share in rpt_regional_summary does not look right"))

RIGHT = {"sql": "fixed", "source": "named the bad source", "none": "said nothing was wrong"}


def cases(wh: Warehouse) -> list[dict]:
    """Every case, in a fixed order, with what makes it one. No model is involved in this."""
    out = [{"id": m["id"], "kind": "sql", "split": m["split"], "node": m["node"], "what": m["what"], "sql": m["sql"],
            "silent": m["silent"]} for m in generate(wh)]
    out += [{"id": f.id, "kind": "source", "split": "held_in", "table": f.table, "what": f.what,
             "source_checks_see_it": f.visible} for f in SOURCE_FAULTS]
    out += [{"id": f"none.mistaken_{i + 1}", "kind": "none", "split": "held_in", "report": r, "said": said,
             "what": "nothing is wrong"} for i, (r, said) in enumerate(MISTAKEN)]
    return out


def as_seen(incident: dict, n: int) -> dict:
    """The incident as whoever triages it may see it. The case id says what kind of fault it is and
    often which table, and the label holds the answer, so neither goes along: the id is replaced by a
    plain number and only the complaint and the health checks are kept."""
    return {"id": f"INC-{n:03d}", "symptom_node": incident["symptom_node"], "symptom": incident["symptom"],
            "node_status": incident["node_status"]}


def stage(wh: Warehouse, c: dict) -> dict:
    """Put the warehouse into the state of this case and return the incident as the agent gets it."""
    wh.mark_good()
    if c["kind"] == "sql":
        wh.set_sql(c["node"], c["sql"])
    elif c["kind"] == "source":
        f = next(x for x in SOURCE_FAULTS if x.id == c["id"])
        wh.corrupt_source(f.table, f.statements)
    wh.run_cycle()
    wrong = wh.affected()
    symptom = c.get("report") or next(r for r in REPORTS if r in wrong)
    return {"id": c["id"], "symptom_node": symptom, "symptom": c.get("said") or wh.complaint(symptom),
            "node_status": observe(wh)}


def restore(wh: Warehouse) -> None:
    for n in wh.changed_procedures():
        wh.set_sql(n, BY_ID[n].sql, record=False)
    wh.reload_sources()
    wh.run_cycle()
    if wh.affected():
        raise RuntimeError(f"restoring did not bring back {wh.affected()}")


def result(c: dict, sql: str | None, kept: bool, closed: dict | None, answer: str) -> str:
    if sql is not None and not kept:
        return "wrong fix, rolled back"
    if sql is not None:
        return {"sql": "fixed", "source": "worked around bad data in SQL", "none": "rewrote SQL that was right"}[c["kind"]]
    if closed and closed["kind"] == "nothing_wrong":
        return "said nothing was wrong"
    if closed:
        if c["kind"] == "source":
            return "named the bad source" if closed["source_table"] == c["table"] else "blamed the wrong source"
        return "blamed a source"
    return "model unavailable" if answer.startswith("model unavailable") else "handed to a person"


def schedule(all_cases: list[dict], nights: int, seed: int = 0, quiet: float = 0.3) -> list[dict | None]:
    """-> one entry per night: a case, or None for a night where every table rebuilt and nobody complained.
    Of the nights where something happens, about half are SQL faults, a third bad source data, the rest
    mistaken complaints. The same seed gives the same nights."""
    rnd = random.Random(seed)
    pools = {k: [c for c in all_cases if c["kind"] == k] for k in ("sql", "source", "none")}
    out: list[dict | None] = []
    for _ in range(nights):
        if rnd.random() < quiet:
            out.append(None)
            continue
        kind = rnd.choices(("sql", "source", "none"), weights=(5, 3, 2))[0]
        out.append(rnd.choice(pools[kind]))
    return out
