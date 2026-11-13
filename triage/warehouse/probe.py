"""Node health, worked out from the warehouse instead of handed over with the incident.

The synthetic incidents arrive with every node already labelled: failed, stale, ok. That is the
answer with the working removed. Here the label is the result of checks a person on call could run
in production, none of which need to know what the right numbers are:

    failed    the last run raised an error
    stale     the last run was skipped because something above it did not succeed
    changed   the columns or their types are not what they were on the last good run
    deviates  it ran, the shape is the same, and something that should hold does not: a key that
              is no longer unique, a row count that no longer matches the table it is built from or
              moved by more than 1%, more nulls than before, or a latest date that went backwards
    ok        none of the above

That last line is the honest part. A break that changes values and nothing else, such as a total that
sums six days in place of seven, passes every one of these and reports ok. The reference checksum
would catch it, and the reference checksum is not available to whoever is doing the triage.
"""
from __future__ import annotations

from .build import RAW, Warehouse
from .pipeline import NODES

ROW_TOLERANCE = 0.01
NULL_TOLERANCE = 0.005


def _checks(wh: Warehouse, node, now: dict, before: dict) -> list[str]:
    out = []
    if now["duplicate_keys"]:
        out.append(f"{now['duplicate_keys']} rows share a key ({', '.join(node.grain)}) that was unique")
    if node.same_rows_as:
        parent = wh.last_run(node.same_rows_as)["stats"]
        if parent and now["rows"] != parent["rows"]:
            out.append(f"{now['rows']} rows, but {node.same_rows_as} has {parent['rows']}")
    elif before["rows"] and abs(now["rows"] - before["rows"]) / before["rows"] > ROW_TOLERANCE:
        out.append(f"{now['rows']} rows, {before['rows']} on the last good run")
    for col, rate in now["nulls"].items():
        if rate - before["nulls"].get(col, 0.0) > NULL_TOLERANCE:
            out.append(f"{col} is null in {rate:.1%} of rows, {before['nulls'].get(col, 0.0):.1%} before")
    if now["latest"] and before["latest"] and now["latest"] < before["latest"]:
        out.append(f"latest {node.date_col} is {now['latest']}, was {before['latest']}")
    return out


def observe(wh: Warehouse, reference_cycle: int = 1) -> dict[str, dict]:
    """-> node_status in the form the tracer and the investigator already read."""
    status: dict[str, dict] = {}
    for n in NODES:
        run = wh.last_run(n.id)
        if run["status"] == "failed":
            status[n.id] = {"status": "failed", "error": run["error"]}
            continue
        if run["status"] == "skipped":
            status[n.id] = {"status": "stale", "note": run["error"]}
            continue
        now, before = run["stats"], wh.last_run(n.id, reference_cycle)["stats"]
        if now["schema"] != before["schema"]:
            gone = [c for c in before["schema"] if c not in now["schema"]]
            retyped = [f"{c} {before['schema'][c]} to {t}" for c, t in now["schema"].items()
                       if c in before["schema"] and before["schema"][c] != t]
            added = [c for c in now["schema"] if c not in before["schema"]]
            changes = ([f"column {c} dropped" for c in gone] + [f"type of {c} changed" for c in retyped]
                       + [f"column {c} added" for c in added])
            status[n.id] = {"status": "changed", "changes": changes}
            continue
        problems = _checks(wh, n, now, before)
        status[n.id] = {"status": "deviates", "note": "; ".join(problems)} if problems else {"status": "ok"}
    return status


def observe_sources(wh: Warehouse, reference_cycle: int = 1) -> dict[str, list[str]]:
    """-> {source table: what is not as it was on the last good run}. Only tables with something to say.

    The same kind of check as for a node, on the tables the pipeline reads and does not build. A
    source only ever gains rows, so losing any is worth saying; a gain has to be more than 1%. Like
    the node checks, these look at shape and not at values, so a column that arrives in the wrong
    unit says nothing here."""
    out: dict[str, list[str]] = {}
    for t in RAW:
        now, before = wh.source_run(t), wh.source_run(t, reference_cycle)
        if not now or not before:
            continue
        said = []
        if now["rows"] < before["rows"]:
            said.append(f"{now['rows']} rows, {before['rows']} on the last good run")
        elif before["rows"] and (now["rows"] - before["rows"]) / before["rows"] > ROW_TOLERANCE:
            said.append(f"{now['rows']} rows, {before['rows']} on the last good run")
        if now["duplicate_keys"] > before["duplicate_keys"]:
            said.append(f"{now['duplicate_keys']} rows share a key that was unique")
        for col, rate in now["nulls"].items():
            if rate - before["nulls"].get(col, 0.0) > NULL_TOLERANCE:
                said.append(f"{col} is null in {rate:.1%} of rows, {before['nulls'].get(col, 0.0):.1%} before")
        if now["latest"] and before["latest"] and now["latest"] < before["latest"]:
            said.append(f"latest date is {now['latest']}, was {before['latest']}")
        if said:
            out[t] = said
    return out


class WarehouseProbe:
    """triage.adapters.DataProbe, answered by the warehouse rather than by notes in the incident."""

    def __init__(self, wh: Warehouse):
        self.wh = wh

    def check(self, node_id: str, incident: dict) -> str | None:
        s = observe(self.wh).get(node_id)
        if s is None or s["status"] == "ok":
            return None
        return s.get("error") or s.get("note") or "; ".join(s.get("changes", [])) or s["status"]
