"""Draft a fix as SQL, apply it, and keep it only if the warehouse is right again.

The health checks can confirm a diagnosis only where a check fails. A break that changes values and
trips nothing cannot be confirmed by looking, so the one thing left is to run the proposed fix and
compare what comes out. This is that loop:

    1. The model reads the warehouse with the same read-only tools as the investigator and submits
       one procedure and its complete replacement SQL.
    2. The SQL is checked before anything runs: one SELECT, nothing that reads the answer key, and it
       has to bind against the tables as they are. A rejection goes back to the model as an error.
    3. The prior definition is recorded, the new one is deployed, and the nightly cycle runs.
    4. `Warehouse.outcome` gives two answers. The fix is kept only if both hold: the report the person
       complained about has the right rows, and so does every other table, with nothing edited
       outside the one procedure. Otherwise the prior definition is put back and the cycle runs again.

Step 4 compares against the reference run, which only means something on a frozen copy of the data
where the last good output is known. That is what this warehouse is, and it is how a fix would be
rehearsed before a person approves it for the real one. It is not a check that can run in production.
"Right" also means the same rows on this data: SQL that happens to agree here and differs elsewhere
would pass.
"""
from __future__ import annotations

import re

import duckdb
from pydantic import BaseModel, Field

from ..dag import PipelineDAG
from ..investigator import TERMINAL, ChatModel, Investigation, ToolSpec, investigate
from .build import Warehouse
from .investigate import tools as read_tools

INSTRUCTIONS = (
    "You are on call for a data warehouse. Every node is a SQL procedure that rebuilds one table from the tables it "
    "reads. A person says a report looks wrong. Find the one procedure whose SQL is wrong and write its corrected SQL. "
    "A table that only carries wrong data from further upstream is not the cause: follow unhealthy nodes upstream "
    "until the nodes above are healthy. The health checks catch some problems and not others, so when every check "
    "passes, read the SQL of the nodes upstream of the report. Change as little as possible and keep every column "
    "the procedure already returns. Call submit_diagnosis once with the node and its complete replacement SELECT. "
    "If you cannot find the cause, do not guess: say so in plain text and the incident goes to a person."
)

ANSWER_KEY = re.compile(r"\b(reference\s*\.|golden\b|run_log\b|procedures\b)", re.I)


class Fix(BaseModel):
    fix_target: str = Field(description="the node whose SQL has to be replaced")
    sql: str = Field(description="the complete replacement SQL for that node: one SELECT statement")
    reason: str = Field(description="what was wrong and what you changed, in one or two sentences")


def validate(wh: Warehouse, dag: PipelineDAG, node: str, sql: str) -> str:
    """-> the SQL as it would be deployed. Raises ValueError with something the model can act on."""
    if node not in dag.nodes:
        raise ValueError(f"'{node}' is not in this pipeline. Call pipeline to see the nodes.")
    sql = sql.strip().rstrip(";").strip()
    if ";" in sql or not re.match(r"(?is)^\s*(select|with)\b", sql):
        raise ValueError("the fix has to be one SELECT statement, with nothing before or after it")
    if ANSWER_KEY.search(sql):
        raise ValueError("a procedure cannot read the reference tables, the run log or the procedure catalogue")
    try:
        wh.db.execute(f"explain {sql}")
    except duckdb.Error as e:
        raise ValueError(f"this SQL does not run: {str(e).splitlines()[0][:240]}") from None
    return sql


def tools(wh: Warehouse, dag: PipelineDAG, incident: dict) -> list[ToolSpec]:
    def submit(fix_target: str, sql: str, reason: str = "") -> str:
        validate(wh, dag, fix_target, sql)
        return f"recorded: replace the SQL of {fix_target}"

    read = [t for t in read_tools(wh, dag, incident) if t.name != TERMINAL]
    return read + [ToolSpec(TERMINAL, "Submit the fix: the node and its complete replacement SQL. Call this once.",
                            Fix, submit)]


def draft(wh: Warehouse, dag: PipelineDAG, incident: dict, model: ChatModel, max_steps: int = 20) -> tuple[Investigation, str | None]:
    """-> (what the model did, the SQL it submitted or None). Nothing is changed by this."""
    task = (f"Incident {incident.get('id')}: {incident.get('symptom')}. "
            "Find the procedure whose SQL is wrong and submit its corrected SQL.")
    inv = investigate(incident, dag, None, model, max_steps=max_steps, specs=tools(wh, dag, incident),
                      instructions=INSTRUCTIONS, task_text=task)
    if not inv.submitted:
        return inv, None
    return inv, validate(wh, dag, inv.fix_target, inv.steps[-1]["args"]["sql"])


def apply_and_verify(wh: Warehouse, node: str, sql: str, symptom_node: str) -> dict:
    """Deploy the fix, run the cycle, and keep it only if both answers hold. Otherwise put the prior
    definition back, so a rejected fix leaves the warehouse exactly as it found it."""
    prior = wh.sql(node)
    wh.set_sql(node, sql)
    wh.run_cycle()
    out = wh.outcome(symptom_node, allowed=(node,))
    out["kept"] = out["functional"] and out["integrity"]
    if not out["kept"]:
        wh.set_sql(node, prior)
        wh.run_cycle()
    out["rolled_back"] = not out["kept"]
    return out
