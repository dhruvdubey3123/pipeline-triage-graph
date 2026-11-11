"""The investigator, pointed at the real warehouse.

On the synthetic incidents the model reads node health that arrived with the incident. Here it gets
what a person on call has: the pipeline, the health checks, what each table looked like on this run
against the last good one, and the SQL of every procedure. It cannot run SQL of its own and it cannot
change anything.

Two things are deliberately not given to it. There is no "what changed recently" tool, because the
procedure with a new definition is the answer and asking for it is not a diagnosis. And there is no
list of failure categories to pick from, because naming the kinds of break is a hint about which ones
exist, including the two families held out to see what happens on a kind it was never tuned on.

The health checks miss some breaks by construction (triage/warehouse/probe.py says which), so this is
the first place the model can be right where the rule-based tracer has nothing to say. If it cannot
find the cause it is told to say so in plain text, which ends the investigation without a diagnosis
and sends the incident to a person.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..dag import PipelineDAG
from ..investigator import TERMINAL, ChatModel, Investigation, NodeArg, ToolSpec, investigate
from ..investigator import tools as shared_tools
from ..knowledge import CaseLibrary
from .build import Warehouse

INSTRUCTIONS = (
    "You are on call for a data warehouse. Every node is a SQL procedure that rebuilds one table from the tables it "
    "reads. A person says a report looks wrong. Find the procedure whose SQL is wrong. A table that only carries "
    "wrong data from further upstream is not the cause. The health checks catch some problems and not others: a node "
    "reported ok can still be the cause, so when the checks do not explain the complaint, read the SQL of the nodes "
    "upstream of the report. When you have found it, call submit_diagnosis once and stop. If you cannot find it, do "
    "not guess: say so in plain text and the incident goes to a person. You cannot change anything from here."
)


class Diagnosis(BaseModel):
    fix_target: str = Field(description="the node whose SQL has to be fixed")
    cause: str = Field("", description="the upstream node that caused it, if a different one")
    reason: str = Field(description="what is wrong in that node's SQL, in one or two sentences")


def _pct(x: float) -> str:
    return f"{x:.1%}"


def tools(wh: Warehouse, dag: PipelineDAG, incident: dict) -> list[ToolSpec]:
    shared = {t.name: t for t in shared_tools(dag, CaseLibrary([]), incident)}

    def known(node: str) -> None:
        if node not in dag.nodes:
            raise ValueError(f"'{node}' is not in this pipeline. Call pipeline to see the nodes.")

    def profile(node: str) -> str:
        known(node)
        run, before = wh.last_run(node), wh.last_run(node, 1)["stats"]
        if run["status"] != "success":
            return f"{node}: last run {run['status']}: {run['error']}"
        now = run["stats"]
        nulls = [f"{c} {_pct(r)} (was {_pct(before['nulls'].get(c, 0.0))})" for c, r in now["nulls"].items()
                 if r or before["nulls"].get(c)]
        parts = [f"{node}: {now['rows']} rows (last good run {before['rows']})",
                 f"rows sharing a key: {now['duplicate_keys']}",
                 "nulls: " + (", ".join(nulls) or "none"),
                 "columns: " + ", ".join(f"{c} {t}" for c, t in now["schema"].items())]
        if now["latest"]:
            parts.insert(2, f"latest date {now['latest']} (was {before['latest']})")
        return "; ".join(parts)

    def read_sql(node: str) -> str:
        known(node)
        return wh.sql(node)

    def submit_diagnosis(fix_target: str, cause: str = "", reason: str = "") -> str:
        known(fix_target)
        if cause:
            known(cause)
        return f"recorded: fix {fix_target}"

    return [
        shared["pipeline"], shared["status"], shared["upstream"],
        ToolSpec("profile", "One table on this run against the last good run: rows, duplicate keys, nulls, "
                 "latest date, column types.", NodeArg, profile),
        ToolSpec("read_sql", "The SQL of one node's procedure, as it is deployed now.", NodeArg, read_sql),
        ToolSpec(TERMINAL, "Submit the diagnosis. Call this once, when you have found the wrong SQL.", Diagnosis,
                 submit_diagnosis),
    ]


def task(incident: dict) -> str:
    return f"Incident {incident.get('id')}: {incident.get('symptom')}. Find the procedure whose SQL is wrong."


def investigate_warehouse(wh: Warehouse, dag: PipelineDAG, incident: dict, model: ChatModel,
                          max_steps: int = 18) -> Investigation:
    """The warehouse has to be in the state the incident describes while this runs: the tools read it live."""
    return investigate(incident, dag, CaseLibrary([]), model, max_steps=max_steps,
                       specs=tools(wh, dag, incident), instructions=INSTRUCTIONS, task_text=task(incident))
