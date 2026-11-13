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

import difflib
import re
from dataclasses import dataclass, field

import duckdb
from typing import Literal

from pydantic import BaseModel, Field

from .. import evidence
from ..dag import PipelineDAG
from ..investigator import TERMINAL, ChatModel, Investigation, NodeArg, ToolSpec, investigate
from .build import RAW as RAW_TABLES
from .build import Warehouse
from .investigate import tools as read_tools
from .probe import observe_sources

INSTRUCTIONS = (
    "You are on call for a data warehouse. Every node is a SQL procedure that rebuilds one table from the tables it "
    "reads. A person says a report looks wrong. Find the one procedure whose SQL is wrong and write its corrected SQL. "
    "A table that only carries wrong data from further upstream is not the cause: follow unhealthy nodes upstream "
    "until the nodes above are healthy. The health checks catch some problems and not others, so when every check "
    "passes, read the SQL of the nodes upstream of the report. Change as little as possible and keep every column "
    "the procedure already returns. Call submit_diagnosis once with the node and its complete replacement SELECT. "
    "If you cannot find the cause, do not guess: say so in plain text and the incident goes to a person."
)

ANSWER_KEY = re.compile(r"\b(reference\s*\.|clean\s*\.|golden\b|run_log\b|source_log\b|deploys\b|procedures\b)", re.I)


class Fix(BaseModel):
    fix_target: str = Field(description="the node whose SQL has to be replaced")
    sql: str = Field(description="the complete replacement SQL for that node: one SELECT statement")
    reason: str = Field(description="what was wrong and what you changed, in one or two sentences")


class Attempt(BaseModel):
    fix_target: str = Field(description="the node whose SQL to replace")
    sql: str = Field(description="the complete replacement SQL for that node: one SELECT statement")
    reason: str = Field("", description="one sentence: what was wrong and what you changed")


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


# ---------------------------------------------------------------- the same loop, with the checker in the model's hands

MAX_TRIES = 3

CHECKED_INSTRUCTIONS = (
    "You are on call for a data warehouse. Every node is a SQL procedure that rebuilds one table from the tables it "
    "reads. A person says a report looks wrong. Find the one procedure whose SQL is wrong, write its corrected SQL, "
    "and prove the fix works before you submit it. Work in this order. First call where_checks_fail. If it names a "
    "node, start there. If it says nowhere, the health checks cannot see this problem: read the SQL of the nodes "
    "upstream of the report and find it yourself. Call check_node on the node you blame; if it is contradicted, the "
    "reason tells you where to look instead. Then call try_fix with that node, its complete replacement SELECT and one sentence saying what was wrong. It "
    f"runs your SQL on a rehearsal copy and tells you whether the warehouse came out right. You have {MAX_TRIES} tries. "
    "Change as little as possible and keep every column the procedure already returns. Call submit_diagnosis only "
    "with a fix that try_fix verified. If you run out of tries, do not submit: say so in plain text and the incident "
    "goes to a person."
)


@dataclass
class Session:
    """What the model did with the checker, and every time the harness had to hold the line for it."""
    asked_where: bool = False
    checked: dict = field(default_factory=dict)        # node -> verdict
    tries: int = 0
    verified: list = field(default_factory=list)       # (node, sql) pairs a rehearsal passed, in order
    reasons: dict = field(default_factory=dict)        # (node, sql) -> the reason given with that rehearsal
    taken_from_rehearsal: bool = False                 # the model stopped without submitting; the harness took the verified fix
    verified_on_try: int | None = None
    refusals: list = field(default_factory=list)       # reasons the harness refused a call
    asked_sources: bool = False
    asked_changes: bool = False
    closed: dict | None = None                         # ended without a fix: {"kind", "source_table", "reason"}


def rehearse(wh: Warehouse, node: str, sql: str, symptom_node: str) -> dict:
    """Run a fix and put everything back, whatever the result. Applying for real is a separate step
    that happens after a person approves."""
    prior = wh.sql(node)
    wh.set_sql(node, sql, record=False)
    wh.run_cycle()
    out = wh.outcome(symptom_node, allowed=(node,))
    wh.set_sql(node, prior, record=False)
    wh.run_cycle()
    return out


HISTORY_NOTE = (" You also have recent_changes: the SQL that was deployed since the last good run, as a diff. A recent "
                "change is the first place to look, but it is not proof: check it and rehearse a fix like any other.")

OPEN_INSTRUCTIONS = (
    "You are on call for a data warehouse. Every node is a SQL procedure that rebuilds one table from the tables it "
    "reads, and the first nodes read source tables that are loaded from outside. A person says a report looks wrong. "
    "There are three things that can be true, and you have to find out which. One: a procedure's SQL is wrong. Then "
    "write its corrected SQL and prove it. Two: the SQL is all correct and a source table arrived bad (rows missing, "
    "loaded twice, a column not filled, a value in the wrong unit). No SQL change is the right fix for that: the "
    "source's owner has to reload it, and working around bad data in SQL hides it. Three: nothing is wrong and the "
    "person is mistaken. Work in this order. Call where_checks_fail and check_sources first. If a source table is "
    "reported bad and the damage starts where that source is read, call close_without_fix with kind source_data and "
    "that table. If a node fails and no source does, call check_node on the node you blame, then try_fix with its "
    f"complete replacement SELECT; you have {MAX_TRIES} tries, and submit_diagnosis takes only a fix try_fix verified. "
    "If no check fails anywhere, read the SQL upstream of the report line by line before you conclude anything: some "
    "faults change values and trip no check. Call close_without_fix with kind nothing_wrong only when you have read "
    "that SQL and it is correct. Change as little as possible and keep every column. If you cannot tell, do not "
    "guess: say so in plain text and the incident goes to a person."
)


class Close(BaseModel):
    kind: Literal["source_data", "nothing_wrong"] = Field(description="source_data: a source table arrived bad. nothing_wrong: the report is right")
    source_table: str = Field("", description="for source_data, the raw table that is bad, e.g. orders")
    reason: str = Field(description="what you found, in one or two sentences")


CLOSE = "close_without_fix"


def _diff(before: str, after: str) -> str:
    split = lambda s: [x.strip() for x in re.sub(r",\s+(?![^()]*\))", ",\n", re.sub(
        r"(?<!left)(?<!inner)\s+(?=(?:from|where|group by|order by|left join|inner join|join)\b)", "\n", s)).split("\n")]
    return "\n".join(l for l in difflib.unified_diff(split(before), split(after), lineterm="", n=0)
                     if not l.startswith(("---", "+++", "@@")))


def checked_tools(wh: Warehouse, dag: PipelineDAG, incident: dict, session: Session, history: bool = False,
                  open_world: bool = False) -> list[ToolSpec]:
    """The checker as tools the model calls, and as rules it cannot skip.

    The model is told to check before it tries and to try before it submits. Being told is not the
    same as doing it, so each of those is also enforced here: an unchecked node cannot be tried, a
    contradicted node cannot be tried, and only SQL a rehearsal passed can be submitted. Every
    refusal is recorded, because how often the harness had to refuse is the measure of how far the
    instruction alone would have got.

    The rehearsal says whether the report and the rest of the warehouse came out right, and nothing
    more. It does not name which tables are still wrong: the first of those is the answer.
    """
    status, symptom = incident["node_status"], incident["symptom_node"]

    def refuse(why: str) -> None:
        session.refusals.append(why)
        raise ValueError(why)

    def where_checks_fail() -> str:
        session.asked_where = True
        first = evidence.first_failures(dag, status, symptom)
        if not first:
            return ("nowhere: every health check passes at and upstream of the report. The checks cannot see this "
                    "problem. Read the SQL of the nodes upstream of the report.")
        return "checks first fail at: " + ", ".join(first) + ". Nothing upstream of there fails a check."

    def check_node(node: str) -> str:
        if node not in dag.nodes:
            raise ValueError(f"'{node}' is not in this pipeline. Call pipeline to see the nodes.")
        ev = evidence.check(dag, status, symptom, node)
        session.checked[node] = ev.verdict
        return f"{ev.verdict}: {ev.why}"

    def gate(node: str) -> None:
        if node not in session.checked:
            refuse(f"call check_node on {node} before proposing a fix for it")
        if session.checked[node] == evidence.CONTRADICTED:
            refuse(f"the checks contradict {node} as the cause, so a fix there is not accepted. Call check_node "
                   "on another node.")

    def try_fix(fix_target: str, sql: str, reason: str = "") -> str:
        gate(fix_target)
        if session.tries >= MAX_TRIES:
            refuse(f"all {MAX_TRIES} tries are used. Do not submit: say in plain text that you could not verify a fix.")
        clean = validate(wh, dag, fix_target, sql)
        session.tries += 1
        out = rehearse(wh, fix_target, clean, symptom)
        left = MAX_TRIES - session.tries
        if out["functional"] and out["integrity"]:
            session.verified.append((fix_target, clean))
            session.reasons[(fix_target, clean)] = reason.strip()
            session.verified_on_try = session.verified_on_try or session.tries
            return ("verified: the report and every other table came out right. Now call submit_diagnosis with "
                    "exactly this node and this SQL, and one sentence saying what was wrong.")
        if out["functional"]:
            return (f"not verified: the report came out right, but other tables in the warehouse did not. The cause is "
                    f"not fixed. {left} tries left.")
        return f"not verified: the report is still wrong. {left} tries left."

    def submit(fix_target: str, sql: str, reason: str = "") -> str:
        gate(fix_target)
        if (fix_target, validate(wh, dag, fix_target, sql)) not in session.verified:
            refuse("only a fix that try_fix verified can be submitted, with the same node and the same SQL")
        return f"recorded: replace the SQL of {fix_target}"

    def recent_changes() -> str:
        session.asked_changes = True
        ds = wh.deploys_since_good()
        if not ds:
            return "no procedure has been deployed since the last good run"
        return "\n\n".join(f"{d['node']} was deployed:\n{_diff(d['before'], d['after'])}" for d in ds)

    def check_sources() -> str:
        session.asked_sources = True
        bad = observe_sources(wh)
        if not bad:
            return ("every source table has the rows, keys, nulls and dates it had on the last good run. These checks "
                    "look at shape, not values: a value in the wrong unit would not show here.")
        return "; ".join(f"raw.{t}: {', '.join(said)}" for t, said in bad.items())

    def close_without_fix(kind: str, source_table: str = "", reason: str = "") -> str:
        if not (session.asked_where and session.asked_sources):
            refuse("call where_checks_fail and check_sources before closing an incident without a fix")
        table = source_table.lower().removeprefix("raw.")
        if kind == "source_data" and table not in RAW_TABLES:
            raise ValueError(f"name the source table that is bad: one of {', '.join(RAW_TABLES)}")
        session.closed = {"kind": kind, "source_table": table if kind == "source_data" else "", "reason": reason.strip()}
        return "recorded: no SQL change" + (f"; raw.{table} goes to its owner" if kind == "source_data" else "")

    class NoArgs(BaseModel):
        pass

    read = [t for t in read_tools(wh, dag, incident) if t.name != TERMINAL]
    extra = []
    if history:
        extra.append(ToolSpec("recent_changes", "The SQL deployed since the last good run, as a diff per procedure.",
                              NoArgs, recent_changes))
    if open_world:
        extra += [ToolSpec("check_sources", "Check the source tables the pipeline reads against the last good run: "
                           "rows, duplicate keys, nulls, latest date. Call this first, with where_checks_fail.", NoArgs, check_sources),
                  ToolSpec(CLOSE, "End the incident with no SQL change: a source table arrived bad, or nothing is wrong.",
                           Close, close_without_fix)]
    return read + extra + [
        ToolSpec("where_checks_fail", "Where the health checks first fail upstream of the report, or that they fail "
                 "nowhere. Call this first.", NoArgs, where_checks_fail),
        ToolSpec("check_node", "Check one node as the cause against the health checks: supported, contradicted or "
                 "unverified, with the reason. Required before a fix for that node.", NodeArg, check_node),
        ToolSpec("try_fix", f"Run a fix on a rehearsal copy and learn whether the warehouse came out right. Nothing is "
                 f"kept. {MAX_TRIES} tries.", Attempt, try_fix),
        ToolSpec(TERMINAL, "Submit a fix that try_fix verified: the same node and the same SQL. Call this once.",
                 Fix, submit),
    ]


def draft_checked(wh: Warehouse, dag: PipelineDAG, incident: dict, model: ChatModel, max_steps: int = 26,
                  history: bool = False, open_world: bool = False) -> tuple[Investigation, str | None, Session]:
    """`history` adds the deploy history as a tool. `open_world` stops assuming the SQL is at fault: the
    model also gets the source checks and a way to end with no fix, and `session.closed` says if it did."""
    session = Session()
    task = (f"Incident {incident.get('id')}: {incident.get('symptom')}. "
            + ("Find out what is wrong, if anything, and end the incident the right way." if open_world else
               "Find the procedure whose SQL is wrong, verify a fix for it, and submit it."))
    inv = investigate(incident, dag, None, model, max_steps=max_steps,
                      specs=checked_tools(wh, dag, incident, session, history, open_world),
                      instructions=(OPEN_INSTRUCTIONS if open_world else CHECKED_INSTRUCTIONS) + (HISTORY_NOTE if history else ""),
                      task_text=task, also_ends=(CLOSE,) if open_world else ())
    if inv.closed:
        return inv, None, session
    if inv.submitted:
        return inv, validate(wh, dag, inv.fix_target, inv.steps[-1]["args"]["sql"]), session
    if session.verified:
        # The rehearsal is the gate, and it passed. A model that then stops, or announces the fix in
        # prose, has still produced a verified fix; the closing call adds nothing the harness needs.
        inv.fix_target, sql = session.verified[-1]
        inv.reason = (session.reasons.get((inv.fix_target, sql))
                      or "(the model stopped after the rehearsal verified this fix and gave no reason)")
        session.taken_from_rehearsal = True
        return inv, sql, session
    return inv, None, session
