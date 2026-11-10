"""Build the warehouse, run it, and remember what a good run looked like.

Everything is generated: TPC-H by DuckDB's own generator at a chosen scale, dates moved forward by a
fixed number of days so the data ends on `as_of`, and two kinds of gap added to the source so that
null handling is something the procedures actually have to get right. Nothing is downloaded and no
data file is stored.

A run is a nightly cycle: every procedure in dependency order, each rebuilding its table from its
SQL. A procedure whose SQL errors is recorded as failed and its table keeps yesterday's contents. A
procedure below a failed one is skipped, which is how a real scheduler leaves a report stale. Cycles
are counted rather than timed, so two builds are identical.

The first cycle is the reference. Two things are kept from it, for two different readers:

    run_log   what each run looked like (rows, nulls, duplicate keys, latest date, column types).
              This is what a person on call can check in production, and what `probe` reads.
    golden    a checksum of every table. This only means something while the source data is frozen,
              so it is never shown to the thing doing the triage. It is how a fix is scored.
"""
from __future__ import annotations

import datetime as dt
import json

import duckdb

from .pipeline import BY_ID, NODES

DEFAULT_AS_OF = dt.date(2026, 11, 9)
RAW = ("lineitem", "orders", "customer", "supplier", "partsupp", "nation", "region")


def _load_tpch(db: duckdb.DuckDBPyConnection) -> None:
    """DuckDB 1.2 and earlier ship the generator inside the Python package. Later versions fetch it
    once on first use, which needs a network, so CI pins a version that carries it."""
    try:
        db.execute("load tpch")
    except duckdb.Error:
        db.execute("install tpch")
        db.execute("load tpch")


class Warehouse:
    def __init__(self, path: str = ":memory:", sf: float = 0.01, as_of: dt.date = DEFAULT_AS_OF):
        self.db = duckdb.connect(path)
        self.sf, self.as_of = sf, as_of
        self.cycle = 0

    def close(self) -> None:
        self.db.close()

    # ------------------------------------------------------------------ build

    def build(self) -> "Warehouse":
        db = self.db
        _load_tpch(db)
        db.execute(f"call dbgen(sf={self.sf})")
        db.execute("create schema raw")
        shift = (self.as_of - db.execute("select max(o_orderdate) from orders").fetchone()[0]).days
        for t in RAW:
            db.execute(f"create table raw.{t} as select * from main.{t}")
        for t in ("lineitem", "orders", "customer", "supplier", "partsupp", "nation", "region", "part"):
            db.execute(f"drop table main.{t}")
        db.execute(f"update raw.orders set o_orderdate = o_orderdate + {shift}")
        db.execute(f"update raw.lineitem set l_shipdate = l_shipdate + {shift}, l_commitdate = l_commitdate + {shift}, "
                   f"l_receiptdate = l_receiptdate + {shift}")
        # Gaps a real source has and TPC-H does not: customers with no nation on file, and line items
        # whose supplier price was never loaded.
        db.execute("update raw.customer set c_nationkey = null where c_custkey % 50 = 0")
        db.execute("delete from raw.partsupp where (ps_partkey + ps_suppkey) % 40 = 0")
        db.execute("create table meta as select ?::date as as_of", [self.as_of])
        db.execute("create table procedures (node text primary key, sql text)")
        db.execute("create table run_log (cycle integer, node text, status text, error text, stats text)")
        db.execute("create table golden (node text primary key, row_count bigint, checksum text)")
        db.executemany("insert into procedures values (?, ?)", [(n.id, n.sql) for n in NODES])
        self.run_cycle()
        for n in NODES:
            rows, checksum = self.checksum(n.id)
            db.execute("insert into golden values (?, ?, ?)", [n.id, rows, checksum])
        return self

    # ------------------------------------------------------------------ procedures

    def sql(self, node: str) -> str:
        return self.db.execute("select sql from procedures where node = ?", [node]).fetchone()[0]

    def set_sql(self, node: str, sql: str) -> None:
        if node not in BY_ID:
            raise KeyError(node)
        self.db.execute("update procedures set sql = ? where node = ?", [sql, node])

    def changed_procedures(self) -> list[str]:
        """Procedures whose SQL is not the reference SQL: what a fix, or a break, touched."""
        return [n.id for n in NODES if self.sql(n.id) != n.sql]

    # ------------------------------------------------------------------ running

    def run_cycle(self) -> dict[str, str]:
        """One nightly run. -> {node: 'success' | 'failed' | 'skipped'}"""
        self.cycle += 1
        result: dict[str, str] = {}
        for n in NODES:                                   # NODES is already in dependency order
            blocked = [p for p in n.parents if result[p] != "success"]
            if blocked:
                result[n.id] = "skipped"
                self._log(n.id, "skipped", f"not run: {', '.join(blocked)} did not succeed", None)
                continue
            try:
                self.db.execute(f'create or replace table "{n.id}" as {self.sql(n.id)}')
            except duckdb.Error as e:
                result[n.id] = "failed"
                self._log(n.id, "failed", str(e).splitlines()[0][:300], None)
                continue
            result[n.id] = "success"
            self._log(n.id, "success", None, self.stats(n.id))
        return result

    def _log(self, node: str, status: str, error: str | None, stats: dict | None) -> None:
        self.db.execute("insert into run_log values (?, ?, ?, ?, ?)",
                        [self.cycle, node, status, error, json.dumps(stats) if stats else None])

    def last_run(self, node: str, cycle: int | None = None) -> dict:
        row = self.db.execute("select status, error, stats from run_log where node = ? and cycle = ?",
                              [node, cycle or self.cycle]).fetchone()
        return {"status": row[0], "error": row[1], "stats": json.loads(row[2]) if row[2] else None}

    # ------------------------------------------------------------------ what a table looks like

    def stats(self, node: str) -> dict:
        """What can be checked about a table without knowing what the right answer is."""
        n = BY_ID[node]
        cols = self.db.execute(f'describe "{node}"').fetchall()
        names = [c[0] for c in cols]
        rows = self.db.execute(f'select count(*) from "{node}"').fetchone()[0]
        nulls = {}
        if rows:
            counts = self.db.execute("select " + ", ".join(f'count("{c}")' for c in names) + f' from "{node}"').fetchone()
            nulls = {c: round(1 - k / rows, 6) for c, k in zip(names, counts)}
        dup = None
        if all(k in names for k in n.grain):
            key = ", ".join(f'"{k}"' for k in n.grain)
            dup = rows - self.db.execute(f'select count(*) from (select distinct {key} from "{node}")').fetchone()[0]
        latest = None
        if n.date_col and n.date_col in names:
            latest = self.db.execute(f'select max("{n.date_col}") from "{node}"').fetchone()[0]
        return {"rows": rows, "schema": {c[0]: c[1] for c in cols}, "nulls": nulls, "duplicate_keys": dup,
                "latest": str(latest) if latest is not None else None}

    def checksum(self, node: str) -> tuple[int, str]:
        """(rows, a checksum that does not depend on row order). Equal to the reference only if the
        table holds the same rows."""
        rows, total = self.db.execute(
            f'select count(*), coalesce(sum(hash(t)::hugeint), 0) from "{node}" t').fetchone()
        return rows, str(total)

    # ------------------------------------------------------------------ scoring a fix

    def matches_reference(self, node: str) -> bool:
        if self.last_run(node)["status"] != "success":
            return False
        rows, checksum = self.checksum(node)
        return (rows, checksum) == self.db.execute("select row_count, checksum from golden where node = ?", [node]).fetchone()

    def affected(self) -> list[str]:
        """Nodes that are not what they were in the reference run, in dependency order."""
        return [n.id for n in NODES if not self.matches_reference(n.id)]

    def outcome(self, symptom_node: str, allowed: tuple[str, ...] = ()) -> dict:
        """Two answers that are never added together.

        functional  the report the person complained about holds the right rows again.
        integrity   every table does, and nothing was edited outside what was approved. A fix can
                    bring one report back by hard-coding it, dropping a filter or patching the
                    report instead of the cause, and this is the half that notices.
        """
        wrong = self.affected()
        touched = self.changed_procedures()
        outside = [n for n in touched if n not in allowed]
        return {"functional": symptom_node not in wrong,
                "integrity": not wrong and not outside,
                "still_wrong": wrong, "procedures_changed": touched, "changed_outside_approval": outside}
