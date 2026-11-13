"""Incidents where the SQL is fine.

Most of what goes wrong in a pipeline is not a bad merge. A file arrives late, a load runs twice, a
source system changes a unit or stops filling a column. Every procedure is exactly as it was, and
the report is wrong anyway. The right answer to these is not a SQL fix: it is to say which source is
bad, so its owner can reload it. An agent that only knows how to propose fixes will propose one.

Each fault changes one raw table. `visible` says whether the checks on the source tables (row count,
duplicate keys, nulls, latest date) can see it at all. One of them cannot: a price column in the
wrong unit has the same rows, keys, nulls and dates as before.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceFault:
    name: str
    table: str
    what: str
    statements: tuple[str, ...]
    visible: bool = True

    @property
    def id(self) -> str:
        return f"source.{self.name}"


SOURCE_FAULTS: tuple[SourceFault, ...] = (
    SourceFault("orders_loaded_twice", "orders", "one batch of orders was loaded a second time",
                ("insert into raw.orders select * from raw.orders where o_orderkey % 20 = 0",)),
    SourceFault("lineitems_late", "lineitem", "the last three days of line items have not arrived",
                ("delete from raw.lineitem where l_orderkey in (select o_orderkey from raw.orders "
                 "where o_orderdate > (select as_of from meta) - 3)",)),
    SourceFault("segment_not_filled", "customer", "the source stopped filling the market segment for some customers",
                ("update raw.customer set c_mktsegment = null where c_custkey % 6 = 0",)),
    SourceFault("suppliers_missing", "supplier", "a fifth of the suppliers are missing from the extract",
                ("delete from raw.supplier where s_suppkey % 5 = 0",)),
    SourceFault("cost_in_cents", "partsupp", "the supplier price started arriving in cents instead of dollars",
                ("update raw.partsupp set ps_supplycost = ps_supplycost * 100",), visible=False),
)
BY_ID = {f.id: f for f in SOURCE_FAULTS}
