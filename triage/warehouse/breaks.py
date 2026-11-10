"""Breaks: the kind of change that gets merged on a Friday.

Each one is a small edit to one procedure's SQL, written as the fragment that changed, so it reads
like the diff of the commit that caused it. Most of them are silent: the procedure runs, the table
fills, and the numbers are wrong. Those are the ones worth building for, because a break that throws
names itself.

Two families are held out whole. Nothing that is tuned against this set may see a null-handling or a
changed-grain break, so a result on them is a result on a kind of failure it has not met.
"""
from __future__ import annotations

from dataclasses import dataclass

from .pipeline import BY_ID

OOD_FAMILIES = ("null_handling", "changed_grain")


@dataclass(frozen=True)
class Break:
    family: str
    node: str
    name: str
    what: str
    find: str
    replace: str

    @property
    def id(self) -> str:
        return f"{self.family}.{self.name}"

    def apply(self, sql: str) -> str:
        if self.find not in sql:
            raise ValueError(f"{self.id}: the SQL of {self.node} no longer contains {self.find!r}")
        return sql.replace(self.find, self.replace)

    def broken_sql(self) -> str:
        return self.apply(BY_ID[self.node].sql)


BREAKS: tuple[Break, ...] = (
    Break("type_drift", "stg_lineitem", "price_as_integer", "prices lose their cents",
          "l_extendedprice as extended_price", "cast(l_extendedprice as integer) as extended_price"),
    Break("type_drift", "stg_lineitem", "discount_as_integer", "a discount of 0.05 becomes 0, so nothing is discounted",
          "l_discount as discount", "cast(l_discount as integer) as discount"),
    Break("dropped_column", "stg_orders", "order_date_dropped", "the staging model stops selecting the order date",
          ", o_orderdate as order_date", ""),
    Break("dropped_column", "stg_customer", "segment_dropped", "the staging model stops selecting the market segment",
          ", c_mktsegment as segment", ""),
    Break("bad_join_key", "int_part_cost", "part_without_supplier", "cost joined on the part alone: one line, every supplier's price",
          " and ps.ps_suppkey = l.supplier_id", ""),
    Break("bad_join_key", "mart_supplier_performance", "order_without_line", "cost joined on the order alone: each line gets every line's cost",
          " and c.line_no = i.line_no", ""),
    Break("date_window", "mart_daily_revenue", "latest_day_excluded", "the window stops one day short of today",
          "order_date <= (select as_of from meta)", "order_date < (select as_of from meta)"),
    Break("date_window", "rpt_revenue_dashboard", "trailing_six_days", "a seven-day total that adds up six days",
          "- 7)", "- 6)"),
    Break("null_handling", "int_customer_geo", "inner_join_nation", "customers with no nation on file disappear",
          "left join raw.nation n", "join raw.nation n"),
    Break("null_handling", "mart_supplier_performance", "cost_not_defaulted", "a line with no supplier price drops out of the margin",
          "sum(i.net_revenue - coalesce(c.supply_cost, 0))", "sum(i.net_revenue - c.supply_cost)"),
    Break("changed_grain", "int_customer_geo", "one_row_per_order", "a join added for one more column makes it one row per order",
          "left join raw.region r on r.r_regionkey = n.n_regionkey",
          "left join raw.region r on r.r_regionkey = n.n_regionkey left join stg_orders o on o.customer_id = c.customer_id"),
    Break("changed_grain", "mart_customer_revenue", "one_row_per_month", "grouped by month as well, so a customer is several rows",
          "group by g.customer_id, g.region, g.segment",
          "group by g.customer_id, g.region, g.segment, date_trunc('month', i.order_date)"),
)


def split_of(b: Break) -> str:
    """evolve: may be tuned against. held_in: same families, never tuned against. ood: families never seen."""
    if b.family in OOD_FAMILIES:
        return "ood"
    same_family = [x for x in BREAKS if x.family == b.family]
    return "evolve" if same_family.index(b) % 2 == 0 else "held_in"
