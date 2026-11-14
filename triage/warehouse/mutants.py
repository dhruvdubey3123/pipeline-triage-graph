"""Breaks nobody wrote by hand.

The twelve in breaks.py were chosen, which means they were also chosen by someone who knew what the
agent can do. These are produced by rules: take a procedure's SQL and make one small edit of a kind
people really merge. A boundary moved by one, a join that stopped being a left join, a default that
went missing, a sum that became an average. Every site in every procedure is tried.

A mutant is kept only if it is a real incident: it changes at least one report. An edit that leaves
every report as it was is not a break anyone would file a ticket for, and it is dropped. Mutants that
reproduce a hand-written break are dropped too, so the two sets do not overlap.

Two kinds are held out whole, the same way two families are in breaks.py.
"""
from __future__ import annotations

import re
from typing import Callable, Iterator

from .breaks import BREAKS, Break
from .build import Warehouse
from .pipeline import NODES, REPORTS

OOD_KINDS = ("null_default", "aggregate")


def _each(pattern: str, sql: str, repl: Callable[[re.Match], str | None]) -> Iterator[str]:
    """One mutant per place the pattern matches: that place changed and every other left alone."""
    for m in re.finditer(pattern, sql):
        new = repl(m)
        if new is not None and new != m.group(0):
            yield sql[:m.start()] + new + sql[m.end():]


FLIP = {"<=": "<", ">=": ">", "<": "<=", ">": ">="}
KINDS: dict[str, tuple[str, Callable[[str], Iterator[str]]]] = {
    "boundary": ("a comparison that includes or excludes one value it should not",
                 lambda s: _each(r"(?<![<>=!])(<=|>=|<|>)(?![<>=])", s, lambda m: FLIP[m.group(1)])),
    "constant": ("a window that is one day longer or shorter",
                 lambda s: [*_each(r"- (\d+)\)", s, lambda m: f"- {int(m.group(1)) - 1})"),
                           *_each(r"- (\d+)\)", s, lambda m: f"- {int(m.group(1)) + 1})")]),
    "join_kind": ("a left join that became an inner join, so unmatched rows disappear",
                  lambda s: _each(r"\bleft join\b", s, lambda m: "join")),
    "join_key": ("a join that lost one of its conditions",
                 lambda s: _each(r" and [\w.]+ = [\w.]+", s, lambda m: "")),
    "filter_dropped": ("a filter that was removed",
                       lambda s: _each(r" and \w+ (?:<=|>=|<|>) \(select \w+ from \w+\)", s, lambda m: "")),
    "sign": ("a discount added where it should be taken off",
             lambda s: _each(r"\(1 - ([\w.]+)\)", s, lambda m: f"(1 + {m.group(1)})")),
    "type_cast": ("a numeric column cast to an integer",
                  lambda s: _each(r"\b([a-z]\w*\.)?(l_extendedprice|l_discount|l_quantity) as (\w+)", s,
                                  lambda m: f"cast({m.group(1) or ''}{m.group(2)} as integer) as {m.group(3)}")),
    "column_dropped": ("a column the staging model stopped selecting",
                       lambda s: _each(r", [a-z]_\w+ as \w+", s, lambda m: "")),
    "null_default": ("a default for missing values that was removed",
                     lambda s: _each(r"coalesce\(([^(),]+|\w+\([^()]*\)), [^(),]+\)", s, lambda m: m.group(1))),
    "aggregate": ("a total that became an average, or a distinct count that stopped being distinct",
                  lambda s: (x for pat, rep in ((r"\bsum\(", lambda m: "avg("), (r"count\(distinct ", lambda m: "count("))
                             for x in _each(pat, s, rep))),
}


def candidates() -> list[Break]:
    """Every single-edit variant of every procedure, in a fixed order. Not yet checked to be a break."""
    out, taken = [], {b.broken_sql() for b in BREAKS}
    for n in NODES:
        for kind, (what, make) in KINDS.items():
            k = 0
            for sql in make(n.sql):
                if sql in taken:
                    continue
                taken.add(sql)
                k += 1
                out.append(Break(f"gen_{kind}", n.id, f"{n.id}_{k}", what, n.sql, sql))
    return out


def generate(wh: Warehouse | None = None) -> list[dict]:
    """-> the mutants that are real incidents, each with what it did. Deterministic."""
    own = wh is None
    wh = wh or Warehouse().build()
    out = []
    for b in candidates():
        wh.set_sql(b.node, b.replace, record=False)
        ran = wh.run_cycle()
        wrong = wh.affected()
        wh.set_sql(b.node, b.find, record=False)
        hit = [r for r in REPORTS if r in wrong]
        if hit:
            out.append({"id": b.id, "family": b.family, "node": b.node, "what": b.what, "sql": b.replace,
                        "silent": all(v == "success" for v in ran.values()), "reports": hit, "split": split_of(b, out)})
    wh.run_cycle()
    if own:
        wh.close()
    return out


def split_of(b: Break, before: list[dict]) -> str:
    kind = b.family.removeprefix("gen_")
    if kind in OOD_KINDS:
        return "ood"
    return "evolve" if sum(1 for x in before if x["family"] == b.family) % 2 == 0 else "held_in"


def as_break(row: dict) -> Break:
    from .pipeline import BY_ID
    return Break(row["family"], row["node"], row["id"].split(".", 1)[1], row["what"], BY_ID[row["node"]].sql, row["sql"])
