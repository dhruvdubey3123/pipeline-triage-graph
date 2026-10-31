"""Pipeline dependency graph and the upstream tracer.

The tracer is the core idea of the project: start at the node where a human noticed
something wrong, walk upstream through unhealthy dependencies only, rule out healthy
branches, and stop at the most upstream unhealthy node. That node, and the evidence
gathered on the way, is what every later agent reasons over.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml

HEALTHY = {"ok"}
# Statuses that mean "something is wrong here", ordered by how strongly they point at a root cause.
ROOT_STRENGTH = {"failed": 3, "missing": 3, "changed": 2, "stale": 1}


@dataclass(frozen=True)
class Node:
    id: str
    kind: str
    owner: str = ""


@dataclass
class PipelineDAG:
    name: str
    nodes: dict[str, Node]
    parents: dict[str, list[str]]
    children: dict[str, list[str]]
    protected: set[str] = field(default_factory=set)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineDAG":
        raw = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "PipelineDAG":
        nodes = {n["id"]: Node(n["id"], n["kind"], n.get("owner", "")) for n in raw["nodes"]}
        parents: dict[str, list[str]] = {nid: [] for nid in nodes}
        children: dict[str, list[str]] = {nid: [] for nid in nodes}
        for a, b in raw["edges"]:
            if a not in nodes or b not in nodes:
                raise ValueError(f"edge references unknown node: {a} -> {b}")
            parents[b].append(a)
            children[a].append(b)
        dag = cls(raw.get("name", "pipeline"), nodes, parents, children, set(raw.get("protected", [])))
        dag._check_acyclic()
        return dag

    def _check_acyclic(self) -> None:
        state: dict[str, int] = {}

        def visit(n: str) -> None:
            if state.get(n) == 1:
                raise ValueError(f"cycle detected at {n}")
            if state.get(n) == 2:
                return
            state[n] = 1
            for c in self.children[n]:
                visit(c)
            state[n] = 2

        for n in self.nodes:
            visit(n)

    def upstream(self, node_id: str) -> list[str]:
        return self._walk(node_id, self.parents)

    def downstream(self, node_id: str) -> list[str]:
        return self._walk(node_id, self.children)

    @staticmethod
    def _walk(start: str, adj: dict[str, list[str]]) -> list[str]:
        seen, order, stack = {start}, [], list(adj[start])
        while stack:
            n = stack.pop(0)
            if n in seen:
                continue
            seen.add(n)
            order.append(n)
            stack.extend(adj[n])
        return order


@dataclass
class TraceHop:
    node: str
    status: str
    detail: str
    ruled_out: bool


@dataclass
class TraceResult:
    hops: list[TraceHop]
    fix_target: str | None
    upstream_cause: str | None
    candidates: list[str]
    ambiguous: bool

    @property
    def nodes_checked(self) -> int:
        return len(self.hops)


def _status(node_status: dict, nid: str) -> str:
    # A node absent from the health snapshot is treated as healthy: we only know what we were told.
    return node_status.get(nid, {}).get("status", "ok")


def _detail(node_status: dict, nid: str) -> str:
    s = node_status.get(nid, {})
    parts = [s.get("error", ""), s.get("note", ""), "; ".join(s.get("changes", []))]
    return " | ".join(p for p in parts if p) or s.get("status", "ok")


def trace(dag: PipelineDAG, node_status: dict, symptom_node: str) -> TraceResult:
    """Walk upstream from the symptom through unhealthy nodes only.

    Healthy parents are checked and recorded as ruled out, but never expanded. An unhealthy
    node whose parents are all healthy is a root candidate. When a candidate only *changed*
    (it ran, but its contract moved) and one of its unhealthy children *failed*, the failed
    child is the fix target and the changed node is recorded as the upstream cause.
    """
    if symptom_node not in dag.nodes:
        raise KeyError(f"unknown symptom node {symptom_node}")

    hops: list[TraceHop] = []
    seen: set[str] = set()
    candidates: list[str] = []
    frontier = [symptom_node]

    while frontier:
        nid = frontier.pop(0)
        if nid in seen:
            continue
        seen.add(nid)
        st = _status(node_status, nid)
        healthy = st in HEALTHY
        hops.append(TraceHop(nid, st, _detail(node_status, nid), ruled_out=healthy and nid != symptom_node))
        if healthy and nid != symptom_node:
            continue
        sick_parents = [p for p in dag.parents[nid] if _status(node_status, p) not in HEALTHY]
        for p in dag.parents[nid]:
            if p not in seen:
                frontier.append(p)
        if not sick_parents and nid != symptom_node:
            candidates.append(nid)

    if not candidates:
        return TraceResult(hops, None, None, [], ambiguous=False)

    # Resolve each candidate to the node a fix should target. A "changed" node that broke a
    # failed child resolves to that child, and inherits the child's (failed) strength.
    # If one change broke several children, each is its own fix target, which makes the
    # result ambiguous on purpose: a multi-target fix goes to a human, not to auto-proposal.
    resolved: dict[str, tuple[str | None, int]] = {}

    def _put(target: str, cause: str | None, st: str) -> None:
        prev, strength = resolved.get(target), ROOT_STRENGTH.get(st, 0)
        if prev is None or strength > prev[1] or (prev[0] is None and cause):
            resolved[target] = (cause, strength)

    for c in candidates:
        st = _status(node_status, c)
        failed_children = [ch for ch in dag.children[c] if _status(node_status, ch) == "failed"] if st == "changed" else []
        if failed_children:
            for ch in failed_children:
                _put(ch, c, "failed")
        else:
            _put(c, None, st)

    # Prefer the strongest evidence: failed/missing beats changed beats merely stale.
    ranked = sorted(resolved, key=lambda t: resolved[t][1], reverse=True)
    best = ranked[0]
    top = resolved[best][1]
    ambiguous = sum(1 for t in ranked if resolved[t][1] == top) > 1
    return TraceResult(hops, best, resolved[best][0], ranked, ambiguous)


def refresh_order(dag: PipelineDAG, start: str) -> list[str]:
    """Nodes to re-run after a fix at `start`, in dependency order."""
    affected = set([start, *dag.downstream(start)])
    order: list[str] = []
    remaining = set(affected)
    while remaining:
        ready = sorted(n for n in remaining if not (set(dag.parents[n]) & remaining))
        if not ready:  # pragma: no cover - guarded by _check_acyclic
            raise RuntimeError("cycle in refresh order")
        order.extend(ready)
        remaining -= set(ready)
    return order


def iter_kinds(dag: PipelineDAG, kinds: Iterable[str]) -> list[str]:
    ks = set(kinds)
    return [n.id for n in dag.nodes.values() if n.kind in ks]
