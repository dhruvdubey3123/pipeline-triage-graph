"""Check a diagnosis against the health checks, instead of against another opinion.

The cross-check in the triage graph lets a model's diagnosis through only when it names the node the
tracer named. That stops every wrong answer the tracer disagrees with, and it also stops every right
answer the tracer has nothing to say about. This asks a different question: whoever produced the
diagnosis, do the checks say the damage starts where it says?

    supported     the checks first fail at the node it blames, and at nothing upstream of it
    contradicted  the checks say otherwise: it is not upstream of the report, something upstream of
                  it is already failing, or the failure is somewhere it cannot explain
    unverified    the checks have nothing to say, because they pass at the node it blames

Only `supported` should reach an approval gate on its own. The other two go to a person with the
reason attached.

What `supported` does not mean: that the SQL there is wrong. It means the first visible damage is
there. A break that passes its own node's checks and only shows further down would get a diagnosis of
the lower node supported. None of the twelve warehouse breaks does that, and nothing here rules it out.
A break that changes values and trips no check at all can never be supported by this; the only thing
that verifies one of those is running the fix and comparing the result.

`origin` is the same evidence used on its own, with no diagnosis to check: if the checks first fail at
exactly one place, that place. It is the rule-only baseline a model has to beat.
"""
from __future__ import annotations

from dataclasses import dataclass

from .dag import PipelineDAG

SUPPORTED, CONTRADICTED, UNVERIFIED = "supported", "contradicted", "unverified"


@dataclass(frozen=True)
class Evidence:
    verdict: str
    why: str


def _bad(node_status: dict, node: str) -> bool:
    return node_status.get(node, {}).get("status", "ok") != "ok"


def first_failures(dag: PipelineDAG, node_status: dict, symptom_node: str) -> list[str]:
    """Nodes at or upstream of the symptom where a check fails and none fails upstream of them."""
    scope = [symptom_node, *dag.upstream(symptom_node)]
    return sorted(n for n in scope if _bad(node_status, n) and not any(_bad(node_status, a) for a in dag.upstream(n)))


def origin(dag: PipelineDAG, node_status: dict, symptom_node: str) -> str | None:
    first = first_failures(dag, node_status, symptom_node)
    return first[0] if len(first) == 1 else None


def check(dag: PipelineDAG, node_status: dict, symptom_node: str, target: str | None, cause: str | None = None) -> Evidence:
    if not target:
        return Evidence(UNVERIFIED, "no diagnosis to check")
    scope = {symptom_node, *dag.upstream(symptom_node)}
    blamed = cause if cause and cause != target else target          # where the diagnosis says it starts
    for n in {target, blamed}:
        if n not in dag.nodes or n not in scope:
            return Evidence(CONTRADICTED, f"{n} is not upstream of {symptom_node}")
    if blamed != target:
        if blamed not in dag.upstream(target):
            return Evidence(CONTRADICTED, f"{blamed} is not upstream of {target}")
        if not _bad(node_status, target):
            return Evidence(CONTRADICTED, f"{target} passes its checks, so nothing upstream broke it")
    first = first_failures(dag, node_status, symptom_node)
    if not first:
        return Evidence(UNVERIFIED, "every check passes at and upstream of the report")
    if _bad(node_status, blamed):
        above = [a for a in dag.upstream(blamed) if _bad(node_status, a)]
        if above:
            return Evidence(CONTRADICTED, f"checks already fail upstream of {blamed}, at {', '.join(sorted(above))}")
        if first == [blamed]:
            return Evidence(SUPPORTED, f"checks first fail at {blamed} and at nothing upstream of it")
        return Evidence(UNVERIFIED, f"checks also first fail at {', '.join(n for n in first if n != blamed)}, "
                                    f"which {blamed} does not explain")
    unexplained = [n for n in first if blamed not in dag.upstream(n)]
    if unexplained:
        return Evidence(CONTRADICTED, f"{blamed} passes its checks, and they fail at {', '.join(unexplained)}, "
                                      f"which is not downstream of it")
    return Evidence(UNVERIFIED, f"{blamed} passes its checks; they first fail downstream of it")
