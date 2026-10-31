"""MCP server exposing pipeline context to any agent (Claude Code, Claude Desktop, your own).

    python -m triage.mcp_server            # stdio
    TRIAGE_PIPELINE=path/to/dag.yaml python -m triage.mcp_server

Tools are read-only on purpose. Changing the pipeline goes through the graph's human gate.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from .dag import PipelineDAG, trace
from .knowledge import CaseLibrary

ROOT = Path(__file__).resolve().parent.parent
DAG = PipelineDAG.from_yaml(os.getenv("TRIAGE_PIPELINE", ROOT / "examples/pipeline.yaml"))
LIB = CaseLibrary.load(os.getenv("TRIAGE_CASES", ROOT / "examples/case_library.yaml"))

server = MCPServer("pipeline-context", instructions="Read-only lineage and precedent lookup for a data pipeline.")


@server.tool()
def list_nodes(kind: str | None = None) -> list[dict]:
    """List pipeline nodes, optionally filtered by kind (source, load, procedure, table, report)."""
    return [{"id": n.id, "kind": n.kind, "owner": n.owner} for n in DAG.nodes.values() if kind in (None, n.kind)]


@server.tool()
def get_upstream(node_id: str) -> list[str]:
    """Every node that feeds into node_id, nearest first."""
    return DAG.upstream(node_id)


@server.tool()
def get_downstream(node_id: str) -> list[str]:
    """Every node that depends on node_id, nearest first."""
    return DAG.downstream(node_id)


@server.tool()
def trace_failure(symptom_node: str, node_status_json: str) -> dict:
    """Walk upstream from symptom_node through unhealthy nodes and return the root cause.

    node_status_json maps node id to {"status": ok|stale|failed|missing|changed, "error": "..."}.
    """
    res = trace(DAG, json.loads(node_status_json), symptom_node)
    return {"fix_target": res.fix_target, "upstream_cause": res.upstream_cause, "ambiguous": res.ambiguous,
            "hops": [h.__dict__ for h in res.hops]}


@server.tool()
def search_precedents(query: str, k: int = 3) -> list[dict]:
    """Resolved cases most similar to the query text."""
    return [{"id": c.id, "category": c.category, "summary": c.summary, "resolution": c.resolution, "score": s}
            for c, s in LIB.search(query, k)]


if __name__ == "__main__":
    server.run()
