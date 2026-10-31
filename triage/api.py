"""HTTP API. Run with: uvicorn triage.api:app --reload"""
from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .service import TriageService

app = FastAPI(title="pipeline-triage-graph", version="0.1.0")
svc = TriageService.from_env()


class NodeStatus(BaseModel):
    status: Literal["ok", "stale", "failed", "missing", "changed"] = "ok"
    error: str | None = None
    note: str | None = None
    changes: list[str] = Field(default_factory=list)


class Incident(BaseModel):
    id: str | None = None
    symptom_node: str
    symptom: str = ""
    node_status: dict[str, NodeStatus] = Field(default_factory=dict)
    observations: dict[str, str] = Field(default_factory=dict)


class Decision(BaseModel):
    approve: bool
    reviewer: str | None = None


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "pipeline": svc.dag.name, "nodes": len(svc.dag.nodes), "cases": len(svc.library.cases)}


@app.post("/incidents")
def open_incident(inc: Incident) -> dict:
    try:
        _, view = svc.open(inc.model_dump(exclude_none=True))
    except ValueError as e:
        raise HTTPException(422, str(e))
    return view


@app.get("/incidents/{run_id}")
def get_incident(run_id: str) -> dict:
    try:
        return svc.view(run_id)
    except KeyError:
        raise HTTPException(404, "unknown run")


@app.post("/incidents/{run_id}/decision")
def decide(run_id: str, d: Decision) -> dict:
    try:
        return svc.decide(run_id, d.approve, d.reviewer)
    except ValueError as e:
        raise HTTPException(409, str(e))
