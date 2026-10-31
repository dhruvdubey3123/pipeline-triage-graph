"""Runs incidents through the graph and handles the human decision."""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from .dag import PipelineDAG
from .graph import build_graph
from .knowledge import CaseLibrary

ROOT = Path(__file__).resolve().parent.parent


class TriageService:
    def __init__(self, dag: PipelineDAG, library: CaseLibrary, **graph_kwargs):
        self.dag = dag
        self.library = library
        self.app = build_graph(dag, library, **graph_kwargs)

    @classmethod
    def from_env(cls) -> "TriageService":
        dag = PipelineDAG.from_yaml(os.getenv("TRIAGE_PIPELINE", ROOT / "examples/pipeline.yaml"))
        lib = CaseLibrary.load(os.getenv("TRIAGE_CASES", ROOT / "examples/case_library.yaml"))
        return cls(dag, lib, min_confidence=float(os.getenv("TRIAGE_MIN_CONFIDENCE", "0.5")))

    def _cfg(self, run_id: str) -> dict:
        return {"configurable": {"thread_id": run_id}}

    def open(self, incident: dict) -> tuple[str, dict]:
        run_id = incident.get("id") or f"run-{uuid.uuid4().hex[:8]}"
        self.app.invoke({"incident": incident, "log": []}, self._cfg(run_id))
        return run_id, self.view(run_id)

    def decide(self, run_id: str, approve: bool, reviewer: str | None = None) -> dict:
        snap = self.app.get_state(self._cfg(run_id))
        if "gate" not in (snap.next or ()):
            raise ValueError(f"{run_id} is not waiting for a decision")
        self.app.update_state(self._cfg(run_id), {"human_decision": "approve" if approve else "reject",
                                                   "reviewer_name": reviewer})
        self.app.invoke(None, self._cfg(run_id))
        return self.view(run_id)

    def view(self, run_id: str) -> dict:
        snap = self.app.get_state(self._cfg(run_id))
        if not snap.values:
            raise KeyError(run_id)
        v = dict(snap.values)
        v["run_id"] = run_id
        v["waiting_for_human"] = "gate" in (snap.next or ())
        v.pop("incident", None)
        return v
