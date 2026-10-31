"""Edges of the system that touch the real world: data probes and the executor.

Both are protocols with an offline default so the graph runs anywhere. In a warehouse
deployment, DataProbe would run checks against semantic views (for example Snowflake
Cortex Analyst) and Executor would call your orchestrator's API to re-run jobs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class DataProbe(Protocol):
    def check(self, node_id: str, incident: dict) -> str | None: ...


class FixtureProbe:
    """Reads pre-recorded observations from the incident payload."""

    def check(self, node_id: str, incident: dict) -> str | None:
        return incident.get("observations", {}).get(node_id)


class Executor(Protocol):
    def apply(self, plan: list[str]) -> list[str]: ...


@dataclass
class DryRunExecutor:
    """Records what it would do. Nothing is executed. This is the default on purpose."""

    log: list[str] = field(default_factory=list)

    def apply(self, plan: list[str]) -> list[str]:
        done = [f"[dry-run] {step}" for step in plan]
        self.log.extend(done)
        return done
