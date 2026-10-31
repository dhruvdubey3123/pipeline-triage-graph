"""Multi-agent incident triage for data pipelines."""
from .dag import PipelineDAG, trace
from .knowledge import CaseLibrary
from .service import TriageService

__all__ = ["PipelineDAG", "trace", "CaseLibrary", "TriageService"]
__version__ = "0.1.0"
