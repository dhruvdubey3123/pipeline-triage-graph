"""The investigator: an LLM agent that walks the dependency graph itself.

The tracer finds the root cause by walking the pipeline with fixed rules. The investigator is given
the same incident and the same pipeline through read-only tools and has to do that walk itself:
read the pipeline, check a node's reported status, look upstream of it, search resolved cases, and
submit a diagnosis. It decides what to look at; it cannot change anything.

What it submits is checked here before anything else sees it. An unknown category, a node that is
not in the pipeline, or a table nobody may patch directly comes back to the model as an error it
can read and correct, and the first valid submission ends the investigation: a model told to
"submit once" will sometimes keep going and talk itself out of a right answer.

The tools are plain functions with Pydantic argument models, so another runtime can execute the
same investigation under its own rules (agent-harness-runtime runs it with a crash-safe log).

Any server that speaks the OpenAI chat-completions API with tool calling works: Ollama, vLLM,
llama.cpp's server, LM Studio. The default is a local Ollama.

    ollama pull qwen3:8b
    python -m evals.run_eval --investigator --skip 60 --limit 20
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationError

from .dag import PipelineDAG, refresh_order
from .knowledge import CATEGORIES, CaseLibrary

INSTRUCTIONS = (
    "You are on call for a data pipeline. An incident says which table looks wrong. Find the node that "
    "actually broke, not the one that looks wrong: a stale report is usually the symptom of a failure "
    "further upstream. Use the tools to read the pipeline and the reported status of each node, and look "
    "for similar resolved cases. When you are confident, call submit_diagnosis once and stop. You cannot "
    "change anything from here, and nothing you submit is applied until a person approves it.\n"
    f"Categories: {', '.join(CATEGORIES)}."
)


# ------------------------------------------------------------------ tool arguments

class NoArgs(BaseModel):
    pass


class NodeArg(BaseModel):
    node: str


class Query(BaseModel):
    query: str


class Submission(BaseModel):
    category: str = Field(description=f"one of: {', '.join(CATEGORIES)}")
    fix_target: str = Field(description="the node that has to be fixed")
    cause: str = Field("", description="the upstream node that caused it, if any")
    reason: str = Field(description="one or two sentences an on-call engineer can act on")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args: type[BaseModel]
    fn: Callable[..., str]

    def schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.args.model_json_schema()}}


TERMINAL = "submit_diagnosis"


def derive_plan(dag: PipelineDAG, target: str) -> list[str]:
    """Derived, not proposed: once the target is known, what else to re-run is pipeline structure.
    The same derivation the rule-based orchestrator uses."""
    return [f"fix {target}", *[f"refresh {n}" for n in refresh_order(dag, target) if n != target]]


def tools(dag: PipelineDAG, library: CaseLibrary, incident: dict) -> list[ToolSpec]:
    """The investigator's whole reach: four ways to look, one way to answer, nothing that writes."""
    reported = incident.get("node_status", {})

    def one_line(node: str) -> str:
        st = reported.get(node)
        if st is None:
            return f"{node}: nothing reported"
        bits = [st.get("status", "?"), st.get("error", ""), st.get("note", ""), *st.get("changes", [])]
        return f"{node}: " + ", ".join(b for b in bits if b)

    def known(node: str) -> None:
        if node not in dag.nodes:
            raise ValueError(f"'{node}' is not in this pipeline. Call pipeline to see the nodes.")

    def pipeline() -> str:
        lines = [f"pipeline {dag.name}, symptom reported at {incident['symptom_node']}: {incident.get('symptom', '')}"]
        for name, node in dag.nodes.items():
            parents = ", ".join(dag.parents[name]) or "none"
            protect = "  [protected: fix the job that produces it instead]" if name in dag.protected else ""
            lines.append(f"  {name} ({node.kind}) reads from: {parents}{protect}")
        return "\n".join(lines)

    def status(node: str) -> str:
        known(node)
        return one_line(node)

    def upstream(node: str) -> str:
        known(node)
        seen, frontier, order = {node}, list(dag.parents[node]), []
        while frontier:
            nxt = []
            for n in frontier:
                if n not in seen:
                    seen.add(n)
                    order.append(n)
                    nxt += list(dag.parents[n])
            frontier = nxt
        return "\n".join(one_line(n) for n in order) or "nothing upstream"

    def precedents(query: str) -> str:
        hits = library.search(query, k=3)
        return "\n".join(f"{c.id} ({c.category}, score {s}): {c.summary} -> {c.resolution}"
                         for c, s in hits) or "no similar resolved cases"

    def submit_diagnosis(category: str, fix_target: str, cause: str = "", reason: str = "") -> str:
        if category not in CATEGORIES:
            raise ValueError(f"'{category}' is not a category. Use one of: {', '.join(CATEGORIES)}")
        known(fix_target)
        if fix_target in dag.protected:
            raise ValueError(f"{fix_target} is protected and cannot be patched directly. "
                             "Diagnose the job that produces it instead.")
        if cause:
            known(cause)
        return f"recorded: fix {fix_target} ({category}), {len(derive_plan(dag, fix_target))} plan steps"

    return [
        ToolSpec("pipeline", "The pipeline: every node, its kind, and what it reads from.", NoArgs, pipeline),
        ToolSpec("status", "What the incident reports about one node.", NodeArg, status),
        ToolSpec("upstream", "Every node this one depends on, nearest first, with its reported status.",
                 NodeArg, upstream),
        ToolSpec("precedents", "Resolved cases that look similar, with how each was fixed.", Query, precedents),
        ToolSpec(TERMINAL, "Submit the diagnosis. Call this once, when you are confident.", Submission,
                 submit_diagnosis),
    ]


def task(incident: dict) -> str:
    return (f"Incident {incident.get('id')}: {incident.get('symptom')}, reported at "
            f"{incident['symptom_node']}. Find what actually broke and submit a diagnosis.")


# ------------------------------------------------------------------ the model

class ChatModel(Protocol):
    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        """-> the assistant message: {"tool_calls": [...]} or {"content": "..."}"""


def _rate_limited(r) -> bool:
    """A hosted tier saying "slow down" (429) or "busy" (503). Not the model failing."""
    return r.status_code in (429, 503)


def _unparsable(r) -> bool:
    """The provider could not turn the model's output into a tool call, and answered 400 instead of a
    message. Seen with gpt-oss-120b on Groq. Nothing was executed, so the same question can be asked
    again, and a sampled model usually gets it right the next time."""
    return r.status_code == 400 and "could not be parsed" in (getattr(r, "text", "") or "")


def _retry_after(r) -> float | None:
    """How long the provider asked us to wait, in seconds. Groq sends it as a number of seconds, and
    says the same thing in prose ("Please try again in 7m2.5s") when the header is missing."""
    header = r.headers.get("retry-after") if hasattr(r, "headers") else None
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", getattr(r, "text", "") or "")
    return int(m.group(1) or 0) * 60 + float(m.group(2)) if m else None


def _too_big_to_ever_fit(r) -> bool:
    """One request that needs more tokens than the whole per-minute budget. No amount of waiting makes
    it fit, so retrying is a slow way of failing. Groq spells this out: "Limit 8000, ... Requested 9000"."""
    m = re.search(r"Limit (\d+),.*?Requested (\d+)", (getattr(r, "text", "") or ""), re.S)
    return bool(m) and int(m.group(2)) > int(m.group(1))


def _wait(r, attempt: int, max_wait: float) -> float | None:
    """Seconds to sleep before asking again, or None to stop trying.

    A provider that asks for longer than we are willing to wait, or that cannot fit this request into
    its budget at all, is telling us to change something rather than to be patient.
    """
    if not _rate_limited(r):
        return 0.5
    if _too_big_to_ever_fit(r):
        return None
    asked = _retry_after(r)
    if asked is None:
        return min(2 ** attempt * 2, max_wait)
    return asked if asked <= max_wait else None

def _note(r, pause: float) -> None:
    """A long wait with no output looks like a hang, so say what is happening."""
    if pause >= 5:
        why = "rate limited" if _rate_limited(r) else "output could not be parsed"
        print(f"  [{why}, waiting {pause:.0f}s]", file=sys.stderr, flush=True)


def _raise_with_body(r) -> None:
    """Raise with what the provider actually said: a hosted API answering 400 names the part of the
    request it rejected, and a bare status line turns that into guesswork."""
    import httpx

    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        body = (getattr(r, "text", "") or "")[:600]
        try:
            request = e.request
        except RuntimeError:            # never sent, so there is no request to point at
            request = None
        raise httpx.HTTPStatusError(f"{e}\n{body}" if body else str(e), request=request, response=e.response) from None


class OpenAICompatChat:
    """Chat completions with tool calling over HTTP. Defaults to a local Ollama, no API key."""

    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None,
                 timeout: float | None = None):
        self.model = model or os.environ.get("TRIAGE_MODEL", "qwen3:8b")
        self.base_url = (base_url or os.environ.get("TRIAGE_BASE_URL", "http://localhost:11434/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("TRIAGE_API_KEY", "not-needed-for-local")
        # Per request. A small local model thinking at length on a laptop can take minutes, so it is
        # configurable; when it is exceeded the step treats the model as unavailable rather than failing.
        self.timeout = timeout or float(os.environ.get("TRIAGE_TIMEOUT", "300"))
        self.retries = int(os.environ.get("TRIAGE_RETRIES", "6"))
        self.max_wait = float(os.environ.get("TRIAGE_MAX_WAIT", "180"))

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        import httpx  # imported here so the rules-only path has no dependency on it

        body = {"model": self.model, "messages": messages, **({"tools": tools} if tools else {}),
                "temperature": 0.0, "stream": False}
        for attempt in range(self.retries + 1):
            r = httpx.post(f"{self.base_url}/chat/completions", timeout=self.timeout,
                           headers={"Authorization": f"Bearer {self.api_key}"}, json=body)
            if attempt < self.retries and (_rate_limited(r) or _unparsable(r)):
                pause = _wait(r, attempt, self.max_wait)
                if pause is not None:
                    _note(r, pause)
                    time.sleep(pause)
                    continue
            _raise_with_body(r)
            return r.json()["choices"][0]["message"]
        _raise_with_body(r)
        return r.json()["choices"][0]["message"]


# ------------------------------------------------------------------ the investigation

@dataclass
class Investigation:
    submitted: bool = False
    category: str | None = None
    fix_target: str | None = None
    cause: str = ""
    reason: str = ""
    plan: list[str] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)   # every call, its arguments, and what came back
    answer: str = ""


def run_tool(spec: ToolSpec, raw_args: Any) -> tuple[bool, str]:
    """-> (ok, text the model sees). Bad arguments and rejected answers are errors, not crashes."""
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        return True, spec.fn(**spec.args(**args).model_dump())
    except json.JSONDecodeError:
        return False, "ERROR: arguments were not valid JSON"
    except ValidationError as e:
        return False, f"ERROR: invalid arguments: {e.errors()[0]['msg']} ({e.errors()[0]['loc']})"
    except ValueError as e:
        return False, f"ERROR: {e}"


def investigate(incident: dict, dag: PipelineDAG, library: CaseLibrary, model: ChatModel,
                max_steps: int = 14, specs: list[ToolSpec] | None = None, instructions: str = INSTRUCTIONS,
                task_text: str | None = None) -> Investigation:
    """`specs`, `instructions` and `task_text` let the same loop run over a different set of read-only
    tools (triage/warehouse/investigate.py runs it against a real warehouse). The terminal tool keeps
    its name, and the first valid call to it still ends the investigation."""
    specs = {t.name: t for t in (specs if specs is not None else tools(dag, library, incident))}
    schemas = [t.schema() for t in specs.values()]
    messages: list[dict] = [{"role": "system", "content": instructions},
                            {"role": "user", "content": task_text or task(incident)}]
    out = Investigation()
    for n in range(max_steps):
        try:
            msg = model.chat(messages, schemas)
        except Exception as e:
            # A model that times out or is unreachable has not diagnosed anything. That is recorded and the
            # investigation ends without a submission, which the reviewer sends to a person. It is never a
            # crash of the whole graph, and never a guess.
            out.steps.append({"tool": "-", "args": {}, "ok": False,
                              "result": f"model unavailable ({type(e).__name__}): no diagnosis"})
            out.answer = f"model unavailable ({type(e).__name__})"
            break
        calls = msg.get("tool_calls") or []
        if not calls:
            out.answer = re.sub(r"<think>.*?</think>", "", msg.get("content") or "", flags=re.S).strip()
            break
        call = calls[0]                             # one action per turn keeps the log readable
        name, raw = call["function"]["name"], call["function"].get("arguments") or {}
        cid = call.get("id") or f"call_{n}"
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": name, "arguments": raw if isinstance(raw, str)
                                                         else json.dumps(raw)}}]})
        spec = specs.get(name)
        ok, text = run_tool(spec, raw) if spec else (False, f"ERROR: no tool named '{name}'. Tools: {', '.join(specs)}")
        args = raw if isinstance(raw, dict) else _loads(raw)
        out.steps.append({"tool": name, "args": args, "ok": ok, "result": text[:300]})
        messages.append({"role": "tool", "tool_call_id": cid, "content": text})
        if ok and name == TERMINAL:
            out.submitted = True
            out.category, out.fix_target = args.get("category"), args["fix_target"]
            out.cause, out.reason = args.get("cause") or "", (args.get("reason") or "").strip()
            out.plan = derive_plan(dag, out.fix_target)
            break                                   # the first valid answer stands
    return out


def _loads(raw: str) -> dict:
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {"_unparsed": raw}
