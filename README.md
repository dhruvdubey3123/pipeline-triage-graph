# pipeline-triage-graph

Multi-agent incident triage for data pipelines. When a report looks wrong, a tracer agent walks the
pipeline's dependency graph upstream from the symptom, rules out healthy branches, and stops at the
node that actually broke. Specialist agents then check the data, find similar past cases, classify
the failure and draft a fix. A separate reviewer checks the proposal, and **nothing changes until a
person approves it**.

The root cause can be found two ways. The **tracer** walks the graph with fixed rules. The
**investigator** is an LLM agent (local Qwen3 8B through Ollama by default) that walks the same graph
itself through read-only tools and submits a diagnosis. When the investigator is on, the tracer runs
after it as a cross-check, and if the two disagree on the root cause the incident goes to a person
with both traces rather than to the approval gate.

This is an open-source rebuild of the pattern behind a triage agent I built during my internship at
Allegis Global Solutions on Snowflake Cortex. None of that code or data is here. This version is
written from scratch, runs anywhere, and ships with its own eval suite.

![Triage demo: a schema change breaks the pipeline, the tracer walks upstream to the root cause, a person approves the fix](docs/demo.gif)

Try it yourself in the [live demo on my portfolio](https://dhruvdubey3123.github.io/#triage): break any box, run triage, approve or reject the fix.

## An LLM agent that walks the graph itself

```
intake -> investigator (LLM, read-only tools) -> tracer (rules, cross-check) -> ... -> reviewer -> [human gate]
```

`triage/investigator.py` gives the model five tools and nothing that writes:

| Tool | What it returns |
|---|---|
| `pipeline()` | every node, its kind, what it reads from, and which tables are protected |
| `status(node)` | what the incident reports about that node: failed, stale, changed, ok |
| `upstream(node)` | every node it depends on, nearest first, each with its status |
| `precedents(query)` | the most similar resolved cases and how each was fixed |
| `submit_diagnosis(category, fix_target, cause, reason)` | its answer |

The model decides where to look. Around it:

* **Its answer is checked before anything sees it.** An unknown category, a node that is not in the pipeline, or a protected
  table comes back as an error it can read and correct.
* **The first valid answer stands.** A real run of Qwen3 8B submitted the right procedure three times and then talked itself
  into a wrong one, so a valid submission ends the investigation.
* **It never acts.** The plan is derived from its root cause the same way the rule-based orchestrator does it, the reviewer
  checks it, and the approval gate is unchanged.
* **The tracer checks its work.** If the two name different root causes, the reviewer escalates with both traces. The eval
  reports how many wrong diagnoses still reached the approval gate, which is the number that matters.

```bash
ollama pull qwen3:8b
pip install -e ".[local]"
python -m evals.run_eval --investigator --skip 40 --limit 20   # TRIAGE_MODEL / TRIAGE_BASE_URL to override
```

The same tools and prompt, run inside agent-harness-runtime on incidents 1 to 40, found the root cause on 28 of 40, against 40 of
40 for the tracer ([measured there](https://github.com/dhruvdubey3123/agent-harness-runtime#is-the-model-any-good-at-it)). The run
above uses incidents 41 to 60, which the investigator was never adjusted against. CI drives it with a scripted model and needs no
model at all.

## Running inside the agent harness

Triage stops at the human gate. What happens after approval, actually applying the fix, now runs inside
[agent-harness-runtime](https://github.com/dhruvdubey3123/agent-harness-runtime#running-the-triage-agent-inside-the-harness) through this
project's `Executor` interface. Every warehouse action is logged before and after it runs, so if the process is killed halfway
through a fix, a new process finishes it without re-running a load or posting the ticket comment twice. Across the 75 approvable
incidents in this repo's eval set, crashed at random points: restarting from scratch duplicated an action in 61 of 75 fixes;
resuming through the harness duplicated none and applied every action exactly once in all 75.

## In plain words

A data pipeline is a chain of jobs that move data from one place to another. When one job breaks, a dashboard
several steps downstream goes wrong, and someone has to work out which job actually failed. This project does
that detective work automatically.

* It starts at the broken report and walks upstream, skipping healthy jobs, until it finds the real culprit.
* It searches past incidents for similar cases and drafts a fix.
* It never applies the fix on its own. A person approves or rejects it first.
* When the evidence is ambiguous, it hands the case to a person instead of guessing.

It is my own version of the triage agent I worked on at Allegis, with tests and evals that block a bad change from merging.

```
intake → [investigator] → tracer → analyst → precedent → classifier → orchestrator → reviewer
                                                                      │
                                       escalate ◄── blocked ──────────┤
                                          │                           └── passed ──► [ human gate ]
                                          │                                            │       │
                                          │                               reject ◄─────┘       └─► approve → apply
                                          ▼                                                              │
                                        learn ◄──────────────────────────────────────────────────────────┘
```

## What each agent does

| Node | Job | Notes |
|---|---|---|
| `intake` | Validate the incident and the symptom node | |
| `investigator` | LLM agent: walk the graph with read-only tools and submit a diagnosis | Only when a model is configured. Its answer drives the plan; the tracer cross-checks it |
| `tracer` | Walk upstream through unhealthy nodes only; rule out healthy branches | Pure graph logic, no LLM. Finds the fix target and any upstream contract change that caused it, or cross-checks the investigator |
| `analyst` | Run data checks on the suspect nodes | `DataProbe` protocol. Default reads recorded observations; plug in a warehouse probe (for example semantic-view queries) |
| `precedent` | Find similar resolved cases | TF-IDF over a YAML case library. Swap for a vector store or Cortex Search |
| `classifier` | Assign a failure category | Rules by default; optional Claude call, validated against allowed categories |
| `orchestrator` | Draft the fix and the re-run plan | Plan is computed from the DAG in topological order |
| `reviewer` | Block unsafe or uncertain proposals | Protected nodes, unknown category, ambiguous trace, low confidence, plan touching the wrong node, investigator and tracer disagreeing |
| `gate` | Human approval | Graph is compiled with `interrupt_before=["gate"]`, so a run always stops here |
| `apply` | Execute the plan | `DryRunExecutor` by default: records what it would do, does nothing |
| `learn` | Append the case to the library | Next similar failure starts from this precedent |

By default the tracer decides the root cause from structure, not from a model, and the eval numbers below are that path.
With the investigator on, a model decides and the tracer checks it. Either way nothing is applied before a person approves.

## Run it

```bash
pip install -e ".[dev]"
pytest -q                                   # 17 tests
python -m evals.run_eval --gate             # eval suite with CI thresholds
uvicorn triage.api:app --reload             # HTTP API on :8000
python -m triage.mcp_server                 # MCP server over stdio
```

Open an incident, then approve or reject it:

```bash
curl -s localhost:8000/incidents -H 'content-type: application/json' -d @examples/incident_schema_drift.json
curl -s localhost:8000/incidents/INC-demo-1/decision -H 'content-type: application/json' -d '{"approve": true, "reviewer": "me"}'
```

What the example run logs before it stops at the gate:

```
      intake  INC-demo-1: Fill-rate tile reads 0% for every client since Tuesday (at rpt_workforce)
      tracer  fct_spend: ruled out, healthy (ok)
      tracer  sp_merge_cases: ruled out, healthy (ok)
      tracer  sp_fill_rate: failed (SQL compilation error: invalid identifier 'REQ_ID')
      tracer  load_timesheets: changed (column REQ_ID renamed to REQUISITION_ID)
      tracer  root cause at sp_fill_rate, caused upstream by load_timesheets
   precedent  CASE-0001 (schema_drift, score 0.417); CASE-0002 (duplicate_merge, score 0.071)
  classifier  schema_drift (schema_drift: 'invalid identifier')
orchestrator  Update sp_fill_rate to match the changed upstream contract (cause: load_timesheets); ...
    reviewer  passed, waiting for human approval
```

### Use it from Claude Code or Claude Desktop (MCP)

```json
{ "mcpServers": { "pipeline-context": { "command": "python", "args": ["-m", "triage.mcp_server"],
  "env": { "TRIAGE_PIPELINE": "/path/to/your/pipeline.yaml" } } } }
```

Tools: `list_nodes`, `get_upstream`, `get_downstream`, `trace_failure`, `search_precedents`. All read-only.

### Optional LLM mode

For the LLM investigator that walks the graph itself, see [above](#an-llm-agent-that-walks-the-graph-itself). The older,
narrower mode below asks Claude only for the category and the wording of the fix:

```bash
pip install -e ".[llm]"
export ANTHROPIC_API_KEY=... TRIAGE_USE_LLM=1   # TRIAGE_MODEL to override the model id
```

## Evaluation

`evals/generate.py` builds labeled incidents from the example DAG by injecting a known fault, marking
everything downstream stale, and adding noise: healthy nodes reported explicitly, a red-herring table
that is late for an unrelated reason, and sometimes a second independent failure (where the right move
is to escalate). `evals/run_eval.py` scores every incident at the gate, before any human decision.

Two sets of 100, both committed under `evals/data/`:

* **clean**: one fault, clear error text
* **noisy**: 35% of incidents also carry a misleading error line from a different failure mode, the way real logs often do

Results from the current commit, offline rules mode, no LLM:

| Metric | Clean | Noisy |
|---|---|---|
| Root cause node correct | 1.00 | 1.00 |
| Upstream cause correct | 1.00 | 1.00 |
| Category correct | 1.00 | 0.81 |
| Right action (propose vs escalate) | 1.00 | 0.95 |
| Escalation recall (should escalate, did) | 1.00 | 0.81 |
| False escalation rate | 0.00 | 0.00 |
| Nodes checked per incident (of 12) | 7.06 | 7.23 |
| p95 latency per incident | 10.8 ms | 12.2 ms |

How to read this honestly:

* **The clean set is a spec test, not an accuracy claim.** I wrote the generator and the rules, so a
  perfect score means the logic does what it was designed to do. It is a regression gate.
* **The noisy set is where it gets interesting.** Structure-based tracing stays correct because it
  never reads error text. Classification does not: a misleading log line fools the rules, and about one
  in five failures that should escalate gets a confident proposal instead. That is exactly why the human
  gate exists, and it is the gap the LLM classifier is meant to close. Measuring LLM mode on the noisy
  set is the next step.
* **These are synthetic incidents on a 12-node example DAG.** They say nothing about failure rates in a
  real warehouse.
* The suite has already earned its keep: an early version of the tracer missed the case where one
  schema change broke two procedures at once. The eval caught it before any test did.

CI (`.github/workflows/ci.yml`) runs the tests and fails the build if either set drops below the floors in
`evals/thresholds*.yaml`.

## Deploy

`Dockerfile` builds a slim non-root image. `k8s/deployment.yaml` has a Deployment with readiness and
liveness probes on `/healthz`, a ConfigMap, a Service, and a HorizontalPodAutoscaler (2 to 6 replicas on CPU).

## Bring your own pipeline

Describe your DAG in YAML (see `examples/pipeline.yaml`), list the nodes the fixer must never write to
under `protected`, and send incidents with a `node_status` map. In a real deployment you would build
the YAML from warehouse lineage and fill `node_status` from your orchestrator's run history.

## Layout

```
triage/dag.py         DAG, tracer, refresh ordering
triage/knowledge.py   classifier rules, case library search, fix templates, confidence heuristic
triage/graph.py       LangGraph state graph with the human gate
triage/service.py     run and resume incidents
triage/api.py         FastAPI
triage/mcp_server.py  MCP server
triage/llm.py         optional Claude calls
evals/                generator, runner, thresholds, committed data
```

MIT licensed.
