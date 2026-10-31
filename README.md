# pipeline-triage-graph

Multi-agent incident triage for data pipelines. When a report looks wrong, a tracer agent walks the
pipeline's dependency graph upstream from the symptom, rules out healthy branches, and stops at the
node that actually broke. Specialist agents then check the data, find similar past cases, classify
the failure and draft a fix. A separate reviewer checks the proposal, and **nothing changes until a
person approves it**.

This is an open-source rebuild of the pattern behind a triage agent I built during my internship at
Allegis Global Solutions on Snowflake Cortex. None of that code or data is here. This version is
written from scratch, runs anywhere, and ships with its own eval suite.

```
intake → tracer → analyst → precedent → classifier → orchestrator → reviewer
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
| `tracer` | Walk upstream through unhealthy nodes only; rule out healthy branches | Pure graph logic, no LLM. Finds the fix target and any upstream contract change that caused it |
| `analyst` | Run data checks on the suspect nodes | `DataProbe` protocol. Default reads recorded observations; plug in a warehouse probe (for example semantic-view queries) |
| `precedent` | Find similar resolved cases | TF-IDF over a YAML case library. Swap for a vector store or Cortex Search |
| `classifier` | Assign a failure category | Rules by default; optional Claude call, validated against allowed categories |
| `orchestrator` | Draft the fix and the re-run plan | Plan is computed from the DAG in topological order |
| `reviewer` | Block unsafe or uncertain proposals | Protected nodes, unknown category, ambiguous trace, low confidence, plan touching the wrong node |
| `gate` | Human approval | Graph is compiled with `interrupt_before=["gate"]`, so a run always stops here |
| `apply` | Execute the plan | `DryRunExecutor` by default: records what it would do, does nothing |
| `learn` | Append the case to the library | Next similar failure starts from this precedent |

The tracer decides the root cause from structure, not from a model. LLMs are only asked narrow
questions (which category, how to word the fix), and their answers are checked before use.

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
