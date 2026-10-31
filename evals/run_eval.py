"""Run the triage graph over the labeled incident set and score it.

    python -m evals.run_eval                 # print metrics, write evals/results.json
    python -m evals.run_eval --gate          # also exit 1 if any threshold in evals/thresholds.yaml fails

Each incident gets a fresh copy of the case library so results do not depend on order.
Human approval is simulated as "approve" so the learn step runs, but scoring happens on
the state at the gate, before any decision.
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path

import yaml

from triage.dag import PipelineDAG
from triage.knowledge import CaseLibrary
from triage.service import TriageService

ROOT = Path(__file__).resolve().parent.parent


def evaluate(data_path: Path, pipeline: Path, cases: Path) -> dict:
    dag = PipelineDAG.from_yaml(pipeline)
    base = CaseLibrary.load(cases)
    rows = [json.loads(l) for l in data_path.read_text().splitlines() if l.strip()]

    root_hits = cause_hits = cause_total = cat_hits = cat_total = act_hits = 0
    should_esc = esc_caught = should_prop = false_esc = 0
    checked, latencies, failures = [], [], []

    for row in rows:
        label = row.pop("label")
        lib = CaseLibrary(copy.deepcopy(base.cases))
        svc = TriageService(dag, lib)
        t0 = time.perf_counter()
        run_id, v = svc.open(row)
        latencies.append((time.perf_counter() - t0) * 1000)
        escalated = not v["waiting_for_human"]

        ok_root = v.get("fix_target") in label["fix_targets"]
        root_hits += ok_root
        if label["cause"] and not label["ambiguous"]:
            cause_total += 1
            cause_hits += v.get("upstream_cause") == label["cause"]
        if not label["ambiguous"]:
            cat_total += 1
            cat_hits += v.get("category") == label["category"]
        act = "escalate" if escalated else "propose"
        act_hits += act == label["expected_action"]
        if label["expected_action"] == "escalate":
            should_esc += 1
            esc_caught += escalated
        else:
            should_prop += 1
            false_esc += escalated
        checked.append(len(v.get("hops", [])))
        if not ok_root or act != label["expected_action"]:
            failures.append({"id": row["id"], "expected": label, "got": {
                "fix_target": v.get("fix_target"), "category": v.get("category"), "action": act,
                "review_issues": v.get("review_issues")}})
        if v["waiting_for_human"]:
            svc.decide(run_id, approve=True, reviewer="eval")

    n = len(rows)
    return {
        "incidents": n,
        "root_accuracy": round(root_hits / n, 3),
        "cause_accuracy": round(cause_hits / cause_total, 3) if cause_total else None,
        "category_accuracy": round(cat_hits / cat_total, 3) if cat_total else None,
        "action_accuracy": round(act_hits / n, 3),
        "escalation_recall": round(esc_caught / should_esc, 3) if should_esc else None,
        "false_escalation_rate": round(false_esc / should_prop, 3) if should_prop else None,
        "mean_nodes_checked": round(statistics.mean(checked), 2),
        "pipeline_nodes": len(dag.nodes),
        "p50_latency_ms": round(statistics.median(latencies), 1),
        "p95_latency_ms": round(sorted(latencies)[int(0.95 * (n - 1))], 1),
        "mode": "offline rules (no LLM)",
        "failures": failures,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "evals/data/incidents.jsonl"))
    ap.add_argument("--pipeline", default=str(ROOT / "examples/pipeline.yaml"))
    ap.add_argument("--cases", default=str(ROOT / "examples/case_library.yaml"))
    ap.add_argument("--out", default=str(ROOT / "evals/results.json"))
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--thresholds", default=str(ROOT / "evals/thresholds.yaml"))
    a = ap.parse_args()

    res = evaluate(Path(a.data), Path(a.pipeline), Path(a.cases))
    Path(a.out).write_text(json.dumps(res, indent=2) + "\n")
    summary = {k: v for k, v in res.items() if k != "failures"}
    print(json.dumps(summary, indent=2))
    print(f"{len(res['failures'])} incidents scored wrong; see {a.out}")

    if a.gate:
        th = yaml.safe_load(Path(a.thresholds).read_text())
        bad = []
        for k, lim in th.get("min", {}).items():
            if res.get(k) is not None and res[k] < lim:
                bad.append(f"{k}={res[k]} < {lim}")
        for k, lim in th.get("max", {}).items():
            if res.get(k) is not None and res[k] > lim:
                bad.append(f"{k}={res[k]} > {lim}")
        if bad:
            print("EVAL GATE FAILED: " + "; ".join(bad))
            sys.exit(1)
        print("eval gate passed")


if __name__ == "__main__":
    main()
