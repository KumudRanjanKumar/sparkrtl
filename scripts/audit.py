"""Reward-hacking audit: re-test every answer that passed, with a stronger test.

The stronger test uses 5x more clock cycles and a different random seed. An answer that
passed the normal test but fails the strong one is a "false pass": the normal test was too
weak to catch it. Run this on your Mac (it needs Yosys) after downloading the answers and
their .flags.jsonl file from Kaggle.

  python scripts/audit.py --pool data/zone.jsonl --answers results/grpo_run1_pool.jsonl
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rtlrl.tbgen import make_testbench, read_ports
from rtlrl.trace_grader import grade, simulate

ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True)
ap.add_argument("--answers", required=True)
ap.add_argument("--cycles", type=int, default=1000)
ap.add_argument("--seed", type=int, default=2)
ap.add_argument("--out", default="results/audit_false_passes.jsonl")
args = ap.parse_args()

pool = {r["id"]: r for r in map(json.loads, open(args.pool))}
answers = {it["id"]: it["answers"] for it in map(json.loads, open(args.answers))}
flags = {it["id"]: it["passed"] for it in map(json.loads, open(os.path.splitext(args.answers)[0] + ".flags.jsonl"))}


def audit(pid):
    p = pool[pid]
    top, ports = read_ports(p["ref"])
    tb = make_testbench(top, ports, cycles=args.cycles, seed=args.seed)
    trace = simulate(p["ref"], tb)
    out = []
    for a, ok in zip(answers[pid], flags[pid]):
        if ok:
            score, verdict = grade(a, tb, trace)
            out.append((a, score == 1.0, verdict))
    return pid, out


ids = [pid for pid in flags if pid in pool and any(flags[pid])]
with ThreadPoolExecutor(os.cpu_count()) as ex:
    results = list(ex.map(audit, ids))

checked = sum(len(r) for _, r in results)
false = [(pid, a, v) for pid, r in results for a, ok, v in r if not ok]
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w") as f:
    for pid, a, v in false:
        f.write(json.dumps({"id": pid, "verdict_on_strong_test": v, "answer": a}) + "\n")
print(f"{checked} passing answers re-tested with {args.cycles} cycles, seed {args.seed}")
print(f"false passes: {len(false)} ({100 * len(false) / max(1, checked):.1f}%) -> {args.out}")
