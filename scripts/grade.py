"""Grade generated answers and print pass@k.

  python scripts/grade.py --answers results/base_ve.jsonl --verilogeval verilog-eval/dataset_spec-to-rtl
  python scripts/grade.py --answers results/base_pool.jsonl --pool data/candidates.jsonl

Writes <answers>.graded.csv with one row per problem: id, n (answers), c (correct),
and <answers>.flags.jsonl saying which answers passed.
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rtlrl.extract import extract_code
from rtlrl.pool import load_pool
from rtlrl.stats import mean_pass_at_k
from rtlrl.trace_grader import grade
from rtlrl.ve_grader import grade_verilogeval

ap = argparse.ArgumentParser()
ap.add_argument("--answers", required=True)
ap.add_argument("--verilogeval")
ap.add_argument("--pool")
args = ap.parse_args()

items = [json.loads(line) for line in open(args.answers)]

if args.verilogeval:
    def check(item):
        stem = os.path.join(args.verilogeval, item["id"])
        test, ref = open(stem + "_test.sv").read(), open(stem + "_ref.sv").read()
        return ["pass" if grade_verilogeval(extract_code(a), test, ref) else "fail" for a in item["answers"]]
else:
    pool = {r["id"]: r for r in load_pool(args.pool)}
    items = [it for it in items if it["id"] in pool]

    def check(item):
        p = pool[item["id"]]
        return [grade(a, p["tb"], p["ref_trace"])[1] for a in item["answers"]]

with ThreadPoolExecutor(os.cpu_count()) as ex:
    results = list(ex.map(check, items))
verdicts = Counter(v for r in results for v in r)
passed = [[v == "pass" for v in r] for r in results]

rows = [(it["id"], len(p), sum(p)) for it, p in zip(items, passed)]
out = os.path.splitext(args.answers)[0] + ".graded.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "n", "c"])
    w.writerows(rows)
with open(os.path.splitext(args.answers)[0] + ".flags.jsonl", "w") as f:
    for it, p in zip(items, passed):
        f.write(json.dumps({"id": it["id"], "passed": p}) + "\n")

counts = [(n, c) for _, n, c in rows]
n_min = min(n for n, _ in counts)
print(f"{len(rows)} problems, {n_min}+ answers each -> {out}")
for k in (1, 5, 10):
    if k <= n_min:
        print(f"  pass@{k}: {100 * mean_pass_at_k(counts, k):.1f}%")
if verdicts:
    print("  verdicts:", dict(verdicts.most_common()))
