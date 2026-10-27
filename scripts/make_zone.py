"""Keep only the problems that can teach: the base model solves them sometimes, not always.

  python scripts/make_zone.py --pool data/candidates.jsonl --answers results/base_pool.jsonl

Run grade.py on the same answers first. Writes data/zone.jsonl (problems for GRPO)
and data/rft.jsonl (correct answers for fine-tuning).
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rtlrl.extract import extract_code

ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True)
ap.add_argument("--answers", required=True)
ap.add_argument("--zone-out", default="data/zone.jsonl")
ap.add_argument("--rft-out", default="data/rft.jsonl")
ap.add_argument("--per-problem", type=int, default=2, help="most correct answers kept per problem")
args = ap.parse_args()

pool = {r["id"]: r for r in map(json.loads, open(args.pool))}
answers = {it["id"]: it["answers"] for it in map(json.loads, open(args.answers))}
flags = {it["id"]: it["passed"] for it in map(json.loads, open(os.path.splitext(args.answers)[0] + ".flags.jsonl"))}

hist = Counter(sum(p) for p in flags.values())
n = max(len(p) for p in flags.values())
print(f"problems by number of correct answers out of {n}:")
for c in range(n + 1):
    print(f"  {c:2d} correct: {hist.get(c, 0)}")
zone = [pid for pid, p in flags.items() if 0 < sum(p) < len(p)]
print(f"learnable zone (some right, some wrong): {len(zone)} of {len(flags)} problems")

kept = 0
with open(args.zone_out, "w") as zf, open(args.rft_out, "w") as rf:
    for pid in zone:
        zf.write(json.dumps(pool[pid]) + "\n")
        seen = set()
        for a, ok in zip(answers[pid], flags[pid]):
            code = extract_code(a)
            if ok and code not in seen and len(seen) < args.per_problem:
                seen.add(code)
                rf.write(json.dumps({"id": pid, "spec": pool[pid]["spec"],
                                     "answer": f"[BEGIN]\n{code}\n[DONE]"}) + "\n")
                kept += 1
print(f"wrote {args.zone_out} and {args.rft_out} ({kept} correct answers)")
