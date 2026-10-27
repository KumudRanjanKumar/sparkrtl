"""Split the pool once, with a fixed seed: a held-out test set you never train on,
and the candidates you will search for the learnable zone.

  python scripts/split_pool.py --pool data/pool.jsonl --heldout 300 --candidates 3000
"""
import argparse
import json
import random

ap = argparse.ArgumentParser()
ap.add_argument("--pool", default="data/pool.jsonl")
ap.add_argument("--heldout", type=int, default=300)
ap.add_argument("--candidates", type=int, default=3000)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

rows = [json.loads(l) for l in open(args.pool)]
random.Random(args.seed).shuffle(rows)
held, cand = rows[:args.heldout], rows[args.heldout:args.heldout + args.candidates]
for name, part in (("data/heldout.jsonl", held), ("data/candidates.jsonl", cand)):
    with open(name, "w") as f:
        for r in part:
            f.write(json.dumps(r) + "\n")
    print(f"{name}: {len(part)} problems")
