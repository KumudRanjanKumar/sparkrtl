"""Is model B really better than model A? Paired bootstrap over problems.

  python scripts/compare.py results/base_ve.graded.csv results/rft_ve.graded.csv
"""
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rtlrl.stats import paired_bootstrap


def per_problem(path):
    return {r["id"]: int(r["c"]) / int(r["n"]) for r in csv.DictReader(open(path))}


a, b = per_problem(sys.argv[1]), per_problem(sys.argv[2])
ids = sorted(set(a) & set(b))
if len(ids) != len(a) or len(ids) != len(b):
    print(f"warning: comparing only the {len(ids)} problems both files share")
pa, pb = [a[i] for i in ids], [b[i] for i in ids]
gain, low, high = paired_bootstrap(pa, pb)
print(f"A pass@1 {100 * sum(pa) / len(ids):.1f}%   B pass@1 {100 * sum(pb) / len(ids):.1f}%   on {len(ids)} problems")
print(f"gain {100 * gain:+.1f} points, 95% CI [{100 * low:+.1f}, {100 * high:+.1f}]")
print("the gain is real (CI above zero)" if low > 0 else "not proven: the CI includes zero")
