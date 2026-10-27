"""Turn your own test problems into data/fresh.jsonl, and warn about overlap with training data.

Each problem is two files in fresh/, named like VerilogEval's:
  Prob001_pwm4_prompt.txt   the spec (ask for a module named TopModule)
  Prob001_pwm4_ref.sv       your reference solution (module RefModule)

  python scripts/build_fresh.py --fresh fresh --train data/zone.jsonl
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rtlrl.decontam import Index
from rtlrl.tbgen import make_testbench, read_ports
from rtlrl.trace_grader import is_weak, simulate

ap = argparse.ArgumentParser()
ap.add_argument("--fresh", default="fresh")
ap.add_argument("--train", default=None, help="training problems to check against (zone.jsonl)")
ap.add_argument("--out", default="data/fresh.jsonl")
args = ap.parse_args()

index = None
if args.train:
    train = [json.loads(l) for l in open(args.train)]
    index = Index([(r["id"], r["spec"], r["ref"]) for r in train])

ok = 0
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w") as f:
    for p in sorted(glob.glob(os.path.join(args.fresh, "*_prompt.txt"))):
        stem = p[:-len("_prompt.txt")]
        name = os.path.basename(stem)
        spec = open(p).read()
        # the candidate will be called TopModule, so the reference must be too
        ref = open(stem + "_ref.sv").read().replace("module RefModule", "module TopModule")
        ports = read_ports(ref)
        tb = make_testbench(*ports) if ports else None
        trace = simulate(ref, tb) if tb else "unsupported interface"
        if isinstance(trace, str) or is_weak(trace):
            print(f"FIX  {name}: {trace if isinstance(trace, str) else 'weak check (outputs mostly x or constant)'}")
            continue
        if index:
            s, sname, c, cname = index.closest(spec, ref)
            if s > 0.2 or c > 0.4:
                print(f"DROP {name}: too close to training problem {sname or cname} (spec {s:.2f}, code {c:.2f})")
                continue
        f.write(json.dumps({"id": name, "spec": spec, "ref": ref, "tb": tb, "ref_trace": trace}) + "\n")
        ok += 1
print(f"{ok} problems written to {args.out}")
