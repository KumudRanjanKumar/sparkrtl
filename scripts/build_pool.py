"""Build the training pool from VeriThoughts (MIT licence).

For every problem: clean the spec, drop it if it looks like a test problem,
generate a testbench from the reference design, record the reference outputs,
and drop problems whose recording cannot tell right from wrong.

  python scripts/build_pool.py --source data/vt_consistent.json \
      --verilogeval verilog-eval/dataset_spec-to-rtl --fresh fresh/ --out data/pool.jsonl
"""
import argparse
import csv
import glob
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rtlrl.decontam import Index
from rtlrl.extract import extract_code
from rtlrl.safety import is_safe
from rtlrl.tbgen import make_testbench, read_ports
from rtlrl.trace_grader import is_weak, simulate

ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True)
ap.add_argument("--verilogeval", required=True)
ap.add_argument("--fresh", default=None, help="your own test set, same file naming as VerilogEval")
ap.add_argument("--out", default="data/pool.jsonl")
ap.add_argument("--report", default="data/decontam_report.csv")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--spec-max", type=float, default=0.2)
ap.add_argument("--code-max", type=float, default=0.4)
ap.add_argument("--max-trace-chars", type=int, default=40000)
args = ap.parse_args()

tests = []
for d in [args.verilogeval] + ([args.fresh] if args.fresh else []):
    for p in sorted(glob.glob(os.path.join(d, "*_prompt.txt"))):
        stem = p[:-len("_prompt.txt")]
        tests.append((os.path.basename(stem), open(p).read(), open(stem + "_ref.sv").read()))
index = Index(tests)
print(f"{len(tests)} test problems indexed")

rows = json.load(open(args.source))[:args.limit]


def clean_spec(s):
    return s.replace("Please start your Verilog code with CODE BEGIN and end with CODE END.", "").strip()


def build(i):
    row = rows[i]
    spec, ref = clean_spec(row["instruction"]), extract_code(row["output"])
    info = {"id": f"vt{i:05d}"}
    if not ref:
        return "no_reference", info
    if not is_safe(ref):
        return "reference_uses_banned_feature", info
    s, sname, c, cname = index.closest(spec, ref)
    info.update(spec_overlap=round(s, 3), spec_match=sname, code_jaccard=round(c, 3), code_match=cname)
    if s > args.spec_max or c > args.code_max:
        return "too_close_to_a_test_problem", info
    ports = read_ports(ref)
    if not ports:
        return "yosys_could_not_read", info
    tb = make_testbench(*ports)
    if not tb:
        return "unsupported_interface", info
    trace = simulate(ref, tb)
    if isinstance(trace, str):
        return "reference_" + trace, info
    if is_weak(trace):
        return "weak_check", info
    if sum(map(len, trace)) > args.max_trace_chars:
        return "outputs_too_wide", info
    info.update(spec=spec, top=ports[0], ref=ref, tb=tb, ref_trace=trace)
    return "kept", info


t0 = time.time()
with ThreadPoolExecutor(os.cpu_count()) as ex:
    results = list(ex.map(build, range(len(rows))))

counts = Counter(v for v, _ in results)
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w") as f:
    for verdict, info in results:
        if verdict == "kept":
            f.write(json.dumps(info) + "\n")
with open(args.report, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "verdict", "spec_overlap", "spec_match", "code_jaccard", "code_match"])
    for verdict, info in results:
        w.writerow([info["id"], verdict, info.get("spec_overlap", ""), info.get("spec_match", ""),
                    info.get("code_jaccard", ""), info.get("code_match", "")])

print(f"{len(rows)} problems in {time.time() - t0:.0f} s")
for k, v in counts.most_common():
    print(f"  {k:32s} {v}")
