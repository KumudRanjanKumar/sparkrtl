"""Sanity check: every VerilogEval reference solution must pass our VerilogEval grader."""
import glob, os, sys, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rtlrl.ve_grader import grade_verilogeval

root = sys.argv[1]  # path to verilog-eval/dataset_spec-to-rtl
probs = sorted(p[:-len("_prompt.txt")] for p in glob.glob(os.path.join(root, "*_prompt.txt")))

def check(p):
    ref = open(p + "_ref.sv").read()
    cand = ref.replace("module RefModule", "module TopModule")
    return os.path.basename(p), grade_verilogeval(cand, open(p + "_test.sv").read(), ref)

t = time.time()
with ThreadPoolExecutor(os.cpu_count()) as ex:
    results = list(ex.map(check, probs))
bad = [n for n, ok in results if not ok]
print(f"{len(results) - len(bad)}/{len(results)} reference solutions pass in {time.time() - t:.1f} s")
for n in bad:
    print("FAIL", n)
