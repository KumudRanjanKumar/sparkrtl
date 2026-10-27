"""Grade one answer exactly the way VerilogEval v2 does (spec-to-rtl task).

Same compile flags, same file order, same 30 s run limit, same pass rule:
no line containing "error", no TIMEOUT, and "Mismatches: 0 in N samples".
"""
import os
import re
import tempfile

from .sim import IVERILOG, VVP, run

FLAGS = ["-Wall", "-Winfloop", "-Wno-timescale", "-g2012", "-s", "tb"]
_ZERO = re.compile(r"^Mismatches: 0 in \d+ samples$", re.M)


def grade_verilogeval(sample: str, test_sv: str, ref_sv: str) -> bool:
    with tempfile.TemporaryDirectory() as d:
        for name, text in (("sample.sv", sample), ("test.sv", test_sv), ("ref.sv", ref_sv)):
            with open(os.path.join(d, name), "w") as f:
                f.write(text)
        rc, log = run([IVERILOG, *FLAGS, "-o", "sim", "sample.sv", "test.sv", "ref.sv"], d, 60)
        if rc != 0:
            return False
        rc, out = run([VVP, "sim"], d, 30)
        log += out
    if "error" in log or "TIMEOUT" in log:
        return False
    return bool(_ZERO.search(log))
