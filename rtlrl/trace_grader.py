"""Simulate a design with a generated testbench and compare its outputs to the reference."""
import os
import secrets
import tempfile

from .extract import extract_code
from .safety import is_safe
from .sim import IVERILOG, VVP, run
from .tbgen import TOKEN


def simulate(design: str, tb: str):
    """Return the list of output lines, or a string saying why there are none."""
    token = "T" + secrets.token_hex(8)
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "design.v"), "w") as f:
            f.write(design)
        with open(os.path.join(d, "tb.v"), "w") as f:
            f.write(tb.replace(TOKEN, token))
        rc, _ = run([IVERILOG, "-g2012", "-s", "rtlrl_tb", "-o", "sim", "design.v", "tb.v"], d, 20)
        if rc is None:
            return "compile_timeout"
        if rc != 0:
            return "compile_error"
        rc, out = run([VVP, "-n", "sim"], d, 10)
        if rc is None:
            return "timeout"
    prefix = token + " "
    return [line[len(prefix):] for line in out.splitlines() if line.startswith(prefix)]


def traces_match(ref, cand) -> bool:
    """x or z in the reference means "don't care"; everything else must match exactly."""
    if len(ref) != len(cand):
        return False
    for r_line, c_line in zip(ref, cand):
        if len(r_line) != len(c_line):
            return False
        for r, c in zip(r_line, c_line):
            if r not in "xXzZ" and r != c:
                return False
    return True


def grade(reply: str, tb: str, ref_trace):
    """Score one model reply: (reward, verdict). Reward is 1.0 only for a full match."""
    code = extract_code(reply)
    if not code:
        return 0.0, "no_code"
    if not is_safe(code):
        return 0.0, "unsafe"
    trace = simulate(code, tb)
    if isinstance(trace, str):
        return 0.0, trace
    if traces_match(ref_trace, trace):
        return 1.0, "pass"
    return 0.0, "mismatch"


def is_weak(ref_trace) -> bool:
    """A reference trace that cannot tell right from wrong: mostly unknown, or never changes."""
    if not ref_trace:
        return True
    values = [line.split(" ", 2)[2] if line.count(" ") >= 2 else "" for line in ref_trace]
    bits = "".join(values).replace(" ", "")
    if not bits or sum(ch in "xXzZ" for ch in bits) > 0.5 * len(bits):
        return True
    return len(set(values)) == 1
