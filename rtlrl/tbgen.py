"""Turn any reference design into an automatic checker.

Idea: we never need a hand-written testbench. We drive the same random
inputs into the reference design and into the model's design, record every
output, and compare the two recordings. The reference and the candidate are
simulated separately, so the candidate can never see the expected answers.
"""
import json
import os
import re
import tempfile

from .sim import YOSYS, run

_SIMPLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
TOKEN = "@@TOKEN@@"  # replaced by a fresh secret every time a design is graded


def read_ports(ref_code: str):
    """Use Yosys to find the top module and its ports: (top, [(name, dir, width)])."""
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "ref.v"), "w") as f:
            f.write(ref_code)
        rc, _ = run([YOSYS, "-q", "-p",
                     "read_verilog -sv ref.v; hierarchy -auto-top; proc; write_json ports.json"], d, 60)
        if rc != 0:
            return None
        with open(os.path.join(d, "ports.json")) as f:
            modules = json.load(f)["modules"]
    tops = [n for n, m in modules.items() if m.get("attributes", {}).get("top")]
    if len(tops) != 1:
        return None
    top = tops[0]
    ports = [(n, p["direction"], len(p["bits"])) for n, p in modules[top]["ports"].items()]
    return top, ports


def _is_clock(name, width):
    return width == 1 and re.search(r"clk|clock", name, re.I) is not None


def _is_reset(name, width):
    return width == 1 and re.search(r"rst|reset|clr|clear", name, re.I) is not None


def _active_low(name):
    n = name.lower()
    return n.endswith("n") or n.endswith("_b") or n.startswith("n")


def _random_expr(width):
    words = (width + 31) // 32
    parts = ", ".join(["$random(rtlrl_seed)"] * words)
    return "{" + parts + "}"


def make_testbench(top, ports, cycles=200, seed=1):
    """Write a testbench for `top`. Returns None if the interface is unsupported."""
    if any(d == "inout" for _, d, _ in ports):
        return None
    if not all(_SIMPLE_NAME.match(n) for n, _, _ in ports):
        return None
    ins = [(n, w) for n, d, w in ports if d == "input"]
    outs = [(n, w) for n, d, w in ports if d == "output"]
    if not outs:
        return None
    clocks = [n for n, w in ins if _is_clock(n, w)]
    if len(clocks) > 1:
        return None
    clk = clocks[0] if clocks else None
    resets = [n for n, w in ins if n != clk and _is_reset(n, w)]
    data = [(n, w) for n, w in ins if n != clk and n not in resets]

    def decl(kind, n, w):
        return f"  {kind} [{w - 1}:0] {n};" if w > 1 else f"  {kind} {n};"

    lines = ["`timescale 1ns/1ps", "module rtlrl_tb;"]
    lines += [decl("reg", n, w) for n, w in ins]
    lines += [decl("wire", n, w) for n, w in outs]
    lines += ["  integer rtlrl_seed;", "  integer rtlrl_i;"]
    conns = ", ".join(f".{n}({n})" for n, _, _ in ports)
    lines.append(f"  {top} dut ({conns});")
    fmt = " ".join(["%b"] * len(outs))
    args = ", ".join(n for n, _ in outs)

    lines += ["  initial begin", f"    rtlrl_seed = {seed};"]
    for n, w in ins:
        on = "1'b0" if _active_low(n) else "1'b1"
        lines.append(f"    {n} = {on if n in resets else 0};")
    lines.append(f"    for (rtlrl_i = 0; rtlrl_i < {cycles}; rtlrl_i = rtlrl_i + 1) begin")
    lines.append("      #2;")
    for n, w in data:
        lines.append(f"      {n} = {_random_expr(w)};")
    for n in resets:
        on, off = ("1'b0", "1'b1") if _active_low(n) else ("1'b1", "1'b0")
        # hold reset for the first 3 cycles, then pulse it at random (1 cycle in 16)
        lines.append(f"      {n} = (rtlrl_i < 3) ? {on} : "
                     f"((($random(rtlrl_seed) & 15) == 0) ? {on} : {off});")
    if clk:
        lines += [f'      #2 $display("{TOKEN} %0d a {fmt}", rtlrl_i, {args});',
                  f"      #1 {clk} = 1;",
                  f'      #3 $display("{TOKEN} %0d b {fmt}", rtlrl_i, {args});',
                  f"      #2 {clk} = 0;"]
    else:
        lines += [f'      #3 $display("{TOKEN} %0d a {fmt}", rtlrl_i, {args});',
                  "      #5;"]
    lines += ["    end", "    $finish;", "  end", "endmodule", ""]
    return "\n".join(lines)
