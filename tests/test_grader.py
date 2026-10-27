"""The grader is the most important code in the project. These tests try to fool it."""
import pytest

from rtlrl.extract import extract_code
from rtlrl.tbgen import make_testbench, read_ports
from rtlrl.trace_grader import grade, simulate

REF = """
module counter4 (input clk, input rst, input en, output reg [3:0] q);
  always @(posedge clk)
    if (rst) q <= 4'd0;
    else if (en) q <= q + 4'd1;
endmodule
"""


@pytest.fixture(scope="module")
def problem():
    top, ports = read_ports(REF)
    tb = make_testbench(top, ports)
    trace = simulate(REF, tb)
    assert isinstance(trace, list) and len(trace) == 400
    return tb, trace


def fenced(code):
    return "[BEGIN]\n" + code + "\n[DONE]"


CASES = [
    ("correct design", fenced(REF), 1.0, "pass"),
    ("adds 2 instead of 1", fenced(REF.replace("q + 4'd1", "q + 4'd2")), 0.0, "mismatch"),
    ("ignores enable", fenced(REF.replace("else if (en)", "else")), 0.0, "mismatch"),
    ("asynchronous reset (spec says synchronous)",
     fenced(REF.replace("posedge clk)", "posedge clk or posedge rst)")), 0.0, "mismatch"),
    ("missing semicolon", fenced(REF.replace("q <= 4'd0;", "q <= 4'd0")), 0.0, "compile_error"),
    ("prints a fake pass line",
     fenced(REF.replace("endmodule", 'initial $display("PASS");\nendmodule')), 0.0, "unsafe"),
    ("ends the simulation early",
     fenced(REF.replace("endmodule", "initial #50 $finish;\nendmodule")), 0.0, "unsafe"),
    ("opens a file", fenced(REF.replace("endmodule", 'integer f; initial f = $fopen("x");\nendmodule')),
     0.0, "unsafe"),
    ("builds $display with a macro",
     fenced("`define D(x) $disp``x\n" + REF), 0.0, "unsafe"),
    ("includes a file", fenced('`include "/etc/hosts"\n' + REF), 0.0, "unsafe"),
    ("hangs the simulator", fenced(REF.replace("endmodule", "initial forever #0;\nendmodule")),
     0.0, "timeout"),
    ("prose, no code", "I think you should use a counter.", 0.0, "no_code"),
    ("draft then fix: last copy wins",
     fenced(REF.replace("q + 4'd1", "q + 4'd2")) + "\nOops, fixed:\n" + fenced(REF), 1.0, "pass"),
]


@pytest.mark.parametrize("name,reply,reward,verdict", CASES, ids=[c[0] for c in CASES])
def test_grader(problem, name, reply, reward, verdict):
    tb, trace = problem
    assert grade(reply, tb, trace) == (reward, verdict)


def test_extract_formats():
    body = "module m(input a, output y); assign y = a; endmodule"
    assert extract_code("[BEGIN]\n" + body + "\n[DONE]") == body
    assert extract_code("[BEGIN]\n```verilog\n" + body + "\n```\n[DONE]") == body
    assert extract_code("CODE BEGIN\n" + body + "\nCODE END") == body
    assert extract_code("Here is the module:\n```verilog\n" + body + "\n```") == body
    assert extract_code("Here is the module:\n" + body) == body
