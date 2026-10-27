"""The reward function must accept exactly what TRL passes to it."""
from rtlrl.reward import verilog_reward
from rtlrl.tbgen import make_testbench, read_ports
from rtlrl.trace_grader import simulate

REF = "module inv(input a, output y); assign y = ~a; endmodule"
WRONG = "module inv(input a, output y); assign y = a; endmodule"


def test_reward_with_trl_chat_format():
    tb = make_testbench(*read_ports(REF))
    trace = simulate(REF, tb)
    completions = [[{"role": "assistant", "content": f"[BEGIN]\n{code}\n[DONE]"}] for code in (REF, WRONG)]
    rewards = verilog_reward(completions, tb=[tb, tb], ref_trace=[trace, trace], prompts=None)
    assert rewards == [1.0, 0.0]
