"""One prompt format for everything: training, baselines and final evaluation.

It is VerilogEval v2's own zero-shot spec-to-rtl format, so your numbers are
comparable with published ones.
"""
SYSTEM = "You are a Verilog RTL designer that only writes code using correct Verilog syntax."
SUFFIX = ("\nEnclose your code with [BEGIN] and [DONE]. Only output the code snippet\n"
          "and do NOT output anything else.\n")


def make_messages(spec: str):
    user = "\nQuestion:\n" + spec.strip() + "\n"
    user = user.rstrip() + "\n" + SUFFIX + "\nAnswer:\n"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
