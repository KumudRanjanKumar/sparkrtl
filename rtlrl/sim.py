"""Run the free EDA tools with hard time limits."""
import os
import subprocess

# Point these at your own install if the tools are not on PATH.
IVERILOG = os.environ.get("RTLRL_IVERILOG", "iverilog")
VVP = os.environ.get("RTLRL_VVP", "vvp")
YOSYS = os.environ.get("RTLRL_YOSYS", "yosys")


def run(cmd, cwd, timeout):
    """Run a command. Returns (return_code, output); return_code is None on timeout."""
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           errors="replace", timeout=timeout)
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
