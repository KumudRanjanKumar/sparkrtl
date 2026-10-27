"""The reward function TRL calls during GRPO training."""
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from .trace_grader import grade

_POOL = ThreadPoolExecutor(max_workers=os.cpu_count())
VERDICT_LOG = os.environ.get("RTLRL_VERDICT_LOG")  # optional CSV of verdict counts per batch


def completion_text(c):
    """TRL gives a list of messages for chat prompts, or a plain string otherwise."""
    return c[0]["content"] if isinstance(c, list) else c


def verilog_reward(completions, tb, ref_trace, **kwargs):
    """1.0 if a design's outputs match the reference on every sample, else 0.0.

    `tb` and `ref_trace` are dataset columns; TRL passes them in automatically,
    one entry per completion. Grading runs in parallel on every CPU core.
    """
    texts = [completion_text(c) for c in completions]
    results = list(_POOL.map(grade, texts, tb, ref_trace))
    if VERDICT_LOG:
        counts = Counter(verdict for _, verdict in results)
        with open(VERDICT_LOG, "a") as f:
            f.write(",".join(f"{k}={v}" for k, v in sorted(counts.items())) + "\n")
    return [score for score, _ in results]
