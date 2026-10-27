"""Load the training pool and re-record every reference with THIS machine's simulator."""
import json
import os
from concurrent.futures import ThreadPoolExecutor

from .trace_grader import is_weak, simulate


def load_pool(path, recompute=True):
    """Rows of the pool. With recompute=True (the default), reference outputs are
    re-simulated here, so references and candidates always use the same simulator."""
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    if not recompute:
        return rows
    with ThreadPoolExecutor(os.cpu_count()) as ex:
        traces = list(ex.map(lambda r: simulate(r["ref"], r["tb"]), rows))
    kept = []
    for row, trace in zip(rows, traces):
        if isinstance(trace, list) and not is_weak(trace):
            row["ref_trace"] = trace
            kept.append(row)
    return kept
