"""Honest numbers: pass@k, and a confidence interval for "model B beats model A"."""
import random
from math import comb


def pass_at_k(n: int, c: int, k: int) -> float:
    """Chance that at least one of k answers is correct, given c correct out of n samples."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def mean_pass_at_k(counts, k):
    """counts: list of (n, c), one per problem."""
    return sum(pass_at_k(n, c, k) for n, c in counts) / len(counts)


def paired_bootstrap(a, b, iters=10000, seed=0):
    """a, b: per-problem pass@1 of models A and B on the SAME problems, in the same order.

    Resamples problems with replacement. Returns (mean gain, low, high) where
    [low, high] is a 95% confidence interval for B minus A. If low > 0, the gain is
    unlikely to be luck in which problems happened to be in the test set.
    """
    assert len(a) == len(b) and a
    diffs = [y - x for x, y in zip(a, b)]
    n = len(diffs)
    rng = random.Random(seed)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return sum(diffs) / n, means[int(0.025 * iters)], means[int(0.975 * iters) - 1]
