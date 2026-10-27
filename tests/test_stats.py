from rtlrl.stats import paired_bootstrap, pass_at_k


def test_pass_at_k_examples():
    assert abs(pass_at_k(10, 3, 1) - 0.3) < 1e-12
    assert abs(pass_at_k(10, 3, 5) - (1 - 21 / 252)) < 1e-12
    assert pass_at_k(8, 0, 1) == 0.0
    assert pass_at_k(8, 8, 1) == 1.0


def test_bootstrap_sees_a_real_gain():
    a = [0.2] * 100
    b = [0.3] * 100
    gain, low, high = paired_bootstrap(a, b, iters=2000)
    assert abs(gain - 0.1) < 1e-9 and low > 0


def test_bootstrap_does_not_invent_a_gain():
    a = [0.0, 1.0] * 50
    b = [1.0, 0.0] * 50
    gain, low, high = paired_bootstrap(a, b, iters=2000)
    assert gain == 0 and low < 0 < high
