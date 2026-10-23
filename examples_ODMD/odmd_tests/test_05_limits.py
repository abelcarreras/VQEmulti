"""
Test 5: where the method breaks, and the guards that should catch it.

  * d must be at least the number of components.
  * Close frequencies: exact data resolves them; noise sets a resolution limit.
  * Aliasing: |omega| * dt must stay below pi, otherwise the frequency folds.
  * Invalid input raises a clear error.

    python test_05_limits.py
"""
import sys
import numpy as np

from odmd import ODMD
import signals
from helpers import omegas, match_error, run_tests


def test_delay_dimension_too_small():
    _, sig, truth = signals.complex_exponentials()     # 3 components
    for d, should_work in ((2, False), (3, True), (4, True)):
        m = ODMD(d=d, svd_rank=-1, tol=1e-8).fit(sig)
        err = match_error(-omegas(m, truth["dt"]), truth["energies"])
        print(f"    d={d}: rank={m.operator.rank}, energy err={err:.1e}")
        assert (err < 1e-8) == should_work


def test_close_frequencies():
    """
    Exact data resolves separations far below the Fourier limit 2*pi/T
    (=0.157 here). With 40 dB noise, 0.05 is still resolved but 0.02 is not:
    the two energies collapse into one (error ~ sep/2). That is the practical
    resolution limit for this record length and noise level.
    """
    for sep, resolvable in ((0.1, True), (0.05, True), (0.02, False)):
        _, sig, truth = signals.complex_exponentials(energies=(0.5, 0.5 + sep),
                                                     weights=(0.5, 0.5))
        exact = ODMD(d=100, svd_rank=2, tol=None).fit(sig)
        noisy = ODMD(d=100, svd_rank=2, tol=None).fit(signals.add_noise(sig, 40, seed=0))
        e_ex = match_error(-omegas(exact, 0.1), truth["energies"])
        e_no = match_error(-omegas(noisy, 0.1), truth["energies"])
        print(f"    separation {sep:<5}: exact err={e_ex:.1e}, 40 dB err={e_no:.1e}"
              f"  (relative to sep: {e_no / sep:.1%})")
        assert e_ex < 1e-8
        if resolvable:
            assert e_no < 0.1 * sep, "the two energies should be distinguishable"


def test_aliasing():
    """E*dt > pi: the recovered energy is folded to E - 2*pi/dt."""
    dt = 0.1
    E = 40.0
    _, sig, _ = signals.complex_exponentials(energies=(E,), weights=(1.0,), dt=dt)

    m = ODMD(d=10, svd_rank=0, tol=None, reduced=True).fit(sig)

    #import cmath
    #for i, lam in enumerate(m.eigs):
    #    mu = cmath.log(lam) / dt
    #    print(f"[ODMD] {i}: lambda={lam}, decay={mu.real}, omega={mu.imag}")

    found = -omegas(m, dt)[0]
    aliased = E - 2 * np.pi / dt
    print(
        f"true E={E}, found {found:.6f}, "
        f"expected alias {aliased:.6f}; "
        f"Nyquist limit (max resolvable |E|) = {np.pi / dt:.6f}"
    )
    assert abs(found - aliased) < 1e-8


def test_invalid_inputs():
    _, sig, _ = signals.sin_cos(n=30)
    cases = {
        "d >= len(signal)": lambda: ODMD(d=30).fit(sig),
        "empty input": lambda: ODMD(d=5).fit(np.array([])),
        "NaN in input": lambda: ODMD(d=5).fit(np.r_[sig[:-1], np.nan]),
        "2D multi-row input": lambda: ODMD(d=5).fit(np.vstack([sig, sig, sig])),
        "bad svd_rank": lambda: ODMD(d=5, svd_rank=2.5).fit(sig),
        "results before fit": lambda: ODMD(d=5).eigs,
    }
    for name, call in cases.items():
        try:
            call()
        except (ValueError, RuntimeError) as e:
            print(f"    {name:20s} -> {type(e).__name__}: {e}")
        else:
            raise AssertionError(f"{name} did not raise")


if __name__ == "__main__":
    ok = run_tests(globals())
    sys.exit(0 if ok else 1)
