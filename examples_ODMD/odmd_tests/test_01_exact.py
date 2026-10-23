"""
Test 1: noise-free signals. ODMD should be exact to round-off.

On exact data use a relative cutoff (tol) for rank selection: svd_rank=0
(Gavish-Donoho) estimates the noise floor from the median singular value and
assumes there IS noise. With pure round-off it can keep spurious directions
(see test_gavish_donoho_on_exact_data).

    python test_01_exact.py          # run
    python test_01_exact.py --plot   # run and show plots
"""
import numpy as np
import sys

from odmd import ODMD

import signals
from helpers import omegas, match_error, recon_error, report, plot_fit, run_tests


def test_sin_cos(plot=False):
    """Real signal: 2 frequencies -> 4 eigenvalues (+/- pairs) on the unit circle."""
    t, sig, truth = signals.sin_cos()
    dt = truth["dt"]
    model = ODMD(d=25, tol=1e-8, svd_rank=0, sorted_eigs="abs", reduced=True).fit(sig)
    report("sin+cos", model, dt, sig)

    assert model.operator.rank == truth["rank"], f"rank {model.operator.rank} != 4"
    assert match_error(omegas(model, dt), truth["omegas"]) < 1e-8
    assert np.allclose(np.abs(model.eigs), 1.0, atol=1e-10), "eigenvalues off unit circle"
    assert recon_error(model, sig) < 1e-8
    if plot:
        plot_fit(t, sig, model, "sin + cos")


def test_complex_exponentials(plot=False):
    """Complex overlap: one eigenvalue per energy; signs are recovered."""
    t, sig, truth = signals.complex_exponentials()
    dt = truth["dt"]
    model = ODMD(d=25, tol=1e-8, svd_rank=-1, reduced=True).fit(sig)
    report("complex exp", model, dt, sig)

    energies = -omegas(model, dt)
    assert model.operator.rank == truth["rank"]
    assert match_error(energies, truth["energies"]) < 1e-8
    # ground_phase() = E_min * dt (lowest energy has the largest phase)
    assert abs(model.ground_phase() / dt - truth["energies"].min()) < 1e-8
    if plot:
        plot_fit(t, sig, model, "complex exponentials")


def test_rank_criteria_agree():
    """On clean data, every rank-selection strategy should give the same answer."""
    _, sig, truth = signals.sin_cos()
    configs = [dict(svd_rank=4, tol=None),
               dict(svd_rank=-1, tol=1e-6), dict(svd_rank=0.999999, tol=None)]
    for cfg in configs:
        m = ODMD(d=25, **cfg).fit(sig)
        print(f"    {cfg} -> rank {m.operator.rank}")
        assert m.operator.rank == truth["rank"], f"{cfg} gave rank {m.operator.rank}"


def test_gavish_donoho_on_exact_data():
    """
    svd_rank=0 on noise-free data can over-estimate the rank, because the
    'noise' it measures is round-off. Adding tol, or any realistic noise,
    fixes it. Spurious modes have negligible amplitude, so the true
    eigenvalues are still present either way.
    """
    _, sig, truth = signals.complex_exponentials()
    raw = ODMD(d=25, svd_rank=0, tol=None).fit(sig)
    print(f"    exact data, svd_rank=0 only     -> rank {raw.operator.rank} (true {truth['rank']})")
    assert match_error(-omegas(raw, truth["dt"]), truth["energies"]) < 1e-8

    combined = ODMD(d=25, svd_rank=0, tol=1e-8).fit(sig)
    print(f"    exact data, svd_rank=0 + tol    -> rank {combined.operator.rank}")
    assert combined.operator.rank == truth["rank"]

    noisy = ODMD(d=25, svd_rank=0, tol=None).fit(signals.add_noise(sig, 120))
    print(f"    120 dB noise, svd_rank=0 only   -> rank {noisy.operator.rank}")
    assert noisy.operator.rank == truth["rank"]


def test_input_shapes():
    """(n,), (1, n) and (n, 1) inputs must give identical results."""
    _, sig, _ = signals.sin_cos()
    ref = np.sort_complex(ODMD(d=25, tol=1e-8).fit(sig).eigs)
    for X in (sig.reshape(1, -1), sig.reshape(-1, 1)):
        e = np.sort_complex(ODMD(d=25, tol=1e-8).fit(X).eigs)
        assert np.allclose(e, ref)

if __name__ == "__main__":
    ok = run_tests(globals(), plot="--plot" in sys.argv)
    sys.exit(0 if ok else 1)
