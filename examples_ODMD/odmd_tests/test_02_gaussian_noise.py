"""
Test 2: signals with additive white Gaussian noise.

Lessons checked here:
  * Accuracy degrades gracefully with SNR.
  * The delay dimension d is the main noise knob: larger d averages noise over
    more Hankel rows (d ~ N/4 to N/3 is a good starting point).
  * Under noise, svd_rank=0 (Gavish-Donoho) adapts to the noise floor, while a
    fixed relative tol can let noise directions through.

    python test_02_gaussian_noise.py [--plot]
"""
import sys
import numpy as np

from odmd import ODMD
import signals
from helpers import omegas, match_error, recon_error, plot_fit, run_tests

SEEDS = range(5)


def _median_error(gen, snr, **cfg):
    t, sig, truth = gen()
    errs, ranks = [], []
    for seed in SEEDS:
        m = ODMD(**cfg).fit(signals.add_noise(sig, snr, seed))
        errs.append(match_error(omegas(m, truth["dt"]), truth["omegas"]))
        ranks.append(m.operator.rank)
    return float(np.median(errs)), int(np.median(ranks))


def test_snr_sweep():
    """
    Frequency error vs SNR with automatic rank selection (svd_rank=0).
    quantum_overlap stops at 20 dB: its least-populated eigenstate has a small
    weight |<E_j|psi0>|^2, and at 10 dB it falls below the noise floor and is
    (correctly) discarded. See test_weak_component_detection.
    """
    limits = {60: 1e-4, 40: 1e-3, 20: 1e-2, 10: 3e-2}
    for gen in (signals.sin_cos, signals.complex_exponentials, signals.quantum_overlap):
        true_rank = gen()[2]["rank"]
        for snr, lim in limits.items():
            if gen is signals.quantum_overlap and snr < 20:
                continue
            err, rank = _median_error(gen, snr, d=100, svd_rank=0, tol=None)
            print(f"    {gen.__name__:22s} SNR={snr:3d} dB  rank={rank} (true {true_rank})"
                  f"  omega_err={err:.1e}  (limit {lim:.0e})")
            assert err < lim, f"{gen.__name__} at {snr} dB: error {err:.2e} > {lim:.0e}"


def test_weak_component_detection():
    """
    A component is only recoverable while its amplitude stands out from the
    noise. Sweep the weight of the smallest component at fixed 30 dB SNR.
    """
    found = {}
    for w in (0.1, 0.03, 0.01, 0.003):
        _, sig, truth = signals.complex_exponentials(weights=(0.6, 0.3, w))
        noisy = signals.add_noise(sig, 30, seed=0)
        m = ODMD(d=100, svd_rank=0, tol=None).fit(noisy)
        err = np.min(np.abs(-omegas(m, truth["dt"]) - truth["energies"][2]))
        found[w] = (m.operator.rank, err)
        print(f"    weight={w:<6} rank={m.operator.rank}  error on weak energy={err:.1e}")
    assert found[0.1][0] == 3 and found[0.1][1] < 1e-2
    assert found[0.003][0] < 3, "a component far below the noise should be dropped"


def test_delay_dimension_helps():
    """At 20 dB, d=100 should be far more accurate than d=25."""
    gen = signals.sin_cos
    e25, _ = _median_error(gen, 20, d=25, svd_rank=4, tol=None)
    e100, _ = _median_error(gen, 20, d=100, svd_rank=4, tol=None)
    print(f"    20 dB: d=25 err={e25:.1e}, d=100 err={e100:.1e}")
    assert e100 < 0.1 * e25


def test_rank_selection_under_noise():
    """Gavish-Donoho finds the true rank; a loose tol keeps noise modes."""
    t, sig, truth = signals.complex_exponentials()
    noisy = signals.add_noise(sig, 20, seed=0)
    gd = ODMD(d=100, svd_rank=0, tol=None).fit(noisy)
    lt = ODMD(d=100, svd_rank=-1, tol=0.01).fit(noisy)
    print(f"    20 dB: svd_rank=0 -> rank {gd.operator.rank}; "
          f"tol=0.01 -> rank {lt.operator.rank}; true {truth['rank']}")
    assert gd.operator.rank == truth["rank"]
    assert lt.operator.rank > truth["rank"]


def test_noise_modes_are_damped(plot=False):
    """
    With too many modes kept, the extra (noise) eigenvalues sit inside the unit
    circle, while the physical ones stay on it. |lambda| is a useful filter.
    """
    t, sig, truth = signals.complex_exponentials()
    noisy = signals.add_noise(sig, 20, seed=0)
    m = ODMD(d=100, svd_rank=20, tol=None).fit(noisy)
    dt = truth["dt"]

    # physical eigenvalue = nearest to each true omega
    w = omegas(m, dt)
    phys = [int(np.argmin(np.abs(w - wt))) for wt in truth["omegas"]]
    other = np.setdiff1d(np.arange(m.eigs.size), phys)
    r_phys = np.abs(m.eigs[phys])
    r_other = np.abs(m.eigs[other])
    print(f"    |lambda| physical: {np.round(r_phys, 4)}")
    print(f"    |lambda| noise:    median {np.median(r_other):.4f}, max {r_other.max():.4f}")
    assert np.all(np.abs(r_phys - 1) < 2e-3)
    assert np.median(r_other) < r_phys.min()
    if plot:
        plot_fit(t, noisy, m, "20 dB, svd_rank=20 (over-fitted)")


def test_reconstruction_denoises(plot=False):
    """The rank-limited reconstruction should be closer to the clean signal than the data."""
    t, sig, truth = signals.complex_exponentials()
    noisy = signals.add_noise(sig, 20, seed=0)
    m = ODMD(d=100, svd_rank=0, tol=None).fit(noisy)
    err_data = np.linalg.norm(noisy - sig) / np.linalg.norm(sig)
    err_model = recon_error(m, sig)
    print(f"    noisy data vs clean: {err_data:.3f}, ODMD vs clean: {err_model:.3f}")
    assert err_model < err_data
    if plot:
        plot_fit(t, noisy, m, "20 dB, svd_rank=0")


if __name__ == "__main__":
    ok = run_tests(globals(), plot="--plot" in sys.argv)
    sys.exit(0 if ok else 1)
