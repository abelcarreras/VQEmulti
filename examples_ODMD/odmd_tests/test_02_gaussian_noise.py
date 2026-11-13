"""
Test 2: signals with additive white Gaussian noise.

Lessons checked here:
  * Accuracy degrades gracefully with SNR.
  * The delay dimension d is the main noise knob: larger d averages noise over
    more Hankel rows (d ~ N/4 to N/3 is a good starting point).
  * Under noise, svd_rank=0 (Gavish-Donoho) adapts to the noise floor, while a
    fixed relative tol can let noise directions through.

    python -m unittest test_02_gaussian_noise -v
"""
import unittest
import numpy as np

try:                         # imported as part of the package
    from . import _imports   # noqa: F401
except ImportError:          # run as a script / top-level module
    import _imports          # noqa: F401

from odmd import ODMD
import signals
from helpers import ODMDTestCase, omegas, match_error, recon_error, log, plot_fit, PLOT

SEEDS = range(5)


def median_error(gen, snr, **cfg):
    _, sig, truth = gen()
    errs, ranks = [], []
    for seed in SEEDS:
        m = ODMD(**cfg).fit(signals.add_noise(sig, snr, seed))
        errs.append(match_error(omegas(m, truth["dt"]), truth["omegas"]))
        ranks.append(m.operator.rank)
    return float(np.median(errs)), int(np.median(ranks))


class TestSNRSweep(ODMDTestCase):
    """Frequency error vs SNR with automatic rank selection (svd_rank=0)."""

    LIMITS = {60: 1e-4, 40: 1e-3, 20: 1e-2, 10: 3e-2}

    def _sweep(self, gen, min_snr=10):
        for snr, lim in self.LIMITS.items():
            if snr < min_snr:
                continue
            with self.subTest(signal=gen.__name__, snr=snr):
                err, rank = median_error(gen, snr, d=100, svd_rank=0, tol=None)
                log(f"{gen.__name__:22s} {snr:3d} dB: rank={rank}, omega_err={err:.1e}")
                self.assertLess(err, lim)

    def test_sin_cos(self):
        self._sweep(signals.sin_cos)

    def test_complex_exponentials(self):
        self._sweep(signals.complex_exponentials)

    def test_quantum_overlap(self):
        # Its least-populated state (weight ~0.026) drops below the noise at
        # 10 dB and is correctly discarded, so the sweep stops at 20 dB.
        self._sweep(signals.quantum_overlap, min_snr=20)


class TestNoiseHandling(ODMDTestCase):

    def test_weak_component_detection(self):
        """A component is recoverable only while it stands out from the noise (30 dB)."""
        results = {}
        for w in (0.1, 0.03, 0.01, 0.003):
            _, sig, truth = signals.complex_exponentials(weights=(0.6, 0.3, w))
            m = ODMD(d=100, svd_rank=0, tol=None).fit(signals.add_noise(sig, 30, seed=0))
            err = np.min(np.abs(-omegas(m, truth["dt"]) - truth["energies"][2]))
            results[w] = (m.operator.rank, err)
            log(f"weight={w:<6} rank={m.operator.rank}  error on weak energy={err:.1e}")

        self.assertEqual(results[0.1][0], 3)
        self.assertLess(results[0.1][1], 1e-2)
        self.assertLess(results[0.003][0], 3, "component far below noise should be dropped")

    def test_delay_dimension_helps(self):
        e25, _ = median_error(signals.sin_cos, 20, d=25, svd_rank=4, tol=None)
        e100, _ = median_error(signals.sin_cos, 20, d=100, svd_rank=4, tol=None)
        log(f"20 dB: d=25 err={e25:.1e}, d=100 err={e100:.1e}")
        self.assertLess(e100, 0.1 * e25)

    def test_rank_selection_under_noise(self):
        _, sig, truth = signals.complex_exponentials()
        noisy = signals.add_noise(sig, 20, seed=0)
        gd = ODMD(d=100, svd_rank=0, tol=None).fit(noisy)
        lt = ODMD(d=100, svd_rank=-1, tol=0.01).fit(noisy)
        log(f"20 dB: svd_rank=0 -> {gd.operator.rank}; tol=0.01 -> {lt.operator.rank}")
        self.assertRank(gd, truth["rank"])
        self.assertGreater(lt.operator.rank, truth["rank"])

    def test_noise_modes_are_damped(self):
        t, sig, truth = signals.complex_exponentials()
        noisy = signals.add_noise(sig, 20, seed=0)
        m = ODMD(d=100, svd_rank=20, tol=None).fit(noisy)

        w = omegas(m, truth["dt"])
        phys = [int(np.argmin(np.abs(w - wt))) for wt in truth["omegas"]]
        other = np.setdiff1d(np.arange(m.eigs.size), phys)
        r_phys, r_other = np.abs(m.eigs[phys]), np.abs(m.eigs[other])
        log(f"|lambda| physical {np.round(r_phys, 4)}, noise median {np.median(r_other):.4f}")

        self.assertOnUnitCircle(m.eigs[phys], 2e-3)
        self.assertLess(np.median(r_other), r_phys.min())
        if PLOT:
            plot_fit(t, noisy, m, "20 dB, svd_rank=20 (over-fitted)")

    def test_reconstruction_denoises(self):
        t, sig, _ = signals.complex_exponentials()
        noisy = signals.add_noise(sig, 20, seed=0)
        m = ODMD(d=100, svd_rank=0, tol=None).fit(noisy)
        err_data = np.linalg.norm(noisy - sig) / np.linalg.norm(sig)
        err_model = recon_error(m, sig)
        log(f"data vs clean {err_data:.3f}, ODMD vs clean {err_model:.3f}")
        self.assertLess(err_model, err_data)
        if PLOT:
            plot_fit(t, noisy, m, "20 dB, svd_rank=0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
