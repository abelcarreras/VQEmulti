"""
Test 3: signals with NO coherent oscillations (negative controls).

ODMD will always return *some* eigenvalues. These tests check the diagnostics
that tell you the result is meaningless:
  * flat singular-value spectrum (s_1 / median(s) close to 1),
  * Gavish-Donoho rank collapses to 0 (white noise),
  * eigenvalues away from the unit circle,
  * frequencies not stable when d changes (red noise / random walk).

    python -m unittest test_03_random_signal -v
"""
import unittest
import numpy as np

try:                         # imported as part of the package
    from . import _imports   # noqa: F401
except ImportError:          # run as a script / top-level module
    import _imports          # noqa: F401

from odmd import ODMD
import signals
from helpers import ODMDTestCase, omegas, match_error, log, plot_fit, PLOT


def spectrum_ratio(model):
    sv = model.singular_values
    return float(sv[0] / np.median(sv))


class TestRandomSignals(ODMDTestCase):

    def test_white_noise_rank_collapses(self):
        for seed in range(3):
            with self.subTest(seed=seed):
                t, sig, _ = signals.random_signal(kind="gaussian", seed=seed)
                # Passes only if the warning is actually emitted inside the block.
                with self.assertWarnsRegex(UserWarning, "Optimal rank is 0"):
                    m = ODMD(d=200, svd_rank=0, tol=None).fit(sig)
                log(f"seed {seed}: rank={m.operator.rank}, |lambda|={np.abs(m.eigs).round(3)}")
                self.assertRank(m, 1)
                self.assertTrue(np.all(np.abs(m.eigs) < 0.95),
                                "white noise mode should not look unitary")
                if PLOT and seed == 0:
                    plot_fit(t, sig, m, "white noise")

    def test_spectrum_flatness_diagnostic(self):
        """s_1/median(s): ~1-2 for white noise, >> 10 for a coherent signal even at 20 dB."""
        _, white, _ = signals.random_signal(kind="gaussian", seed=0)
        _, coh, _ = signals.complex_exponentials()
        coh = signals.add_noise(coh, 20, seed=0)
        r_white = spectrum_ratio(ODMD(d=200, svd_rank=-1, tol=None).fit(white))
        r_coh = spectrum_ratio(ODMD(d=200, svd_rank=-1, tol=None).fit(coh))
        log(f"s1/median(s): white noise {r_white:.2f}, coherent @20dB {r_coh:.1f}")
        self.assertLess(r_white, 3)
        self.assertGreater(r_coh, 20)

    def test_modes_unstable_under_delay_change(self):
        """
        Physical modes do not depend on d; fitted noise does. Compare frequencies
        found with d=60 and d=120.
        """
        def instability(sig, dt=0.1):
            a = ODMD(d=60, svd_rank=0, tol=None, reduced=True).fit(sig)
            b = ODMD(d=120, svd_rank=0, tol=None, reduced=True).fit(sig)
            return a.operator.rank, b.operator.rank, match_error(omegas(b, dt), omegas(a, dt))

        _, coh, _ = signals.complex_exponentials()
        ra, rb, e_coh = instability(signals.add_noise(coh, 20, seed=0))
        log(f"coherent @20dB: ranks {ra}/{rb}, max omega shift {e_coh:.1e}")
        self.assertEqual(ra, rb)
        self.assertLess(e_coh, 1e-2)

        _, walk, _ = signals.random_signal(kind="walk", seed=0)
        ra, rb, e_walk = instability(walk)
        log(f"random walk:    ranks {ra}/{rb}, max omega shift {e_walk:.1e}")
        self.assertNotEqual(ra, rb)
        self.assertGreater(e_walk, 10 * e_coh)


if __name__ == "__main__":
    unittest.main(verbosity=2)
