"""
Test 1: noise-free signals. ODMD should be exact to round-off.

On exact data use a relative cutoff (tol) for rank selection: svd_rank=0
(Gavish-Donoho) estimates the noise floor from the median singular value and
assumes there IS noise. With pure round-off it can keep spurious directions
(see test_gavish_donoho_on_exact_data).

    python -m unittest test_01_exact -v
    ODMD_PLOT=1 python -m unittest test_01_exact -v
"""
import unittest
import numpy as np

try:                         # imported as part of the package
    from . import _imports   # noqa: F401
except ImportError:          # run as a script / top-level module
    import _imports          # noqa: F401

from odmd import ODMD
import signals
from helpers import ODMDTestCase, omegas, recon_error, report, log, plot_fit, PLOT


class TestSinCos(ODMDTestCase):
    """Real signal: 2 frequencies -> 4 eigenvalues (+/- pairs) on the unit circle."""

    @classmethod
    def setUpClass(cls):
        cls.t, cls.sig, cls.truth = signals.sin_cos()
        cls.dt = cls.truth["dt"]
        cls.model = ODMD(d=200, tol=1e-8, svd_rank=0, sorted_eigs="abs",
                         reduced=True).fit(cls.sig)
        report("sin+cos", cls.model, cls.dt, cls.sig)

    def test_rank(self):
        self.assertRank(self.model, self.truth["rank"])

    def test_frequencies(self):
        self.assertMatches(omegas(self.model, self.dt), self.truth["omegas"], 1e-8, "omega")

    def test_unit_circle(self):
        self.assertOnUnitCircle(self.model.eigs, 1e-10)

    def test_reconstruction(self):
        self.assertLess(recon_error(self.model, self.sig), 1e-8)

    @unittest.skipUnless(PLOT, "set ODMD_PLOT=1 to show plots")
    def test_plot(self):
        plot_fit(self.t, self.sig, self.model, "sin + cos")


class TestComplexExponentials(ODMDTestCase):
    """Complex overlap: one eigenvalue per energy; signs are recovered."""

    @classmethod
    def setUpClass(cls):
        cls.t, cls.sig, cls.truth = signals.complex_exponentials()
        cls.dt = cls.truth["dt"]
        cls.model = ODMD(d=200, tol=1e-8, svd_rank=-1, reduced=True).fit(cls.sig)
        report("complex exp", cls.model, cls.dt, cls.sig)

    def test_rank(self):
        self.assertRank(self.model, self.truth["rank"])

    def test_energies(self):
        self.assertMatches(-omegas(self.model, self.dt), self.truth["energies"], 1e-8, "E")

    def test_ground_phase(self):
        """ground_phase() = E_min * dt (lowest energy has the largest phase)."""
        self.assertAlmostEqual(self.model.ground_phase() / self.dt,
                               self.truth["energies"].min(), delta=1e-8)

    @unittest.skipUnless(PLOT, "set ODMD_PLOT=1 to show plots")
    def test_plot(self):
        plot_fit(self.t, self.sig, self.model, "complex exponentials")


class TestRankSelection(ODMDTestCase):

    def test_rank_criteria_agree(self):
        """On clean data, every rank-selection strategy should give the same answer."""
        _, sig, truth = signals.sin_cos()
        configs = [dict(svd_rank=4, tol=None),
                   dict(svd_rank=-1, tol=1e-6),
                   dict(svd_rank=0.999999, tol=None)]
        for cfg in configs:
            with self.subTest(**cfg):
                m = ODMD(d=25, **cfg).fit(sig)
                log(f"{cfg} -> rank {m.operator.rank}")
                self.assertRank(m, truth["rank"])

    def test_gavish_donoho_on_exact_data(self):
        """
        svd_rank=0 on noise-free data can over-estimate the rank (round-off is
        taken as noise). Adding tol, or any realistic noise, fixes it. The true
        eigenvalues are present either way.
        """
        _, sig, truth = signals.complex_exponentials()

        raw = ODMD(d=100, svd_rank=0, tol=None).fit(sig)
        log(f"exact data, svd_rank=0 only   -> rank {raw.operator.rank} (true {truth['rank']})")
        self.assertMatches(-omegas(raw, truth["dt"]), truth["energies"], 1e-8, "E")

        with self.subTest("svd_rank=0 + tol"):
            combined = ODMD(d=100, svd_rank=0, tol=1e-8).fit(sig)
            log(f"exact data, svd_rank=0 + tol  -> rank {combined.operator.rank}")
            self.assertRank(combined, truth["rank"])

        with self.subTest("svd_rank=0 + 120 dB noise"):
            noisy = ODMD(d=100, svd_rank=0, tol=None).fit(signals.add_noise(sig, 120))
            log(f"120 dB noise, svd_rank=0 only -> rank {noisy.operator.rank}")
            self.assertRank(noisy, truth["rank"])


class TestInterface(ODMDTestCase):

    def test_input_shapes(self):
        """(n,), (1, n) and (n, 1) inputs must give identical results."""
        _, sig, _ = signals.sin_cos()
        ref = ODMD(d=200, tol=1e-8).fit(sig).eigs
        for shape in ((1, -1), (-1, 1)):
            with self.subTest(shape=shape):
                e = ODMD(d=200, tol=1e-8).fit(sig.reshape(shape)).eigs
                self.assertSameSpectrum(e, ref, 1e-10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
