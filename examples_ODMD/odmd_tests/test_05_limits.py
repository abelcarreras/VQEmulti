"""
Test 5: where the method breaks, and the guards that should catch it.

  * d must be at least the number of components.
  * Close frequencies: exact data resolves them; noise sets a resolution limit.
  * Aliasing: |omega| * dt must stay below pi, otherwise the frequency folds.
  * Invalid input raises a clear error.

    python -m unittest test_05_limits -v
"""
import unittest
import numpy as np

try:                         # imported as part of the package
    from . import _imports   # noqa: F401
except ImportError:          # run as a script / top-level module
    import _imports          # noqa: F401

from odmd import ODMD
import signals
from helpers import ODMDTestCase, omegas, match_error, log


class TestLimits(ODMDTestCase):

    def test_delay_dimension_too_small(self):
        _, sig, truth = signals.complex_exponentials()     # 3 components
        for d, should_work in ((2, False), (3, True), (4, True)):
            with self.subTest(d=d):
                m = ODMD(d=d, svd_rank=-1, tol=1e-8).fit(sig)
                err = match_error(-omegas(m, truth["dt"]), truth["energies"])
                log(f"d={d}: rank={m.operator.rank}, energy err={err:.1e}")
                self.assertEqual(err < 1e-8, should_work)

    def test_close_frequencies(self):
        """
        Exact data resolves separations far below the Fourier limit 2*pi/T
        (=0.157 here). With 40 dB noise, 0.05 is still resolved but 0.02 is not:
        the two energies collapse into one (error ~ sep/2). That is the practical
        resolution limit for this record length and noise level.
        """
        for sep, resolvable in ((0.1, True), (0.05, True), (0.02, False)):
            with self.subTest(sep=sep):
                _, sig, truth = signals.complex_exponentials(energies=(0.5, 0.5 + sep),
                                                             weights=(0.5, 0.5))
                exact = ODMD(d=100, svd_rank=2, tol=None).fit(sig)
                noisy = ODMD(d=100, svd_rank=2, tol=None).fit(
                    signals.add_noise(sig, 40, seed=0))
                e_no = match_error(-omegas(noisy, 0.1), truth["energies"])
                log(f"separation {sep:<5}: 40 dB err={e_no:.1e} ({e_no / sep:.1%} of sep)")
                self.assertMatches(-omegas(exact, 0.1), truth["energies"], 1e-8, "E exact")
                if resolvable:
                    self.assertLess(e_no, 0.1 * sep,
                                    "the two energies should be distinguishable")

    def test_aliasing(self):
        """
        E*dt > pi: the recovered energy is folded to E - 2*pi/dt.

        With svd_rank=0 and tol=None on exact data, Gavish-Donoho also keeps
        round-off modes (rank 5 here), so the folded energy is searched for
        among all eigenvalues instead of assuming it is eigs[0].
        """
        dt, E = 0.1, 40.0
        _, sig, _ = signals.complex_exponentials(energies=(E,), weights=(1.0,), dt=dt)
        m = ODMD(d=10, svd_rank=0, tol=None, reduced=True).fit(sig)

        aliased = E - 2 * np.pi / dt
        log(f"true E={E}, expected alias {aliased:.6f}, rank={m.operator.rank}; "
            f"Nyquist limit (max resolvable |E|) = {np.pi / dt:.6f}")
        self.assertMatches(-omegas(m, dt), [aliased], 1e-8, "aliased E")


class TestInvalidInput(unittest.TestCase):
    """Each bad input must raise a clear error (one test per case)."""

    @classmethod
    def setUpClass(cls):
        _, cls.sig, _ = signals.sin_cos(n=30)

    def test_d_too_large(self):
        with self.assertRaisesRegex(ValueError, "len"):
            ODMD(d=30).fit(self.sig)

    def test_empty_input(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            ODMD(d=5).fit(np.array([]))

    def test_nan_input(self):
        with self.assertRaisesRegex(ValueError, "NaN"):
            ODMD(d=5).fit(np.r_[self.sig[:-1], np.nan])

    def test_multirow_input(self):
        with self.assertRaisesRegex(ValueError, "single sequence"):
            ODMD(d=5).fit(np.vstack([self.sig, self.sig, self.sig]))

    def test_bad_svd_rank(self):
        with self.assertRaisesRegex(ValueError, "svd_rank"):
            ODMD(d=5, svd_rank=2.5).fit(self.sig)

    def test_results_before_fit(self):
        with self.assertRaisesRegex(RuntimeError, "fit"):
            ODMD(d=5).eigs


if __name__ == "__main__":
    unittest.main(verbosity=2)
