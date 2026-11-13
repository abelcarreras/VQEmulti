"""
Test 7 (optional): cross-check against pydmd's Hankel DMD.

Skipped automatically if pydmd is not installed. Run it in the environment
where pydmd is available to confirm the standalone version reproduces it:

    ODMD(d, svd_rank=r, tol=None)  ==  hankel_preprocessing(DMD(svd_rank=r, exact=True), d=d)

    python -m unittest test_07_compare_pydmd -v
"""
import importlib.util
import unittest
import numpy as np

try:                         # imported as part of the package
    from . import _imports   # noqa: F401
except ImportError:          # run as a script / top-level module
    import _imports          # noqa: F401

from odmd import ODMD
import signals
from helpers import ODMDTestCase, match_error, log

HAS_PYDMD = importlib.util.find_spec("pydmd") is not None


@unittest.skipUnless(HAS_PYDMD, "pydmd not installed")
class TestAgainstPyDMD(ODMDTestCase):

    def _compare(self, sig, d, svd_rank, label):
        from pydmd import DMD
        from pydmd.preprocessing import hankel_preprocessing

        ref_model = hankel_preprocessing(DMD(svd_rank=svd_rank, exact=True), d=d)
        ref_model.fit(np.asarray(sig).reshape(1, -1))
        ref = np.asarray(ref_model.eigs)
        mine = ODMD(d=d, svd_rank=svd_rank, tol=None, sorted_eigs="abs",
                    reduced=True).fit(sig).eigs

        diff = max(match_error(mine, ref), match_error(ref, mine)) if ref.size else 0.0
        log(f"{label}: pydmd {ref.size} eigs, ODMD {mine.size} eigs, "
            f"max |lambda_pydmd - lambda_odmd| = {diff:.1e}")
        self.assertEqual(ref.size, mine.size, "different ranks selected")
        # Order-independent comparison (np.sort_complex can mis-pair conjugates).
        self.assertSameSpectrum(mine, ref, 1e-6)

    def test_match_pydmd_real_noisy(self):
        _, sig, _ = signals.sin_cos()
        sig = signals.add_noise(sig, 10, seed=0)
        for r in (0, 2, 4, 6, 8):  # tested range of truncations for this example
            with self.subTest(svd_rank=r):
                self._compare(sig, d=25, svd_rank=r, label=f"sin+cos @10dB, svd_rank={r}")

    def test_match_pydmd_complex(self):
        _, sig, _ = signals.quantum_overlap()
        sig = signals.add_noise(sig, 40, seed=0)
        for d in (25, 100):
            with self.subTest(d=d):
                self._compare(sig, d=d, svd_rank=0, label=f"quantum overlap @40dB, d={d}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
