"""
Test 6: time handling and forecasting.

  * fit(..., t0, dt) puts frequencies/growth rates in physical units.
  * dmd_time can be extended past the data to forecast.
  * A model fitted on a short window should predict the rest of the signal.

    python -m unittest test_06_forecast -v
    ODMD_PLOT=1 python -m unittest test_06_forecast -v
"""
import unittest
import numpy as np

try:                         # imported as part of the package
    from . import _imports   # noqa: F401
except ImportError:          # run as a script / top-level module
    import _imports          # noqa: F401

from odmd import ODMD
import signals
from helpers import ODMDTestCase, log, PLOT


class TestTimeUnits(ODMDTestCase):

    def test_physical_time_units(self):
        t, sig, truth = signals.complex_exponentials(dt=0.05)
        m = ODMD(d=100, tol=1e-8, reduced=True).fit(sig, t0=t[0], dt=truth["dt"])
        # frequency (cycles per unit time) = omega / (2 pi) = -E / (2 pi)
        np.testing.assert_allclose(np.sort(m.frequency),
                                   np.sort(-truth["energies"] / (2 * np.pi)), atol=1e-8)
        np.testing.assert_allclose(m.growth_rate, 0, atol=1e-8)
        np.testing.assert_allclose(m.original_timesteps, t)


class TestForecast(ODMDTestCase):
    """Fit on the first 300 steps, predict the remaining 500."""

    N_TRAIN = 300
    LIMITS = {None: 1e-6, 40: 5e-2, 20: 0.3}

    @classmethod
    def setUpClass(cls):
        cls.t, cls.sig, cls.truth = signals.quantum_overlap(n=800)

    def _forecast(self, snr):
        """Returns (model, data, prediction, relative error on the unseen part)."""
        t, sig, dt = self.t, self.sig, self.truth["dt"]
        data = sig if snr is None else signals.add_noise(sig, snr, seed=0)
        # Each call builds its own model: the shared class data is never modified.
        m = ODMD(d=100, svd_rank=0, tol=None if snr else 1e-8)
        m.fit(data[: self.N_TRAIN], t0=t[0], dt=dt)
        m.dmd_time = {"tend": t[-1]}                    # extend to the full record

        pred = m.reconstructed_data[0]
        future = slice(self.N_TRAIN, None)
        err = np.linalg.norm(pred[future] - sig[future]) / np.linalg.norm(sig[future])
        label = "exact" if snr is None else f"{snr} dB"
        log(f"{label:6s}: forecast of {sig.size - self.N_TRAIN} unseen steps, "
            f"rel. error {err:.1e}")
        return m, data, pred, err

    def test_forecast(self):
        for snr, limit in self.LIMITS.items():
            with self.subTest(snr=snr):
                _, _, pred, err = self._forecast(snr)
                self.assertEqual(pred.size, self.sig.size)
                self.assertLess(err, limit)

    @unittest.skipUnless(PLOT, "set ODMD_PLOT=1 to show plots")
    def test_plot(self):
        import matplotlib.pyplot as plt
        m, data, pred, _ = self._forecast(20)
        plt.figure(figsize=(9, 4))
        plt.plot(self.t, data.real, color="0.7", lw=0.8, label="noisy data")
        plt.plot(self.t, self.sig.real, "k-", lw=1, label="clean")
        plt.plot(m.dmd_timesteps, pred.real, "r--", lw=1, label="ODMD")
        plt.axvline(self.t[self.N_TRAIN], color="b", ls=":", label="end of training")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    unittest.main(verbosity=2)
