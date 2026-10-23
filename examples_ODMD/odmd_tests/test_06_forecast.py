"""
Test 6: time handling and forecasting.

  * fit(..., t0, dt) puts frequencies/growth rates in physical units.
  * dmd_time can be extended past the data to forecast.
  * A model fitted on a short window should predict the rest of the signal.

    python test_06_forecast.py [--plot]
"""
import sys
import numpy as np

from odmd import ODMD
import signals
from helpers import run_tests


def test_physical_time_units():
    t, sig, truth = signals.complex_exponentials(dt=0.05)
    m = ODMD(d=100, tol=1e-8, reduced=True).fit(sig, t0=t[0], dt=truth["dt"])
    # frequency (cycles per unit time) = omega / (2 pi) = -E / (2 pi)
    assert np.allclose(np.sort(m.frequency), np.sort(-truth["energies"] / (2 * np.pi)), atol=1e-8)
    assert np.allclose(m.growth_rate, 0, atol=1e-8)
    assert np.allclose(m.original_timesteps, t)


def test_forecast(plot=False):
    t, sig, truth = signals.quantum_overlap(n=800)
    dt = truth["dt"]
    n_train = 300

    for snr in (None, 40, 20):
        data = sig if snr is None else signals.add_noise(sig, snr, seed=0)
        m = ODMD(d=100, svd_rank=0, tol=None if snr else 1e-8)
        m.fit(data[:n_train], t0=t[0], dt=dt)
        m.dmd_time = {"tend": t[-1]}                     # extend to the full record

        pred = m.reconstructed_data[0]
        future = slice(n_train, None)
        err = np.linalg.norm(pred[future] - sig[future]) / np.linalg.norm(sig[future])
        label = "exact" if snr is None else f"{snr} dB"
        print(f"    {label:6s}: forecast of {sig.size - n_train} unseen steps, rel. error {err:.1e}")
        assert pred.size == sig.size
        assert err < {None: 1e-6, 40: 5e-2, 20: 0.3}[snr]

        if plot and snr == 20:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(9, 4))
            plt.plot(t, data.real, color="0.7", lw=0.8, label="noisy data")
            plt.plot(t, sig.real, "k-", lw=1, label="clean")
            plt.plot(m.dmd_timesteps, pred.real, "r--", lw=1, label="ODMD")
            plt.axvline(t[n_train], color="b", ls=":", label="end of training")
            plt.legend()
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    ok = run_tests(globals(), plot="--plot" in sys.argv)
    sys.exit(0 if ok else 1)
