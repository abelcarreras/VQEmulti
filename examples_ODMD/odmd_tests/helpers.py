"""
Shared helpers for the unittest-based ODMD validation suite.

Environment switches (unittest has no custom command-line flags):
    ODMD_PLOT=1      show plots
    ODMD_VERBOSE=0   hide the diagnostic numbers printed by the tests
"""
import os
import unittest
import numpy as np

PLOT = os.environ.get("ODMD_PLOT", "0") == "1"
VERBOSE = os.environ.get("ODMD_VERBOSE", "1") == "1"


# ---------------------------------------------------------------------------
# Metrics (unchanged from the script version)
# ---------------------------------------------------------------------------
def omegas(model, dt):
    """Angular frequencies: omega_k = arg(lambda_k) / dt."""
    return np.angle(model.eigs) / dt


def decays(model, dt):
    """Continuous-time growth rates: Re(log lambda_k) / dt (negative = decay)."""
    return np.log(np.abs(model.eigs)) / dt


def match_error(found, expected):
    """
    For every expected value, distance to the nearest found value.
    Returns the worst case (max over expected).
    """
    found = np.asarray(found)
    expected = np.asarray(expected)
    if found.size == 0:
        return np.inf
    return float(max(np.min(np.abs(found - e)) for e in expected))


def recon_error(model, sig):
    """Relative L2 error of the reconstructed signal (row 0 of reconstructed_data)."""
    rec = model.reconstructed_data[0]
    return float(np.linalg.norm(rec - sig) / np.linalg.norm(sig))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def log(msg):
    """Diagnostic print; silenced with ODMD_VERBOSE=0."""
    if VERBOSE:
        print(f"    {msg}")


def report(name, model, dt, sig=None):
    """One-line summary plus the eigenvalue table (silenced with ODMD_VERBOSE=0)."""
    if not VERBOSE:
        return
    line = f"{name}: rank={model.operator.rank}, n_eigs={model.eigs.size}"
    if sig is not None:
        line += f", recon_err={recon_error(model, sig):.2e}"
    print(f"\n    {line}")
    for k, lam in enumerate(model.eigs):
        print(f"      {k}: |lambda|={abs(lam):.6f}  omega={np.angle(lam) / dt:+.6f}"
              f"  decay={np.log(abs(lam)) / dt:+.2e}")


def plot_fit(t, sig, model, title):
    """Data vs reconstruction, and eigenvalues on the unit circle."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(t, np.real(sig), "k-", lw=1, label="Re data")
    rec = model.reconstructed_data[0]
    ax1.plot(t, rec.real, "r--", lw=1, label="Re ODMD")
    ax1.set_xlabel("t")
    ax1.legend()
    ax1.set_title(title)

    th = np.linspace(0, 2 * np.pi, 400)
    ax2.plot(np.cos(th), np.sin(th), "k:", lw=0.8)
    ax2.plot(model.eigs.real, model.eigs.imag, "o", mfc="none")
    ax2.set_aspect("equal")
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.axvline(0, color="gray", lw=0.5)
    ax2.set_title("Eigenvalues")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Base TestCase with ODMD-specific assertions
# ---------------------------------------------------------------------------
class ODMDTestCase(unittest.TestCase):
    """Subclass this instead of unittest.TestCase to get the assertions below."""

    def assertMatches(self, found, expected, tol, what="values"):
        """Every expected value has a found value within tol (order-independent)."""
        err = match_error(found, expected)
        self.assertLess(
            err, tol,
            f"{what}: max mismatch {err:.2e} >= {tol:.0e}\n"
            f"  found    = {np.round(np.sort_complex(np.asarray(found, complex)), 8)}\n"
            f"  expected = {np.round(np.sort_complex(np.asarray(expected, complex)), 8)}",
        )
        return err

    def assertSameSpectrum(self, a, b, tol):
        """Same set of eigenvalues, regardless of order (robust to near-ties)."""
        a, b = np.asarray(a), np.asarray(b)
        self.assertEqual(a.size, b.size, f"{a.size} vs {b.size} eigenvalues")
        self.assertMatches(a, b, tol, "spectrum a -> b")
        self.assertMatches(b, a, tol, "spectrum b -> a")

    def assertOnUnitCircle(self, eigs, atol):
        dev = float(np.max(np.abs(np.abs(eigs) - 1.0)))
        self.assertLess(dev, atol, f"max ||lambda| - 1| = {dev:.2e} >= {atol:.0e}")

    def assertRank(self, model, expected):
        self.assertEqual(model.operator.rank, expected,
                         f"selected rank {model.operator.rank}, expected {expected}")
