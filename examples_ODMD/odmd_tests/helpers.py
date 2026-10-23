"""Shared helpers for the ODMD validation scripts."""
import numpy as np


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


def report(name, model, dt, sig=None):
    """Print a one-line summary plus the eigenvalue table."""
    line = f"{name}: rank={model.operator.rank}, n_eigs={model.eigs.size}"
    if sig is not None:
        line += f", recon_err={recon_error(model, sig):.2e}"
    print(line)
    for k, lam in enumerate(model.eigs):
        print(f"    {k}: |lambda|={abs(lam):.6f}  omega={np.angle(lam) / dt:+.6f}"
              f"  decay={np.log(abs(lam)) / dt:+.2e}")


def plot_fit(t, sig, model, title):
    """Optional plot: data vs reconstruction, and eigenvalues on the unit circle."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(t, np.real(sig), "k-", lw=1, label="Re data")
    rec = model.reconstructed_data[0]
    ax1.plot(model.dmd_timesteps, rec.real, "r--", lw=1, label="Re ODMD")
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


def run_tests(namespace, plot=False):
    """
    Minimal runner: calls every function named test_* in `namespace`.
    Works without pytest; the same files also run under pytest.
    """
    tests = {k: v for k, v in namespace.items() if k.startswith("test_") and callable(v)}
    passed = 0
    for name, fn in tests.items():
        try:
            if plot and "plot" in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
                fn(plot=True)
            else:
                fn()
            print(f"PASS  {name}\n")
            passed += 1
        except SkipTest as e:
            print(f"SKIP  {name}: {e}\n")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {name}: {e}\n")
    print(f"{passed}/{len(tests)} passed")
    return passed == len(tests)


try:  # under pytest, report skips as real skips
    import pytest
    SkipTest = pytest.skip.Exception
except ImportError:
    class SkipTest(Exception):
        pass
