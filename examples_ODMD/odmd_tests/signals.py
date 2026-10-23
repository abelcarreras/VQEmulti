"""
Synthetic signals with known ground truth for validating ODMD.

Every generator returns (t, sig, truth), where `truth` is a dict describing
what ODMD should find. Angular frequencies follow the convention
lambda = exp(i * omega * dt), so omega = angle(lambda) / dt.
For quantum-like overlaps s(t) = sum c_j exp(-i E_j t), omega_j = -E_j.
"""
import numpy as np


def time_grid(n=400, dt=0.1, t0=0.0):
    return t0 + dt * np.arange(n), dt


# ---------------------------------------------------------------------------
# Exact signals
# ---------------------------------------------------------------------------
def sin_cos(w1=1.3, w2=0.4, a1=1.0, a2=0.5, n=400, dt=0.1):
    """a1 sin(w1 t) + a2 cos(w2 t): real, so each frequency appears as +/- pair."""
    t, dt = time_grid(n, dt)
    sig = a1 * np.sin(w1 * t) + a2 * np.cos(w2 * t)
    return t, sig, {"omegas": np.array([w1, -w1, w2, -w2]), "rank": 4, "dt": dt}


def complex_exponentials(energies=(-1.1, -0.4, 0.7), weights=(0.6, 0.3, 0.1),
                         n=400, dt=0.1):
    """s(t) = sum_j c_j exp(-i E_j t): one eigenvalue per energy."""
    t, dt = time_grid(n, dt)
    E = np.asarray(energies, float)
    c = np.asarray(weights, complex)
    sig = np.exp(-1j * np.outer(t, E)) @ c
    return t, sig, {"energies": E, "omegas": -E, "rank": E.size, "dt": dt}


def damped(omegas=(1.3, 0.4), gammas=(0.05, 0.2), amps=(1.0, 0.7), n=400, dt=0.1):
    """sum_j a_j exp((-gamma_j + i omega_j) t): complex, decaying modes."""
    t, dt = time_grid(n, dt)
    w = np.asarray(omegas, float)
    g = np.asarray(gammas, float)
    a = np.asarray(amps, complex)
    sig = np.exp(np.outer(t, -g + 1j * w)) @ a
    return t, sig, {"omegas": w, "gammas": g, "rank": w.size, "dt": dt}


def quantum_overlap(n_levels=4, n=400, dt=0.1, energy_scale=2.0, seed=1):
    """
    s(t) = <psi0| exp(-i H t) |psi0> for a random Hermitian H.
    Mimics a quantum overlap signal. |E| * dt is kept below pi (no aliasing).
    """
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n_levels, n_levels)) + 1j * rng.standard_normal((n_levels, n_levels))
    H = (M + M.conj().T) / 2
    H *= energy_scale / np.max(np.abs(np.linalg.eigvalsh(H)))
    E, V = np.linalg.eigh(H)

    psi0 = rng.standard_normal(n_levels) + 1j * rng.standard_normal(n_levels)
    psi0 /= np.linalg.norm(psi0)
    weights = np.abs(V.conj().T @ psi0) ** 2  # |<E_j|psi0>|^2

    t, dt = time_grid(n, dt)
    sig = np.exp(-1j * np.outer(t, E)) @ weights
    return t, sig, {"energies": E, "weights": weights, "omegas": -E,
                    "rank": n_levels, "dt": dt, "H": H}


# ---------------------------------------------------------------------------
# Perturbations
# ---------------------------------------------------------------------------
def add_noise(sig, snr_db, seed=0):
    """Add white Gaussian noise at a given SNR (dB). Complex noise for complex signals."""
    rng = np.random.default_rng(seed)
    p_sig = np.mean(np.abs(sig) ** 2)
    p_noise = p_sig / 10 ** (snr_db / 10)
    if np.iscomplexobj(sig):
        noise = np.sqrt(p_noise / 2) * (rng.standard_normal(sig.shape)
                                        + 1j * rng.standard_normal(sig.shape))
    else:
        noise = np.sqrt(p_noise) * rng.standard_normal(sig.shape)
    return sig + noise


def random_signal(n=400, dt=0.1, kind="gaussian", seed=1000):
    """Signal with no coherent structure: white Gaussian noise or a random walk."""
    rng = np.random.default_rng(seed)
    t, dt = time_grid(n, dt)
    if kind == "gaussian":
        sig = rng.standard_normal(n)
    elif kind == "walk":
        sig = np.cumsum(rng.standard_normal(n))
    else:
        raise ValueError(kind)
    return t, sig, {"rank": None, "dt": dt}
