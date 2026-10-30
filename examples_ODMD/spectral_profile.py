"""
Spectral diagnostics for sampling C(t) = <psi0| exp(-iHt) |psi0>.

C(t) = sum_k w_k exp(-i E_k t),  w_k = |<E_k|psi0>|^2
Sampling at t_n = n*dt only resolves E_k modulo 2*pi/dt, so the time step
is set by the width of the part of the spectrum that psi0 actually touches.

Works with an OpenFermion QubitOperator (JW-mapped), coefficients in Hartree,
times in hbar/Eh.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------
# Data container
# --------------------------------------------------------------------------
@dataclass
class SpectralProfile:
    n_qubits: int
    identity: float  # c_0, coefficient of the identity string
    l1: float  # sum_j |c_j| (j != I):  ||H - c0 I|| <= l1
    l2: float  # sqrt(sum_j c_j^2):     ||H - c0 I|| >= l2
    n_terms: int
    e_ref: float  # <psi0|H|psi0>
    sigma_ref: float  # sqrt(<H^2> - <H>^2) on psi0
    e_min: float | None = None  # exact extremes in the N-electron sector
    e_max: float | None = None  # (optional, small systems only)


# --------------------------------------------------------------------------
# Norms from the Pauli decomposition
# --------------------------------------------------------------------------
def pauli_norms(H):
    """Return (c0, l1, l2, n_terms) of a Hermitian QubitOperator."""
    c0 = complex(H.terms.get((), 0.0))
    c = np.array([complex(v) for k, v in H.terms.items() if k != ()])
    if abs(c0.imag) > 1e-10 or np.max(np.abs(c.imag), initial=0.0) > 1e-10:
        raise ValueError("Complex Pauli coefficients: H is not Hermitian.")
    a = np.abs(c.real)
    return c0.real, float(a.sum()), float(np.sqrt((a ** 2).sum())), int(a.size)


# --------------------------------------------------------------------------
# Energy moments of a computational-basis reference (e.g. JW HF state)
# --------------------------------------------------------------------------
def hf_bits(n_qubits, n_electrons):
    """OpenFermion JW convention: first n_electrons spin-orbitals occupied."""
    return [1] * n_electrons + [0] * (n_qubits - n_electrons)


def reference_moments(H, bits):
    """
    computes ⟨H⟩ and Var(H) on the HF bitstring without building any matrix.
    Each Pauli string flips some bits and adds a phase.
    Terms are grouped by flip mask: the mask-0 group gives ⟨H⟩,
    and the squared amplitudes of all other groups add up to the variance.

    <H> and Var(H) on a basis state |x>, with no matrices.

    Each Pauli string maps |x> -> phase * |x XOR flipmask>. Grouping terms
    by flip mask m gives H|x> = sum_m A_m |x XOR m>, hence
        <H>   = Re A_0
        Var H = sum_{m != 0} |A_m|^2
    Cost: O(number of Pauli terms).
    """
    x = np.asarray(bits, dtype=int)
    amps = defaultdict(complex)
    for term, coeff in H.terms.items():
        phase = 1.0 + 0.0j
        flips = []
        for q, p in term:
            b = x[q]
            if p == "Z":
                phase *= (-1) ** b
            elif p == "X":
                flips.append(q)
            else:  # 'Y':  Y|b> = i(-1)^b |1-b>
                phase *= 1j * (-1) ** b
                flips.append(q)
        amps[tuple(sorted(flips))] += coeff * phase
    mean = amps.get((), 0.0).real
    var = sum(abs(a) ** 2 for m, a in amps.items() if m != ())
    return float(mean), float(var)


# --------------------------------------------------------------------------
# Exact extremes in the particle-number sector (small systems)
# --------------------------------------------------------------------------
def sector_extremes(H, n_qubits, n_electrons):
    """
    diagonalises H restricted to the 2-electron sector to get the true E_min and E_max there.
    """
    from openfermion import get_sparse_operator
    from openfermion.linalg import jw_number_restrict_operator
    from scipy.sparse.linalg import eigsh

    Hs = get_sparse_operator(H, n_qubits=n_qubits)
    Hn = jw_number_restrict_operator(Hs, n_electrons, n_qubits=n_qubits)
    if Hn.shape[0] <= 3:
        ev = np.linalg.eigvalsh(Hn.toarray())
        return float(ev[0]), float(ev[-1])
    e_min = eigsh(Hn, k=1, which="SA", return_eigenvectors=False)[0]
    e_max = eigsh(Hn, k=1, which="LA", return_eigenvectors=False)[0]
    return float(e_min), float(e_max)


def build_profile(H, n_qubits, n_electrons, exact_sector=False):
    c0, l1, l2, nt = pauli_norms(H)
    e_ref, var = reference_moments(H, hf_bits(n_qubits, n_electrons))
    prof = SpectralProfile(n_qubits, c0, l1, l2, nt, e_ref, float(np.sqrt(var)))
    if exact_sector:
        prof.e_min, prof.e_max = sector_extremes(H, n_qubits, n_electrons)
    return prof


# --------------------------------------------------------------------------
# Time step / Nyquist
# --------------------------------------------------------------------------
def spectral_window(prof, mode="rigorous", k_sigma=6.0):
    """
    Return (E_shift, W): all relevant E_k are assumed in [E_shift - W, E_shift + W].

    rigorous : shift c0, W = l1                      (always valid, often loose)
    sector   : midpoint of exact sector extremes     (tight, needs exact_sector)
    reference: shift <H>_ref, W = k*sigma            (heuristic; weight outside
               the window is <= 1/k^2 by Chebyshev, i.e. aliased weight)
    """
    if mode == "rigorous":
        return prof.identity, prof.l1
    if mode == "sector":
        if prof.e_min is None:
            raise ValueError("Build the profile with exact_sector=True.")
        return 0.5 * (prof.e_min + prof.e_max), 0.5 * (prof.e_max - prof.e_min)
    if mode == "reference":
        w_rig = prof.l1 + abs(prof.identity - prof.e_ref)
        return prof.e_ref, min(k_sigma * prof.sigma_ref, w_rig)
    raise ValueError(f"unknown mode {mode!r}")


def choose_time_step(prof, mode="rigorous", safety=0.9, k_sigma=6.0):
    """
    builds the window in one of three ways. rigorous uses c₀ ± L1.
    reference uses E_HF ± 6σ, capped by a rigorous bound. sector uses the midpoint and
    half-range of the exact sector spectrum.

    Largest dt with no aliasing after recentering: W * dt < pi.
    Returns dict with dt, Nyquist angular frequency, shift and window.
    """
    e_shift, W = spectral_window(prof, mode, k_sigma)
    dt = safety * np.pi / W
    return {
        "dt": dt,
        "omega_nyquist": np.pi / dt,
        "e_shift": e_shift,
        "half_width": W,
        "mode": mode,
    }


def samples_for_resolution(dt, epsilon):
    """Heisenberg scaling: total time T ~ pi/epsilon -> N ~ T/dt samples."""
    return int(np.ceil(np.pi / (epsilon * dt)))


# --------------------------------------------------------------------------
# Classical recentering (no extra circuits)
# --------------------------------------------------------------------------
def recenter(signal, times, e_shift):
    """Multiply the signal giving
    exp(+i E_s t) C(t) = sum_k w_k exp(-i (E_k - E_s) t)."""
    return np.exp(1j * e_shift * np.asarray(times)) * np.asarray(signal)


def eigvals_to_energies(lams, dt, e_shift):
    """Map DMD/ODMD eigenvalues lambda = exp(-i (E - E_s) dt) back to E."""
    return -np.angle(np.asarray(lams)) / dt + e_shift


# --------------------------------------------------------------------------
def report(prof):
    print(f"qubits={prof.n_qubits}  Pauli terms={prof.n_terms}")
    print(f"c0={prof.identity:.6f}  l1={prof.l1:.6f}  l2={prof.l2:.6f}  "
          f"(l1/l2={prof.l1 / prof.l2:.1f})")
    print(f"E_ref={prof.e_ref:.6f}  sigma_ref={prof.sigma_ref:.6f}")
    if prof.e_min is not None:
        print(f"sector spectrum: [{prof.e_min:.6f}, {prof.e_max:.6f}]")
    modes = ["rigorous", "reference"] + (["sector"] if prof.e_min is not None else [])
    for m in modes:
        r = choose_time_step(prof, m)
        print(f"  {m:9s}: dt={r['dt']:.4f}  omega_N={r['omega_nyquist']:.4f}  "
              f"shift={r['e_shift']:.4f}  W={r['half_width']:.4f}")


if __name__ == "__main__":
    # Example: H2/STO-3G at equilibrium (requires openfermion + pyscf).
    from openfermion import MolecularData, jordan_wigner
    from openfermionpyscf import run_pyscf

    #mol = MolecularData([("H", (0, 0, 0)), ("H", (0, 0, 0.74))], "sto-3g", 1, 0)
    #mol = MolecularData([("Li", (0, 0, 0))], "sto-3g", 1, 1)
    mol = MolecularData([("Ne", (0, 0, 0))], "sto-3g", 1, 8)
    mol = run_pyscf(mol)
    Hq = jordan_wigner(mol.get_molecular_hamiltonian())
    prof = build_profile(Hq, mol.n_qubits, mol.n_electrons, exact_sector=True)
    report(prof)

#    print("Hamiltonian:")
#    for term, coeff in Hq.terms.items():
#        print(f"{term}: {coeff:.6f}")

    C0, L1, L2, nt = pauli_norms(Hq)
    print("Pauli Norms:")
    print(f"  c0={C0:.6f}, L1={L1:.6f}, L2={L2:.6f}, n_terms={nt}")