"""
Test 4: physically motivated signals.

  * Damped oscillations: ODMD must recover both omega and the decay rate gamma
    (|lambda| < 1), not only the phase.
  * Quantum overlap s(t) = <psi0|exp(-iHt)|psi0> for a random Hermitian H:
    recovered energies must match eig(H), ground_phase() must give E_0, and
    the amplitudes must reproduce the populations |<E_j|psi0>|^2.

    python test_04_damped_and_quantum.py [--plot]
"""
import sys
import numpy as np

from odmd import ODMD
import signals
from helpers import omegas, decays, match_error, report, plot_fit, run_tests


def test_damped_modes(plot=False):
    t, sig, truth = signals.damped()
    dt = truth["dt"]
    m = ODMD(d=100, svd_rank=-1, tol=1e-8, sorted_eigs="abs").fit(sig)
    report("damped", m, dt, sig)

    assert m.operator.rank == truth["rank"]
    assert match_error(omegas(m, dt), truth["omegas"]) < 1e-8
    assert match_error(-decays(m, dt), truth["gammas"]) < 1e-8
    if plot:
        plot_fit(t, sig, m, "damped oscillations")


def test_damped_with_noise():
    t, sig, truth = signals.damped()
    m = ODMD(d=100, svd_rank=0, tol=None).fit(signals.add_noise(sig, 30, seed=0))
    dt = truth["dt"]
    e_w = match_error(omegas(m, dt), truth["omegas"])
    e_g = match_error(-decays(m, dt), truth["gammas"])
    print(f"    30 dB: rank={m.operator.rank}, omega err={e_w:.1e}, gamma err={e_g:.1e}")
    # A decaying signal has a falling SNR along the record, so the automatic
    # rank can keep one extra (noise) mode; the physical modes must still be
    # accurate.
    assert truth["rank"] <= m.operator.rank <= truth["rank"] + 1
    assert e_w < 1e-2 and e_g < 1e-2


def test_quantum_energies_and_ground_state(plot=False):
    t, sig, truth = signals.quantum_overlap()
    dt = truth["dt"]
    m = ODMD(d=100, svd_rank=0, tol=1e-8, reduced=True).fit(sig)
    energies = np.sort(-omegas(m, dt))
    print(f"    eig(H):  {np.round(truth['energies'], 6)}")
    print(f"    ODMD:    {np.round(energies, 6)}")
    print(f"    E0 from ground_phase(): {m.ground_phase() / dt:.8f}")

    print("energies:", energies)
    print("energies shape:", np.shape(energies))
    print("truth energies:", truth["energies"])
    print("truth shape:", np.shape(truth["energies"]))

    assert np.allclose(energies, truth["energies"], atol=1e-8)

    assert np.allclose(energies, truth["energies"], atol=1e-8)
    assert abs(m.ground_phase() / dt - truth["energies"].min()) < 1e-8
    assert np.allclose(np.abs(m.eigs), 1.0, atol=1e-10), "unitary evolution -> |lambda| = 1"
    if plot:
        plot_fit(t, sig, m, "quantum overlap")


def test_quantum_populations():
    """
    For s_k = sum_j w_j lambda_j^k, the signal itself (row 0 of the modes)
    times the amplitudes gives back the populations w_j.
    """
    _, sig, truth = signals.quantum_overlap()
    dt = truth["dt"]
    m = ODMD(d=100, svd_rank=0, tol=1e-8, reduced=True).fit(sig)
    pops = m.modes[0] * m.amplitudes          # contribution of each mode to s_0
    order = np.argsort(-omegas(m, dt))        # sort by energy
    pops = pops[order].real
    print(f"    true weights: {np.round(truth['weights'], 6)}")
    print(f"    ODMD weights: {np.round(pops, 6)}")
    assert np.allclose(pops, truth["weights"], atol=1e-8)


def test_real_part_loses_energy_sign():
    """Using only Re s(t) duplicates each energy as +/-E: the sign is lost."""
    _, sig, truth = signals.quantum_overlap()
    dt = truth["dt"]
    m = ODMD(d=100, svd_rank=0, tol=1e-8, reduced=True).fit(sig.real)
    found = np.sort(-omegas(m, dt))
    expected = np.sort(np.concatenate([truth["energies"], -truth["energies"]]))
    print(f"    Re-only rank={m.operator.rank} (complex rank {truth['rank']})")
    assert m.operator.rank == 2 * truth["rank"]
    assert np.allclose(found, expected, atol=1e-8)


if __name__ == "__main__":
    ok = run_tests(globals(), plot="--plot" in sys.argv)
    sys.exit(0 if ok else 1)
