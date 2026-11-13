"""
Test 4: physically motivated signals.

  * Damped oscillations: ODMD must recover both omega and the decay rate gamma
    (|lambda| < 1), not only the phase.
  * Quantum overlap s(t) = <psi0|exp(-iHt)|psi0> for a random Hermitian H:
    recovered energies must match eig(H), ground_phase() must give E_0, and
    the amplitudes must reproduce the populations |<E_j|psi0>|^2.

    python -m unittest test_04_damped_and_quantum -v
"""
import unittest
import numpy as np

try:                         # imported as part of the package
    from . import _imports   # noqa: F401
except ImportError:          # run as a script / top-level module
    import _imports          # noqa: F401

from odmd import ODMD
import signals
from helpers import ODMDTestCase, omegas, decays, match_error, report, log, plot_fit, PLOT


class TestDamped(ODMDTestCase):

    def test_damped_modes(self):
        t, sig, truth = signals.damped()
        dt = truth["dt"]
        m = ODMD(d=100, svd_rank=-1, tol=1e-8, sorted_eigs="abs").fit(sig)
        report("damped", m, dt, sig)

        self.assertRank(m, truth["rank"])
        self.assertMatches(omegas(m, dt), truth["omegas"], 1e-8, "omega")
        self.assertMatches(-decays(m, dt), truth["gammas"], 1e-8, "gamma")
        if PLOT:
            plot_fit(t, sig, m, "damped oscillations")

    def test_damped_with_noise(self):
        t, sig, truth = signals.damped()
        dt = truth["dt"]
        m = ODMD(d=100, svd_rank=0, tol=None).fit(signals.add_noise(sig, 30, seed=0))
        e_w = match_error(omegas(m, dt), truth["omegas"])
        e_g = match_error(-decays(m, dt), truth["gammas"])
        log(f"30 dB: rank={m.operator.rank}, omega err={e_w:.1e}, gamma err={e_g:.1e}")
        # A decaying signal has a falling SNR along the record, so the automatic
        # rank can keep one extra (noise) mode; the physical modes must still be
        # accurate.
        self.assertGreaterEqual(m.operator.rank, truth["rank"])
        self.assertLessEqual(m.operator.rank, truth["rank"] + 1)
        self.assertLess(e_w, 1e-2)
        self.assertLess(e_g, 1e-2)


class TestQuantumOverlap(ODMDTestCase):
    """s(t) = <psi0|exp(-iHt)|psi0>; one shared fit checked from several angles."""

    @classmethod
    def setUpClass(cls):
        cls.t, cls.sig, cls.truth = signals.quantum_overlap()
        cls.dt = cls.truth["dt"]
        cls.model = ODMD(d=100, svd_rank=0, tol=1e-8, reduced=True).fit(cls.sig)

    def test_energies(self):
        energies = np.sort(-omegas(self.model, self.dt))
        log(f"eig(H):  {np.round(self.truth['energies'], 6)}")
        log(f"ODMD:    {np.round(energies, 6)}")
        self.assertEqual(energies.shape, self.truth["energies"].shape)
        np.testing.assert_allclose(energies, self.truth["energies"], atol=1e-8)

    def test_ground_state(self):
        E0 = self.model.ground_phase() / self.dt
        log(f"E0 from ground_phase(): {E0:.8f}")
        self.assertAlmostEqual(E0, self.truth["energies"].min(), delta=1e-8)

    def test_unitary(self):
        """Unitary evolution -> |lambda| = 1."""
        self.assertOnUnitCircle(self.model.eigs, 1e-10)

    def test_populations(self):
        """
        For s_k = sum_j w_j lambda_j^k, row 0 of the modes times the amplitudes
        gives back the populations w_j.
        """
        pops = self.model.modes[0] * self.model.amplitudes
        order = np.argsort(-omegas(self.model, self.dt))      # sort by energy
        pops = pops[order].real
        log(f"true weights: {np.round(self.truth['weights'], 6)}")
        log(f"ODMD weights: {np.round(pops, 6)}")
        np.testing.assert_allclose(pops, self.truth["weights"], atol=1e-8)

    def test_real_part_loses_energy_sign(self):
        """Using only Re s(t) duplicates each energy as +/-E: the sign is lost."""
        m = ODMD(d=100, svd_rank=0, tol=1e-8, reduced=True).fit(self.sig.real)
        found = np.sort(-omegas(m, self.dt))
        expected = np.sort(np.concatenate([self.truth["energies"], -self.truth["energies"]]))
        log(f"Re-only rank={m.operator.rank} (complex rank {self.truth['rank']})")
        self.assertRank(m, 2 * self.truth["rank"])
        np.testing.assert_allclose(found, expected, atol=1e-8)

    @unittest.skipUnless(PLOT, "set ODMD_PLOT=1 to show plots")
    def test_plot(self):
        plot_fit(self.t, self.sig, self.model, "quantum overlap")


if __name__ == "__main__":
    unittest.main(verbosity=2)
