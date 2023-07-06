from vqemulti.energy.exact import exact_vqe_energy, exact_vqe_energy_gradient
from vqemulti.energy.simulation import simulate_vqe_energy


def get_vqe_energy(coefficients, ansatz, hf_reference_fock, hamiltonian, energy_simulator):
    if energy_simulator is None:
        return exact_vqe_energy(coefficients, ansatz, hf_reference_fock, hamiltonian)
    else:
        return simulate_vqe_energy(coefficients, ansatz, hf_reference_fock, hamiltonian, energy_simulator)
