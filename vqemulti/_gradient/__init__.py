from vqemulti.gradient.exact import compute_gradient_vector, exact_vqe_energy_gradient, exact_adapt_vqe_energy_gradient
from vqemulti.gradient.simulation import simulate_gradient, simulate_vqe_energy_gradient


def get_adapt_vqe_energy_gradient(coefficients, ansatz, hf_reference_fock, hamiltonian, simulator=None):
    if simulator is None:
        return exact_adapt_vqe_energy_gradient(coefficients, ansatz, hf_reference_fock, hamiltonian)
    else:
        return simulate_vqe_energy_gradient(coefficients, ansatz, hf_reference_fock, hamiltonian, simulator)


def get_vqe_energy_gradient(coefficients, ansatz, hf_reference_fock, hamiltonian, simulator=None):
    if simulator is None:
        return exact_vqe_energy_gradient(coefficients, ansatz, hf_reference_fock, hamiltonian)
    else:
        return simulate_vqe_energy_gradient(coefficients, ansatz, hf_reference_fock, hamiltonian, simulator)
