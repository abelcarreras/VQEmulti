# example of He2 calculation using adaptVQE method
from vqemulti.utils import get_hf_reference_in_fock_space, generate_reduced_hamiltonian
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
from vqemulti.energy import get_adapt_vqe_energy
from vqemulti.energy.simulation import simulate_adapt_vqe_variance
from vqemulti.utils import store_hamiltonian, load_hamiltonian
from vqemulti.pool.tools import OperatorList

import matplotlib.pyplot as plt

n_electrons = 4
n_orbitals = 4  # molecule.n_orbitals
# store_hamiltonian(hamiltonian, file='hamiltonian.npz')


hamiltonian = load_hamiltonian(file='hamiltonian.npz')
hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals)  # using active space


# set an ultra simplified Hamiltonian
from openfermion.ops import InteractionOperator


#import numpy as np
#hamiltonian = InteractionOperator(0, np.ones((1, 1)), np.ones((1, 1, 1 ,1)))
##print(hamiltonian)

# Choose specific pool of operators for adapt-VQE
pool = get_pool_singlet_sd(n_electrons=n_electrons, n_orbitals=n_orbitals)

# Get Hartree Fock reference in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

simulator_exact = Simulator(trotter=True, test_only=True, hamiltonian_grouping=False)

variance = simulate_adapt_vqe_variance([], OperatorList([]), hf_reference_fock, hamiltonian, simulator_exact)
# print('variance: ', variance)
# variance = 0.16030744


def get_energy_sampled(coeff, s_opt):

    global variance
    n_coeff = len(coeff)
    ansatz = pool[:n_coeff]

    # variance = simulate_adapt_vqe_variance(coeff, ansatz, hf_reference_fock, hamiltonian, simulator_exact)
    # print('variance_in: ', variance)

    n_shots = int(variance / s_opt ** 2)
    n_shots = max(1, n_shots)

    # define simulator
    simulator = Simulator(trotter=True, test_only=False, hamiltonian_grouping=True, shots=n_shots)


    energy = get_adapt_vqe_energy(coeff,
                                  ansatz,
                                  hf_reference_fock,
                                  hamiltonian,
                                  energy_simulator=simulator
                                  )

    return energy


if __name__ == '__main__':
    print('Energy VQE: {:.8f}'.format(get_energy_sampled([0.1], 0.1)))

