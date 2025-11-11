# Example of computing the density matrix of an adaptVQE wave function
# to compare with FullCI density matrix and get a measure of fidelity
# using the ansatz (list of operators) and coefficients

from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from vqemulti.pool import get_pool_singlet_sd, get_pool_spin_complement_gsd
from vqemulti.utils import generate_reduced_hamiltonian, get_hf_reference_in_fock_space
from vqemulti.adapt_vqe import adaptVQE
from vqemulti.preferences import Configuration
from vqemulti.density import get_density_matrix, density_fidelity


r = 3.0
h2_molecule = MolecularData(geometry=[['H', [0, 0, 0]],
                                      ['H', [0, 0, r]],
                                      ['H', [0, 0, 2*r]],
                                      ['H', [0, 0, 3*r]]
                                      ],
                            basis='sto-3g',
                            multiplicity=1,
                            charge=0,
                            description='H2')

Configuration().mapping = 'jw'  # bk

# run classical calculation
molecule = run_pyscf(h2_molecule, run_fci=True, run_ccsd=True, nat_orb=False, guess_mix=False)

# get properties from classical SCF calculation
n_electrons = molecule.n_electrons
n_orbitals = molecule.n_orbitals_active

hamiltonian = molecule.get_molecular_hamiltonian()
hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals)

# print data
print('n_electrons: ', n_electrons)
print('n_orbitals: ', n_orbitals)
print('n_qubits:', hamiltonian.n_qubits)

# Get a pool of operators for adapt-VQE
operators_pool = get_pool_singlet_sd(n_electrons=n_electrons, n_orbitals=n_orbitals)
#operators_pool = get_pool_spin_complement_gsd(n_orbitals=n_orbitals)

# Get Hartree Fock reference in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

# Simulator
#from vqemulti.simulators.penny_simulator import PennylaneSimulator as Simulator
from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
#from vqemulti.simulators.cirq_simulator import CirqSimulator as Simulator

simulator = Simulator(trotter=True,
                      trotter_steps=1,
                      test_only=True,
                      shots=1000)

from vqemulti.energy import get_adapt_vqe_energy, exact_adapt_vqe_energy, simulate_adapt_vqe_energy
from vqemulti.pool.tools import OperatorList

# energy_test_sym = simulate_vqe_energy([1.3223, 2.1322, 1.435], OperatorList(operators_pool[0: 3]), hf_reference_fock, hamiltonian, simulator)
# energy_test = exact_vqe_energy([1.3223, 2.1322, 1.435], OperatorList(operators_pool[0: 3]), hf_reference_fock, hamiltonian)

operator_list = operators_pool[9: 10]
#operator_list = operators_pool[0: 1]

coefficients_list = [-2.56933]
OperatorList(operator_list).print_compact_representation()
energy_test_sym = simulate_adapt_vqe_energy(coefficients_list, operator_list, hf_reference_fock, hamiltonian, simulator)
energy_test = exact_adapt_vqe_energy(coefficients_list, operator_list, hf_reference_fock, hamiltonian)

print('comparison exact: ', energy_test)
print('comparison sym: ', energy_test_sym)
print('comparison diff: ', abs(energy_test - energy_test_sym))

print('Energy HF: {:.8f}'.format(molecule.hf_energy))

# exit()




# FIRST CALCULATION
result_first = adaptVQE(hamiltonian,
                        operators_pool, # fermionic operators
                        hf_reference_fock,
                        opt_qubits=False,
                        energy_simulator=simulator,
                        gradient_simulator=simulator
                        )


print('Energy HF: {:.8f}'.format(molecule.hf_energy))
print('Energy adaptVQE: ', result_first['energy'])
print('Energy FullCI: ', molecule.fci_energy)
print('Coefficients:', result_first['coefficients'])

import numpy as np
print('\nFullCI density matrix')
print(np.round(molecule.fci_one_rdm, decimals=6))
print(np.trace(molecule.fci_one_rdm))


density_matrix = get_density_matrix(result_first['coefficients'],
                                    result_first['ansatz'],
                                    hf_reference_fock,
                                    n_orbitals)

print('\nadaptVQE density matrix')
print(density_matrix)
print(np.trace(density_matrix))

# WF quantum fidelity (1: perfect match, 0: totally different)
fidelity = density_fidelity(density_matrix, molecule.fci_one_rdm)
print('\nFidelity measure: {:5.2f}'.format(fidelity))
