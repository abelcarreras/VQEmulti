# Example of computing the density matrix of an adaptVQE wave function
# to compare with FullCI density matrix and get a measure of fidelity
# using the ansatz (list of operators) and coefficients

from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.utils import generate_reduced_hamiltonian, get_hf_reference_in_fock_space
from vqemulti.adapt_vqe import adaptVQE
from vqemulti.density import get_density_matrix
from vqemulti.preferences import Configuration
from vqemulti.vqe import vqe

Configuration().mapping = 'bk'

r = 0.5
h2_molecule = MolecularData(geometry=[['H', [0, 0, 0]],
                                      ['H', [0, 0, r]],
                                      ['H', [0, 0, 2*r]],
                                      ['H', [0, 0, 3*r]]
                                      ],
                            basis='3-21g',
                            multiplicity=1,
                            charge=0,
                            description='H4')

# run classical calculation
molecule = run_pyscf(h2_molecule, run_fci=True, run_ccsd=True, nat_orb=False, guess_mix=False)

# get properties from classical SCF calculation
n_electrons = molecule.n_electrons
n_orbitals = 4 #  molecule.n_orbitals

hamiltonian = molecule.get_molecular_hamiltonian()
hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals)

# print data
print('n_electrons: ', n_electrons)
print('n_orbitals: ', n_orbitals)
print('n_qubits:', hamiltonian.n_qubits)

# Get a pool of operators for adapt-VQE
operators_pool = get_pool_singlet_sd(n_electrons=n_electrons, n_orbitals=n_orbitals)
# operators_pool = get_pool_spin_complement_gsd(n_orbitals=n_orbitals)

# Get Hartree Fock reference in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

# Simulator
# from vqemulti.simulators.penny_simulator import PennylaneSimulator as Simulator
from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator

simulator = Simulator(trotter=False,
                      # trotter_steps=2,
                      test_only=True,
                      shots=10000)

if True:
    # FIRST CALCULATION
    result_first = adaptVQE(hamiltonian,
                            operators_pool, # fermionic operators
                            hf_reference_fock,
                            opt_qubits=False,
                            energy_threshold=1e-4,
                            # coeff_tolerance=1,
                            energy_simulator=simulator,
                            #gradient_simulator=simulator
                            )


    print('Energy HF: {:.8f}'.format(molecule.hf_energy))
    print('Energy adaptVQE: ', result_first['energy'])
    print('Energy FullCI: ', molecule.fci_energy)
    #print('Coefficients:', result_first['coefficients'])



exit()

print('\n-----------------------------\n')
result_first = vqe(hamiltonian, operators_pool, hf_reference_fock, energy_threshold=1e-4, energy_simulator=simulator)


print('Energy HF: {:.8f}'.format(molecule.hf_energy))
print('Energy adaptVQE: ', result_first['energy'])
print('Energy FullCI: ', molecule.fci_energy)
#print('Coefficients:', result_first['coefficients'])

exit()
import numpy as np
print('\nFullCI density matrix')
print(np.round(molecule.fci_one_rdm, decimals=6))
print(np.trace(molecule.fci_one_rdm))
print(molecule.fci_two_rdm)


density_matrix = get_density_matrix(result_first['coefficients'],
                                    result_first['ansatz'],
                                    hf_reference_fock,
                                    n_orbitals)

print('\nadaptVQE density matrix')
print(density_matrix)
print(np.trace(density_matrix))

if False:
    density_2_matrix = get_2_order_density_matrix(result_first['coefficients'],
                                                  result_first['ansatz'],
                                                  hf_reference_fock,
                                                  n_orbitals)
    print(density_2_matrix)
    print(np.trace(density_2_matrix))


# WF fidelity proposal based in density matrix comparison (1: perfect match, 0: totally different)
#fidelity = 1 - (np.linalg.norm(molecule.fci_one_rdm - density_matrix)/n_electrons)

#print('\nFidelity measure: {:5.2f}'.format(fidelity))

from vqemulti.density import density_fidelity, density_fidelity_simple

fidelity = density_fidelity(density_matrix, molecule.fci_one_rdm)
print('\nFidelity measure original: {:5.2f}'.format(fidelity))

fidelity = density_fidelity_simple(density_matrix, molecule.fci_one_rdm)
print('\nFidelity measure new: {:5.2f}'.format(fidelity))
