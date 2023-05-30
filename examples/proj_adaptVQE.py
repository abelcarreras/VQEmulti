# Example of restarting a calculation using previous calculation
# using the ansatz (list of operators) and coefficients

from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from pool_definitions import get_pool_singlet_sd
from utils import generate_reduced_hamiltonian, get_hf_reference_in_fock_space
from basis_projection import get_basis_overlap_matrix, project_basis, prepare_ansatz_for_restart
from adapt_vqe import adaptVQE


h2_molecule = MolecularData(geometry=[['H', [0, 0, 0]],
                                      ['H', [0, 0, 1.74]]],
                            basis='sto-3g',
                            multiplicity=1,
                            charge=0,
                            description='H2')

# run classical calculation
molecule = run_pyscf(h2_molecule, run_fci=True, run_ccsd=True)

mo_mol_1 = molecule._pyscf_data['scf'].mo_coeff.T
mol_1 = molecule._pyscf_data['mol']

# get properties from classical SCF calculation
n_electrons = molecule.n_electrons
n_orbitals = 2  # molecule.n_orbitals

hamiltonian = molecule.get_molecular_hamiltonian()
hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals)

# print data
print('n_electrons: ', n_electrons)
print('n_orbitals: ', n_orbitals)
print('n_qubits:', hamiltonian.n_qubits)

# Get a pool of operators for adapt-VQE
operators_pool = get_pool_singlet_sd(electronNumber=n_electrons,
                                     orbitalNumber=n_orbitals)

# Get Hartree Fock reference in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

# Simulator
from simulators.penny_simulator import PennylaneSimulator as Simulator

simulator = Simulator(trotter=True,
                      trotter_steps=1,
                      test_only=True,
                      shots=1000)

# FIRST CALCULATION
result_first = adaptVQE(operators_pool, # fermionic operators
                        hamiltonian,    # fermionic hamiltonian
                        hf_reference_fock,
                        threshold=0.1,
                        energy_simulator=simulator,
                        gradient_simulator=simulator)


print('Energy HF: {:.8f}'.format(molecule.hf_energy))
print('Energy adaptVQE: ', result_first['energy'])
print('Energy FullCI: ', molecule.fci_energy)
print('Coefficients:', result_first['coefficients'])

print('\n\nSecond calculation\n')

h2_molecule.basis = '3-21g'

molecule_2 = run_pyscf(h2_molecule, run_fci=True, run_ccsd=True)

mo_mol_2 = molecule_2._pyscf_data['scf'].mo_coeff.T
mol_2 = molecule_2._pyscf_data['mol']

# get properties from classical SCF calculation
n_electrons = molecule_2.n_electrons
n_orbitals = 3  # molecule.n_orbitals

hamiltonian = molecule_2.get_molecular_hamiltonian()
hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals)

# print data
print('n_electrons: ', n_electrons)
print('n_orbitals: ', n_orbitals)
print('n_qubits:', hamiltonian.n_qubits)

# Get Hartree Fock reference in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

# operator basis projection
print('previous ansatz')
print('coef: ', result_first['coefficients'])
print('ansatz: ', result_first['ansatz'])

ansatz = sum(op * coeff for op, coeff in zip(result_first['ansatz'], result_first['coefficients']))

basis_overlap_matrix = get_basis_overlap_matrix(mol_1, mo_mol_1, mol_2, mo_mol_2)
projected_ansatz = project_basis(ansatz, basis_overlap_matrix, n_orb_2=n_orbitals)
restart_coefficients, restart_ansatz = prepare_ansatz_for_restart(projected_ansatz, max_val=1e-1)

print('projected ansatz')
print('coef: ', restart_coefficients)
print('ansatz: ', restart_ansatz)

# SECOND (restarted) CALCULATION
print('restarting calculation')
result = adaptVQE(operators_pool,                        # fermionic operators
                  hamiltonian,                           # fermionic hamiltonian
                  hf_reference_fock,
                  threshold=0.1,
                  coefficients=restart_coefficients,   # projected restart coefficients
                  ansatz=restart_ansatz,               # projected restart ansatz
                  energy_simulator=simulator,
                  gradient_simulator=simulator)

# FINAL RESULTS
print('Energy HF: {:.8f}'.format(molecule_2.hf_energy))
print('Energy adaptVQE: ', result['energy'])
print('Energy FullCI: ', molecule_2.fci_energy)

error = result['energy'] - molecule_2.fci_energy
print('Error:', error)

print('Ansatz:', result['ansatz'])
print('Coefficients:', result['coefficients'])
print('Operator Indices:', result['indices'])
print('Num operators: {}'.format(len(result['ansatz'])))
