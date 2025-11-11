# Example of using basis projection to project the WF
# obtained from a previous calculation with a small basis [3-21G] and a restricted active space [3 orb]
# into a large one [6-31G] with a full active space [4 orb]
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.utils import generate_reduced_hamiltonian, get_hf_reference_in_fock_space
from vqemulti.basis_projection import get_basis_overlap_matrix, project_basis, prepare_ansatz_for_restart
from vqemulti import adaptVQE


# molecule definition
he2_molecule = MolecularData(geometry=[['He', [0, 0, 0]],
                                       ['He', [0, 0, 1.0]]],
                             basis='3-21g',
                             multiplicity=1,
                             charge=0,
                             description='He2')


# run classical calculation
molecule_1 = run_pyscf(he2_molecule, run_fci=True, run_ccsd=True)

# get properties from classical SCF calculation
n_electrons = molecule_1.n_electrons
n_orbitals = 3 # molecule_1.n_orbitals

hamiltonian = molecule_1.get_molecular_hamiltonian()
hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals)

# print data
print('n_electrons: ', n_electrons)
print('n_orbitals: ', n_orbitals)
print('n_qubits: ', hamiltonian.n_qubits)

# Get a pool of operators for adapt-VQE
operators_pool = get_pool_singlet_sd(n_electrons=n_electrons, n_orbitals=n_orbitals)

# Get Hartree Fock reference in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

# Simulator
from vqemulti.simulators.penny_simulator import PennylaneSimulator as Simulator

simulator = Simulator(trotter=True,
                      trotter_steps=1,
                      test_only=True,
                      shots=1000)

# FIRST CALCULATION
result_first = adaptVQE(hamiltonian,
                        operators_pool, # fermionic operators
                        hf_reference_fock,
                        threshold=0.1,
                        # energy_simulator=simulator,
                        # gradient_simulator=simulator
                        )


print('Energy HF: {:.8f}'.format(molecule_1.hf_energy))
print('Energy adaptVQE: ', result_first['energy'])
print('Energy FullCI: ', molecule_1.fci_energy)
print('Coefficients:', result_first['coefficients'])

print('\n\nSecond calculation\n')

he2_molecule.basis = '6-31g'
molecule_2 = run_pyscf(he2_molecule, run_fci=True, run_ccsd=True)

# get properties from classical SCF calculation
n_electrons = molecule_2.n_electrons
n_orbitals = molecule_2.n_orbitals_active

hamiltonian = molecule_2.get_molecular_hamiltonian()
hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals)

# print data
print('n_electrons: ', n_electrons)
print('n_orbitals: ', n_orbitals)
print('n_qubits:', hamiltonian.n_qubits)

# Get a pool of operators for adapt-VQE
operators_pool = get_pool_singlet_sd(n_electrons=n_electrons, n_orbitals=n_orbitals)

# Get Hartree Fock reference in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

# operator basis projection
print('previous ansatz')
print('coef: ', result_first['coefficients'])
print('ansatz: ', result_first['ansatz'])

ansatz = sum(op * coeff for op, coeff in zip(result_first['ansatz'], result_first['coefficients']))
basis_overlap_matrix = get_basis_overlap_matrix(molecule_1, molecule_2)
projected_ansatz = project_basis(ansatz, basis_overlap_matrix, n_orb_2=n_orbitals)
restart_coefficients, restart_ansatz = prepare_ansatz_for_restart(projected_ansatz, max_val=1e-2)

print('projected ansatz')
print('coef: ', restart_coefficients)
print('ansatz: ', restart_ansatz)

# SECOND (restarted) CALCULATION
print('restarting calculation')
result = adaptVQE(hamiltonian,
                  operators_pool,                        # fermionic operators
                  hf_reference_fock,
                  threshold=0.1,
                  coefficients=restart_coefficients,   # projected restart coefficients
                  ansatz=restart_ansatz,               # projected restart ansatz
                  # energy_simulator=simulator,
                  # gradient_simulator=simulator
                  )

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