# Example of restarting a calculation using previous calculation
# using the ansatz (list of operators) and coefficients
# this can be also understood as a simple projection if the basis set/operators pool is different

from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.utils import generate_reduced_hamiltonian, get_hf_reference_in_fock_space
from vqemulti.adapt_vqe import adaptVQE


h2_molecule = MolecularData(geometry=[['He', [0, 0, 0]],
                                       ['He', [0, 0, 1.0]]],
                             basis='6-31g',
                             multiplicity=1,
                             charge=0,
                             description='He2')

# run classical calculation
molecule = run_pyscf(h2_molecule, run_fci=True, run_ccsd=True)

# get properties from classical SCF calculation
n_electrons = molecule.n_electrons
n_orbitals = 4  # molecule.n_orbitals

hamiltonian = molecule.get_molecular_hamiltonian()
hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals)

# print data
print('n_electrons: ', n_electrons)
print('n_orbitals: ', n_orbitals)
print('n_qubits:', hamiltonian.n_qubits)

# Get a pool of operators for adapt-VQE
operators_pool = get_pool_singlet_sd(n_electrons=n_electrons,
                                     n_orbitals=n_orbitals)

# Get Hartree Fock reference in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

# Simulator
from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator

simulator = Simulator(trotter=True,
                      trotter_steps=1,
                      test_only=True,
                      shots=1000)

from vqemulti.method.adapt_vanila import AdapVanilla

method = AdapVanilla(gradient_threshold=1e-6,
                     diff_threshold=0,
                     coeff_tolerance=1e-10,
                     gradient_simulator=None,
                     operator_update_number=1,
                     operator_update_max_grad=2e-2,
                     )
# FIRST CALCULATION
result_first = adaptVQE(hamiltonian,  # fermionic hamiltonian
                          operators_pool,  # fermionic operators
                          hf_reference_fock,
                          energy_threshold=0.0001,
                          method=method,
                          max_iterations=20,
                          energy_simulator=None,
                          variance_simulator=None,
                          reference_dm=None,
                          optimizer_params=None)

# SECOND (restarted) CALCULATION
print('restarting calculation')
result = adaptVQE(hamiltonian, # fermionic hamiltonian
                  operators_pool, # fermionic operators
                  hf_reference_fock,
                  energy_threshold=0.0001,
                  method=method,
                  max_iterations=20,
                  energy_simulator=None,
                  variance_simulator=None,
                  coefficients=result_first['coefficients'],  # previous calculation coefficients
                  ansatz=result_first['ansatz'],  # previous calculation ansatz
                  reference_dm=None,
                  optimizer_params=None
                  )

# FINAL RESULTS
print('Energy HF: {:.8f}'.format(molecule.hf_energy))
print('Energy adaptVQE: ', result['energy'])
print('Energy FullCI: ', molecule.fci_energy)

error = result['energy'] - molecule.fci_energy
print('Error:', error)

print('Ansatz:', result['ansatz'])
print('Coefficients:', result['coefficients'])
print('Operator Indices:', result['indices'])
print('Num operators: {}'.format(len(result['ansatz'])))
