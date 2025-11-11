from vqemulti.utils import get_hf_reference_in_fock_space, generate_reduced_hamiltonian, get_string_from_fermionic_operator
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.vqe import vqe
from vqemulti.preferences import Configuration
from openfermionpyscf import run_pyscf
from vqemulti.errors import NotConvergedError
from vqemulti.basis_projection import get_basis_overlap_matrix, project_basis, prepare_ansatz_for_restart


def project_pool(pool, proj_coef, proj_ansatz):
    new_coef = []
    for op in pool:
        coef = 0
        for c, op_2 in zip(proj_coef, proj_ansatz):
            if op == op_2:
                coef = c
                break
        new_coef.append(coef)
    return new_coef



vqe_energies = []
energies_fullci = []
energies_hf = []

Configuration().verbose = True

# molecule definition
from generate_mol import tetra_h4_mol, linear_h4_mol, square_h4_mol
h4_molecule = linear_h4_mol(distance=3.0, basis='3-21g')

# run classical calculation
molecule = run_pyscf(h4_molecule, run_fci=True, run_ccsd=True, nat_orb=False, guess_mix=False)


# get properties from classical SCF calculation
n_electrons = molecule.n_electrons
n_orbitals = 3  # molecule_2.n_orbitals

hamiltonian = molecule.get_molecular_hamiltonian()
hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals)

# print data
print('n_electrons: ', n_electrons)
print('n_orbitals: ', n_orbitals)
print('n_qubits:', hamiltonian.n_qubits)

# Get a pool of operators for adapt-VQE
pool = get_pool_singlet_sd(n_electrons=n_electrons, n_orbitals=n_orbitals)

# Get Hartree Fock reference in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
#from vqemulti.simulators.penny_simulator import PennylaneSimulator as Simulator
#from vqemulti.simulators.cirq_simulator import CirqSimulator as Simulator

simulator = Simulator(trotter=True,
                      trotter_steps=1,
                      # test_only=True,
                      shots=10000)

try:
    result = vqe(hamiltonian,
                 pool,
                 hf_reference_fock,
                 #opt_qubits=True,
                 energy_simulator=simulator
                 )

except NotConvergedError as e:
    result = e.results

# h4_molecule.basis = '3-21g'
# molecule_2 = run_pyscf(h4_molecule, run_fci=True, run_ccsd=True, nat_orb=False, guess_mix=False)
molecule_2 = molecule

n_electrons = molecule_2.n_electrons
n_orbitals = 5  # molecule_2.n_orbitals

hamiltonian = molecule_2.get_molecular_hamiltonian()
hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals)

# print data
print('n_orbitals: ', n_orbitals)
print('n_qubits:', hamiltonian.n_qubits)
pool = get_pool_singlet_sd(n_electrons=n_electrons, n_orbitals=n_orbitals)

# Get Hartree Fock reference in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

coefficients = project_pool(pool, result['coefficients'], result['ansatz'])

try:
    result = vqe(hamiltonian,
                 pool,
                 hf_reference_fock,
                 #opt_qubits=True,
                 energy_simulator=simulator,
                 coefficients=coefficients,
                 )

except NotConvergedError as e:
    result = e.results


# Compare error vs FullCI calculation
print('VQE energy: {:12.6}'.format(result['energy']))
error = result['energy'] - molecule_2.fci_energy
print('Error: {:12.6f}'.format(error))

# run results
print('Ansatz:')
for c, op in zip(result['coefficients'], result['ansatz']):
    print('{:12.6f}  '.format(c) + get_string_from_fermionic_operator(op))

#print("Coefficients:", result["coefficients"])
print('Num operators: {}'.format(len(result['ansatz'])))



simulator.print_statistics()
