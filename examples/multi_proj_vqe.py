# example to illustrate the use of active space projections in VQE to speedup calculation
from vqemulti.utils import get_hf_reference_in_fock_space, generate_reduced_hamiltonian
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.vqe import vqe
from vqemulti.preferences import Configuration
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from vqemulti.errors import NotConvergedError
from vqemulti.basis_projection import get_basis_overlap_matrix, project_basis, prepare_ansatz_for_restart, state_projection_into_pool
from vqemulti.density import get_density_matrix, density_fidelity
import time


Configuration().verbose = False

# molecule definition
def linear_h4_mol(distance, basis='sto-3g'):
    mol = MolecularData(geometry=[['H', [0, 0, 0]],
                                  ['H', [0, 0, distance]],
                                  ['H', [0, 0, 2 * distance]],
                                  ['H', [0, 0, 3 * distance]]],
                        basis=basis,
                        multiplicity=1,
                        charge=0,
                        description='H4')
    return mol

h4_molecule = linear_h4_mol(distance=3.0, basis='3-21g')

# run classical calculation
molecule = run_pyscf(h4_molecule, run_fci=True, nat_orb=False, guess_mix=False)

# show FullCI data
print('FullCI energy:', molecule.fci_energy)

# get properties from classical SCF calculation
n_electrons = molecule.n_electrons
n_orbitals = 4  # molecule.n_orbitals
hamiltonian_full = molecule.get_molecular_hamiltonian()


print('\nPOOL\n---------------------------')
pool = get_pool_singlet_sd(n_electrons=n_electrons, n_orbitals=n_orbitals)
pool.print_compact_representation()

# Initial data
print('N electrons', n_electrons)

# simulator
from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
simulator = Simulator(trotter=True,
                      trotter_steps=1,
                      test_only=True,
                      shots=100)


# Get Hartree Fock reference in Fock space
coefficients = None
precision = 1e-1

st = time.time()

while True:

    # print info
    print('N Orbitals', n_orbitals)
    print('precision', precision)

    # get hamiltonian and HF reference
    hamiltonian = generate_reduced_hamiltonian(hamiltonian_full, n_orbitals)
    hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

    try:
        result = vqe(hamiltonian,
                     pool,
                     hf_reference_fock,
                     coefficients=coefficients,
                     energy_threshold=precision,
                     energy_simulator=simulator,
                     )

    except NotConvergedError as e:
        result = e.results

    # print results
    print('VQE energy:', result["energy"])
    print('n_evaluations: ', result["f_evaluations"])
    print('n_coefficients: ', len(result["coefficients"]))
    print("Coefficients:", result["coefficients"])

    # fidelity
    density_matrix = get_density_matrix(result['coefficients'],
                                        result['ansatz'],
                                        hf_reference_fock,
                                        n_orbitals)

    print('Fidelity measure: {:5.2f}\n'.format(density_fidelity(molecule.fci_one_rdm, density_matrix)))

    n_orbitals += 1
    if n_orbitals > 5:
        break

    # projection into larger ative space
    state_operators = sum(op * coeff for op, coeff in zip(result['ansatz'], result['coefficients']))
    pool = get_pool_singlet_sd(n_electrons=n_electrons, n_orbitals=n_orbitals)  # new pool
    coefficients = state_projection_into_pool(pool, state_operators)

et = time.time()

print('='*40)

print("HF energy:", molecule.hf_energy)
print("Final VQE energy:", result["energy"])
print("FullCI energy:", molecule.fci_energy)

error = result["energy"] - molecule.fci_energy
print("Error:", error)
print('total running time: ', et - st, 's')
