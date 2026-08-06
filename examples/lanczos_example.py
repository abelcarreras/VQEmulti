import matplotlib.pyplot as plt
import numpy as np
from openfermionpyscf import run_pyscf
from openfermion import MolecularData
from vqemulti.utils import get_hf_reference_in_fock_space
from vqemulti.utils import get_dmrg_energy, fermion_to_qubit
from vqemulti.ansatz.exponential import ExponentialAnsatz
from vqemulti.utils import get_sparse_operator
from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
from qiskit_aer import AerSimulator
from openfermion import QubitOperator

# cache data
cache_data = {}

# Define the time values
n_kr_dimensions = 11  # number of dimensions in krylov space
dt = 2.5  # time evolution steps


# simulator
# config = Configuration()
# config.verbose = 2

#backend = FakeTorino()
# service = QiskitRuntimeService()
# backend = service.backend('ibm_basquecountry')
backend = AerSimulator()

simulator = Simulator(trotter=True,
                      trotter_steps=1,
                      test_only=True,
                      shots=1000000,
                      backend=backend,
                      #use_ibm_runtime=True
                      )


distance = 2.0 # 0.74
hydrogen = MolecularData(geometry=[('H', [0.0, 0.0, 0.0]),
                                   ('H', [0.0, 0.0, distance])],
                         basis='3-21g',
                         multiplicity=1,
                         charge=0,
                         description='molecule')

# run reference calculation
molecule = run_pyscf(hydrogen, run_fci=False, nat_orb=False, guess_mix=False, verbose=True,
                               frozen_core=0, n_orbitals=4, run_ccsd=True, run_casci=True)

n_electrons = molecule.n_electrons
n_orbitals = molecule.n_orbitals
n_qubits = molecule.n_qubits

print('N_electrons: ', n_electrons)
print('N_orb: ', n_orbitals)

# get hamiltonian
hamiltonian = molecule.get_molecular_hamiltonian()
hamiltonian_te = fermion_to_qubit(hamiltonian)
print('H terms:', len(hamiltonian_te.terms))
hamiltonian_te.compress(4e-2)
print('H terms compress:', len(hamiltonian_te.terms))

# FCI energy
e_fci = molecule.casci_energy
print('e_fci: ', e_fci)

# reference
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, n_qubits)


def get_hamiltonian_element_exact(index_1, index_2):

    generator = [1j * hamiltonian_te]
    phi = ExponentialAnsatz([dt * index_1], generator, hf_reference_fock)
    psi = ExponentialAnsatz([dt * index_2], generator, hf_reference_fock)

    bra = phi.get_state_vector().transpose().conj()
    ket = psi.get_state_vector()

    sparse_hamiltonian = get_sparse_operator(hamiltonian, n_qubits)

    return np.sum(bra @ sparse_hamiltonian @ ket)


def get_overlap_element_exact(index_1, index_2):

    generator = [1j * hamiltonian_te]
    phi = ExponentialAnsatz([dt * index_1], generator, hf_reference_fock)
    psi = ExponentialAnsatz([dt * index_2], generator, hf_reference_fock)

    bra = phi.get_state_vector().transpose().conj()
    ket = psi.get_state_vector()

    return np.sum(bra @ ket)


def get_hamiltonian_element_simulator(index_1, index_2):

    generator = [1j * hamiltonian_te]
    phi = ExponentialAnsatz([dt * index_1], generator, hf_reference_fock)
    psi = ExponentialAnsatz([dt * index_2], generator, hf_reference_fock)
    hamiltonian_qubit = fermion_to_qubit(hamiltonian)

    expectation_value, std_error = simulator.get_operator_matrix_element(hamiltonian_qubit,
                                                                         phi.get_preparation_gates(simulator),
                                                                         psi.get_preparation_gates(simulator),
                                                                         compute_imag=True,
                                                                         n_qubits=n_qubits,
                                                                         )

    return expectation_value


def get_overlap_element_simulator(index_1, index_2):

    generator = [1j * hamiltonian_te]
    phi = ExponentialAnsatz([dt * index_1], generator, hf_reference_fock)
    psi = ExponentialAnsatz([dt * index_2], generator, hf_reference_fock)

    identity = QubitOperator(())

    expectation_value, std_error = simulator.get_operator_matrix_element(identity,
                                                                         phi.get_preparation_gates(simulator),
                                                                         psi.get_preparation_gates(simulator),
                                                                         compute_imag=True,
                                                                         n_qubits=n_qubits,
                                                                         )

    return expectation_value


def get_hs_matrices(n_dim, type='exact'):
    h_matrix = np.identity(n_dim, dtype=complex)
    s_matrix = np.identity(n_dim, dtype=complex)
    for i in range(n_dim):
        for j in range(i, n_dim):

            # cache data
            if type in cache_data:
                if i < len(cache_data[type]['h_matrix']) and j < len(cache_data[type]['h_matrix']):
                    h_matrix[i, j] = cache_data[type]['h_matrix'][i, j]
                    s_matrix[i, j] = cache_data[type]['s_matrix'][i, j]
                    h_matrix[j, i] = h_matrix[i, j].conjugate()
                    s_matrix[j, i] = s_matrix[i, j].conjugate()
                    continue

            if type == 'exact':
                h_matrix[i, j] = get_hamiltonian_element_exact(i, j)
                s_matrix[i, j] = get_overlap_element_exact(i, j) if i != j else 1.0

            if type == 'simulator':
                h_matrix[i, j] = get_hamiltonian_element_simulator(i, j)
                s_matrix[i, j] = get_overlap_element_simulator(i, j) if i != j else 1.0

            h_matrix[j, i] = h_matrix[i, j].conjugate()
            s_matrix[j, i] = s_matrix[i, j].conjugate()

    if type in cache_data:
        if len(cache_data[type]['h_matrix']) > len(h_matrix):
            return s_matrix, h_matrix

    cache_data[type] = {'h_matrix': h_matrix, 's_matrix': s_matrix}

    # symmetrize
    # h_matrix = (h_matrix + h_matrix.T.conj())/2
    # s_matrix = (s_matrix + s_matrix.T.conj())/2

    print('-----------', type)
    print('h_real')
    print(h_matrix.real)
    print('h_imag')
    print(h_matrix.imag)
    print('s_real')
    print(s_matrix.real)
    print('s_imag')
    print(s_matrix.imag)

    return s_matrix, h_matrix


def effective_rank(matrix, threshold=1e-12):
    # Eigenvalues of a symmetric positive semidefinite matrix
    eigvals = np.linalg.eigvalsh(matrix)
    eigvals = eigvals[eigvals > threshold]  # Discard small/negative values
    p = eigvals / np.sum(eigvals)
    entropy = -np.sum(p * np.log(p))
    return np.exp(entropy)

def get_regularized_matrices(s_matrix, h_matrix, threshold=1e-2):

    # Congruence transformation (Lowdin method)
    eigvals, eigvecs = np.linalg.eigh(s_matrix)
    # print('eigvals: ', eigvals)
    # Apply threshold to eigenvalues to avoid numerical instability
    idx = eigvals > threshold
    S_inv_sqrt = eigvecs[:, idx] @ np.diag(eigvals[idx] ** -0.5) @ eigvecs[:, idx].T
    # print('shape_S_inv: ', eigvecs[:, idx].shape)

    # Transform the tensor to the orthonormal basis
    h_eff = S_inv_sqrt.conj().T @ h_matrix @ S_inv_sqrt
    s_eff = S_inv_sqrt.conj().T @ s_matrix @ S_inv_sqrt

    return s_eff, h_eff


energies_exact = []
energies_hadamard_test = []
effective_rank_list = []
inv_cond_numbers_list = []

for n_dim in range(1, n_kr_dimensions + 1, 1):

    s_matrix, h_matrix = get_hs_matrices(n_dim, type='simulator')

    efective_rank = effective_rank(s_matrix)/n_dim
    inv_cond_numbers = 1/np.linalg.cond(s_matrix)

    # hadamard trotter
    s_matrix, h_matrix = get_regularized_matrices(s_matrix, h_matrix)
    test_gs = np.linalg.eigvalsh(h_matrix)[0]
    # test_gs = get_energy(s_matrix, h_matrix)
    energies_hadamard_test.append(test_gs)
    effective_rank_list.append(efective_rank)
    inv_cond_numbers_list.append(inv_cond_numbers)
    print('GS energy H test: ', test_gs)

    # exact trotter
    s_matrix, h_matrix = get_hs_matrices(n_dim, type='exact')
    s_matrix, h_matrix = get_regularized_matrices(s_matrix, h_matrix)

    test_gs = np.linalg.eigvalsh(h_matrix)[0]
    #test_gs = get_energy(s_matrix, h_matrix)
    energies_exact.append(test_gs)

    print('GS energy exact: ', test_gs)
    print('shape H: ', h_matrix.shape)
    print('Normalized efective rank: ', efective_rank)
    print('Inv Condition numbers: ', inv_cond_numbers)

# plot data
plt.plot(range(1, n_kr_dimensions + 1, 1), energies_exact, 'o-', label='exact trotter')
plt.plot(range(1, n_kr_dimensions + 1, 1), energies_hadamard_test, 'o-', label='hadamard test')
plt.legend()
plt.axhline(y=e_fci, color='r', linestyle='--')
plt.xlabel('Krylov space dimension')
plt.ylabel('Energy [Ha]')

plt.figure()
plt.title('Totter analysis')
plt.plot(range(1, n_kr_dimensions + 1, 1), effective_rank_list, 'o-', label='Norm effec rank')
plt.plot(range(1, n_kr_dimensions + 1, 1), inv_cond_numbers_list, 'o-', label='Inverse cond num')
plt.legend()
plt.ylim(-0.05, 1.05)
plt.xlabel('Krylov space dimension')
# plt.ylabel('Normlized effective rank]')

plt.figure()
plt.title('Effective rank')
plt.plot(range(1, n_kr_dimensions + 1, 1), [x*(i+1) for i, x in enumerate(effective_rank_list)], 'o-', label='effec rank')
plt.xlabel('Krylov space dimension')
plt.ylabel('Effective rank')
plt.show()
