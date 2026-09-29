# implementation of VQSM (https://doi.org/10.1103/6hpp-zl5h)
from openfermionpyscf import run_pyscf
from openfermion import MolecularData
from vqemulti.utils import get_hf_reference_in_fock_space, fermion_to_qubit
from vqemulti.ansatz.exponential import ExponentialAnsatz
from vqemulti.ansatz.hea_particle import HardwareEfficientAnsatz
from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
from vqemulti.optimizers import OptimizerParams
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
import numpy as np
import scipy



def orthogonalize_basis(H_mat, S_mat):
    """
    Construct an orthonormal basis from the non-orthogonal basis
    represented by the overlap matrix S using Cholesky.
    """

    L = scipy.linalg.cholesky(S_mat, lower=True)
    X = scipy.linalg.solve_triangular(L, H_mat, lower=True)  # L^-1 H
    H_orth = scipy.linalg.solve_triangular(L, X.conj().T, lower=True).conj().T  # X L^-dagger

    return H_orth

class SubspaceConverged(Exception):
    """Raised when the Krylov subspace has converged."""
    pass


# define optimizer
#optimizer = OptimizerParams(method='COBYLA', options={'rhobeg': 0.5})
optimizer = OptimizerParams(method='COBYLA', options={})

# simulator
# config = Configuration()
# config.verbose = 2

#backend = FakeTorino()
# service = QiskitRuntimeService()
# backend = service.backend('ibm_basquecountry')
backend = AerSimulator()

simulator = Simulator(trotter=True,
                      trotter_steps=1,
                      test_only=False,
                      shots=100000,
                      backend=backend,
                      #use_ibm_runtime=True
                      )
# use state vector
simulator = None

print('simulator: ', simulator)

distance = 1.0 # 0.74
hydrogen = MolecularData(geometry=[('H', [0.0, 0.0, 0.0]),
                                   ('H', [0.0, 0.0, distance],),
                                   #('H', [0.0, 0.0, distance*2],),
                                   #('H', [0.0, 0.0, distance*3],),
                                   ],
                         basis='3-21g',
                         #basis='sto-3g',
                         multiplicity=1,
                         charge=0,
                         description='molecule')

# run reference calculation
molecule = run_pyscf(hydrogen, run_fci=False, verbose=True, frozen_core=0, n_orbitals=4, run_casci=True, run_ccsd=True)

n_electrons = molecule.n_electrons
n_orbitals = molecule.n_orbitals
n_qubits = molecule.n_qubits

print('N_electrons: ', n_electrons)
print('N_orb: ', n_orbitals)

# get hamiltonian
hamiltonian = molecule.get_molecular_hamiltonian()
hamiltonian = fermion_to_qubit(hamiltonian)

# FCI energy
e_fci = molecule.casci_energy
print('e_fci: ', e_fci)
print('e_ccsd: ', molecule.ccsd_energy)

# params
n_dim = 5


# reference state
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, n_qubits)
ref = HardwareEfficientAnsatz(hf_reference_fock, init='zeros', n_terms=1, mixed_spin=True)

ener_ref = ref.get_energy(ref.parameters, hamiltonian, energy_simulator=simulator)
print("E0 =", ener_ref)

# initialize
basis = [ref]
H_mat = np.array([[ener_ref]], dtype=complex)
S_mat = np.array([[1.0]], dtype=complex)

def get_optimized_new_state(hamiltonian, basis, H_mat, S_mat, tol_conv=1e-8):

    # define trial ansatz
    trial_state = HardwareEfficientAnsatz(hf_reference_fock, init='random', n_terms=1, mixed_spin=True)

    n_step = len(H_mat)
    assert n_step == len(basis)

    # expand H and S matrices
    H_mat = np.pad(H_mat, ((0, 1), (0, 1)))
    S_mat = np.pad(S_mat, ((0, 1), (0, 1)))
    S_mat[k, k] = 1.0

    def cost_function(coefficients):
        trial_state.parameters = coefficients

        # build trial H and S matrices
        for i in range(n_step):
            H_mat[i, n_step] = basis[i].get_matrix_element(hamiltonian, trial_state, simulator=simulator)
            H_mat[n_step, i] = H_mat[i, n_step].conj()

            S_mat[i, n_step] = basis[i].get_overlap(trial_state, simulator=simulator)
            S_mat[n_step, i] = S_mat[i, n_step].conj()

        # check convergence
        eigvals = scipy.linalg.eigvalsh(S_mat)
        if min(abs(eigvals)) < tol_conv * max(abs(eigvals)):
            raise SubspaceConverged

        # orthogonalize
        H_orth = orthogonalize_basis(H_mat, S_mat)
        # print('H_orth: ', H_orth)

        # define cost
        return -np.abs(H_orth[n_step-1, n_step])

    results = scipy.optimize.minimize(cost_function,
                                      trial_state.parameters,
                                      method=optimizer.method,
                                      options=optimizer.options,
                                      tol=1e-2)

    trial_state.parameters = results.x

    # return new basis state
    return trial_state


# main loop
energy_list = [ener_ref]
for k in range(1, n_dim):
    print('k:', k)

    try:
        phi = get_optimized_new_state(hamiltonian, basis, H_mat, S_mat)
    except SubspaceConverged:
        print('Converged at k:', k)
        break

    # add new state to basis
    basis.append(phi)

    # expand H and S  matrices
    H_mat = np.pad(H_mat, ((0, 1), (0, 1)))
    S_mat = np.pad(S_mat, ((0, 1), (0, 1)))
    S_mat[k, k] = 1.0

    # fill H and S  matrices
    for i in range(k):
        H_mat[i, k] = basis[i].get_matrix_element(hamiltonian, basis[k], simulator=simulator)
        S_mat[i, k] = basis[i].get_overlap(basis[k], simulator=simulator)

        H_mat[k, i] = H_mat[i, k].conj()
        S_mat[k, i] = S_mat[i, k].conj()

    H_mat[k, k] = basis[k].get_energy(basis[k].parameters, hamiltonian, energy_simulator=simulator)

    print(np.round(H_mat, decimals=3))
    print(np.round(S_mat, decimals=3))

    energies, coefficients = scipy.linalg.eigh(H_mat, S_mat)

    E0 = energies[0]
    c = coefficients[:, 0]

    print("E0 =", E0)
    print("coefficients =", c)
    energy_list.append(E0)


print('\n ----- FINAL RESULTS -----')

# orthogonalize
H_orth = orthogonalize_basis(H_mat, S_mat)
print("H_orth =")
print(np.round(H_orth, 3).real)
print(np.round(H_orth, 3).imag)

print('\nEnergies:', np.linalg.eigvalsh(H_orth))


# plot data
plt.plot(energy_list)
plt.ylabel('Energy [Ha]')
plt.xlabel('Number of steps')
plt.show()

