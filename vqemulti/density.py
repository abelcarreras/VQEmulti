from vqemulti.utils import get_sparse_ket_from_fock, get_sparse_operator
from openfermion import FermionOperator
import numpy as np
import scipy


def get_density_matrix(ansatz, frozen_core=0):
    """
    Calculates the one particle density matrix in the molecular orbitals basis

    :param coefficients: the list of coefficients of the ansatz operators
    :param ansatz: ansatz expressed in qubit/fermion operators
    :param hf_reference_fock: HF reference in Fock space vector
    :param n_orbitals: number of molecular orbitals
    :return: exact energy
    """

    # get number of qubits
    n_orbitals = ansatz.n_qubits // 2 + frozen_core
    n_qubit = (n_orbitals - frozen_core) * 2

    # get state vector
    ket = ansatz.get_state_vector()
    bra = ket.transpose().conj()

    # initialize density matrices
    density_matrix_alpha = np.zeros((n_orbitals, n_orbitals))
    density_matrix_beta = np.zeros((n_orbitals, n_orbitals))
    for i in range(frozen_core):
        density_matrix_alpha[i, i] = 1
        density_matrix_beta[i, i] = 1

    # build density matrix
    for i in range(n_orbitals - frozen_core):
        for j in range(n_orbitals - frozen_core):
            # alpha
            density_operator = FermionOperator(((j*2, 1), (i*2, 0)))
            sparse_operator = get_sparse_operator(density_operator, n_qubit)
            density_matrix_alpha[i+frozen_core, j+frozen_core] = np.sum(bra * sparse_operator * ket).real

            # beta
            density_operator = FermionOperator(((j*2+1, 1), (i*2+1, 0)))
            sparse_operator = get_sparse_operator(density_operator, n_qubit)
            density_matrix_beta[i+frozen_core, j+frozen_core] = np.sum(bra * sparse_operator * ket).real

    return density_matrix_alpha + density_matrix_beta


def get_second_order_density_matrix(ansatz, frozen_core=0):
    """
    Calculates the 2 particles density matrix in molecular orbitals basis

    :param coefficients: the list of coefficients of the ansatz operators
    :param ansatz: ansatz expressed in qubit/fermion operators
    :param hf_reference_fock: HF reference in Fock space vector
    :param n_orbitals: number of molecular orbitals
    :return: exact energy
    """

    # get number of qubits
    n_orbitals = ansatz.n_qubits // 2
    n_qubit = (n_orbitals - frozen_core) * 2

    # get state vector
    ket = ansatz.get_state_vector()
    bra = ket.transpose().conj()

    # initialize density matrices
    density_matrix_alpha = np.zeros((n_orbitals, n_orbitals, n_orbitals, n_orbitals))
    density_matrix_beta = np.zeros((n_orbitals, n_orbitals, n_orbitals, n_orbitals))
    density_matrix_cross = np.zeros((n_orbitals, n_orbitals, n_orbitals, n_orbitals))

    for i in range(frozen_core):
        density_matrix_alpha[i, i, i, i] = 1
        density_matrix_beta[i, i, i, i] = 1

    # build density matrix
    for i1 in range(n_orbitals - frozen_core):
        for i2 in range(n_orbitals - frozen_core):

            for j1 in range(n_orbitals - frozen_core):
                for j2 in range(n_orbitals - frozen_core):

                    # alpha
                    density_operator = FermionOperator(((j1*2, 1), (j2*2, 1), (i2*2, 0), (i1*2, 0)))
                    #print(density_operator)
                    sparse_operator = get_sparse_operator(density_operator, n_qubit)
                    density_matrix_alpha[i1+frozen_core, i2+frozen_core,
                                         j2+frozen_core, j1+frozen_core] = np.sum(bra * sparse_operator * ket).real

                    # beta
                    density_operator = FermionOperator(((j1*2+1, 1), (j2*2+1, 1), (i2*2+1, 0), (i1*2+1, 0)))
                    sparse_operator = get_sparse_operator(density_operator, n_qubit)
                    density_matrix_beta[i1+frozen_core, i2+frozen_core,
                                        j2+frozen_core, j1+frozen_core] = np.sum(bra * sparse_operator * ket).real

                    # cross 1
                    density_operator = FermionOperator(((j1*2, 1), (j2*2+1, 1), (i2*2, 0), (i1*2+1, 0)))
                    sparse_operator = get_sparse_operator(density_operator, n_qubit)
                    density_matrix_cross[i1+frozen_core, i2+frozen_core,
                                         j2+frozen_core, j1+frozen_core] -= np.sum(bra * sparse_operator * ket).real
                    #print(np.sum(bra * sparse_operator * ket).real)

                    # cross 2
                    density_operator = FermionOperator(((j1*2+1, 1), (j2*2, 1), (i2*2+1, 0), (i1*2, 0)))
                    sparse_operator = get_sparse_operator(density_operator, n_qubit)
                    density_matrix_cross[i1+frozen_core, i2+frozen_core,
                                         j2+frozen_core, j1+frozen_core] -= np.sum(bra * sparse_operator * ket).real
                    #print(np.sum(bra * sparse_operator * ket).real)

                   # cross 3
                    density_operator = FermionOperator(((j1*2+1, 1), (j2*2, 1), (i2*2, 0), (i1*2+1, 0)))
                    sparse_operator = get_sparse_operator(density_operator, n_qubit)
                    density_matrix_cross[i1+frozen_core, i2+frozen_core,
                                         j2+frozen_core, j1+frozen_core] += np.sum(bra * sparse_operator * ket).real
                    #print(np.sum(bra * sparse_operator * ket).real)

                    # cross 4
                    density_operator = FermionOperator(((j1*2, 1), (j2*2+1, 1), (i2*2+1, 0), (i1*2, 0)))
                    sparse_operator = get_sparse_operator(density_operator, n_qubit)
                    density_matrix_cross[i1+frozen_core, i2+frozen_core,
                                         j2+frozen_core, j1+frozen_core] += np.sum(bra * sparse_operator * ket).real
                    #print(np.sum(bra * sparse_operator * ket).real)

    return 0.5 * (3*density_matrix_alpha + 3*density_matrix_beta + density_matrix_cross)


def density_fidelity(density_ref, density_vqe):
    """
    compute quantum fidelity based in 1p-density matrices.
    source: https://en.wikipedia.org/wiki/Fidelity_of_quantum_states.
    The result is a real number between 0 (low fidelity) and 1 (high fidelity)

    :param density_ref: density matrix of reference wave function (usually fullCI)
    :param density_vqe: density matrix of target wave function (usually VQE)
    :return: quantum fidelity measure
    """

    n_electrons = np.trace(density_ref)

    density_ref = np.array(density_ref)/n_electrons
    density_vqe = np.array(density_vqe)/n_electrons

    # padding with zero if different sizes
    n_pad = len(density_ref) - len(density_vqe)
    if n_pad > 0:
        density_vqe = np.pad(density_vqe, (0, n_pad), 'constant')
    else:
        density_ref = np.pad(density_ref, (0, -n_pad), 'constant')

    sqrt_vqe = scipy.linalg.sqrtm(density_vqe)

    dens = np.dot(np.dot(sqrt_vqe, density_ref), sqrt_vqe)
    trace = np.trace(scipy.linalg.sqrtm(dens))

    # alternative
    # sqrt_ref = scipy.linalg.sqrtm(density_ref)
    # dens = np.dot(sqrt_ref, sqrt_vqe)
    # trace = np.trace(scipy.linalg.sqrtm(np.dot(dens.T, dens)))

    return np.absolute(trace)**2


def density_fidelity_simple(density_ref, density_vqe):
    """
    compute Quantum fidelity based in 1p-density matrices.
    Simple implementation defined as 1 - (the norm of the difference between density matrices)
    The result is a real number between 0 (low fidelity) and 1 (high fidelity)

    :param density_ref: density matrix of reference wave function (usually fullCI)
    :param density_vqe: density matrix of target wave function (usually VQE)
    :return: fidelity measure
    """
    density_ref = np.array(density_ref)
    density_vqe = np.array(density_vqe)
    n_electrons = np.trace(density_ref)

    n_pad = len(density_ref) - len(density_vqe)
    fidelity = 1 - (np.linalg.norm(density_ref - np.pad(density_vqe, (0, n_pad), 'constant')) / n_electrons)

    return fidelity


if __name__ == '__main__':
    from openfermion import MolecularData
    from openfermionpyscf import run_pyscf
    from utils import generate_reduced_hamiltonian, get_hf_reference_in_fock_space
    from pool.singlet_sd import get_pool_singlet_sd
    from vqemulti.preferences import Configuration
    from vqemulti.ansatz.exponential import ExponentialAnsatz

    # set Bravyi-Kitaev mapping
    Configuration().mapping = 'bk'
    Configuration().verbose = True

    h2_molecule = MolecularData(geometry=[['He', [0, 0, 0]],
                                          ['He', [0, 0, 1.0]]],
                                basis='3-21g',
                                # basis='sto-3g',
                                multiplicity=1,
                                charge=-2,
                                description='H2')

    # run classical calculation
    molecule = run_pyscf(h2_molecule, run_fci=True, run_ccsd=True)

    # get properties from classical SCF calculation
    n_electrons = molecule.n_electrons
    n_orbitals = 4  # molecule.n_orbitals

    print('n_electrons: ', n_electrons)
    print('n_orbitals: ', n_orbitals)

    hamiltonian = molecule.get_molecular_hamiltonian()
    hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals, frozen_core=2)
    # print(hamiltonian)

    print('n_qubits:', hamiltonian.n_qubits)

    # Get UCCSD params
    uccsd_pool = get_pool_singlet_sd(n_electrons, n_orbitals, frozen_core=2)
    # uccsd_ansatz = []

    # Get reference Hartree Fock state
    hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits, frozen_core=2)
    print('hf reference', hf_reference_fock)

    # get ansatz
    initial_parameters = np.ones_like(uccsd_pool)
    uccsd_ansatz = ExponentialAnsatz(initial_parameters, uccsd_pool, hf_reference_fock)

    matrix = get_density_matrix(uccsd_ansatz)
    print('Density matrix:\n', matrix)

    print('eigenvals:', np.linalg.eigvals(matrix), sum(np.linalg.eigvals(matrix)))

