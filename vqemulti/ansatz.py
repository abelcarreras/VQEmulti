from openfermion import FermionOperator, hermitian_conjugated, normal_ordered
from vqemulti.pool.tools import OperatorList
from vqemulti.utils import fermion_to_qubit
import numpy as np


def get_ucc_ansatz(t1, t2, tolerance=1e-6, use_qubit=False, full_amplitudes=False):
    """
    Get unitary coupled cluster ansatz from 1-excitation and 2-excitation CC amplitudes

    :param t1: 1-excitation amplitudes in spin-orbital basis ( n x n )
    :param t2: 1-excitation amplitudes in spin-orbital basis ( n x n x n x n )
    :param tolerance: amplitude cutoff to include term in the ansatz
    :param use_qubit: define UCC ansatz in qubit operators
    :return: coefficients as list and ansatz as OperatorsList
    """

    operator_tot = FermionOperator()

    # 1-exciation terms
    if t1 is not None:
        n_spin_orbitals = len(t1)
        for i in range(n_spin_orbitals):
            for j in range(n_spin_orbitals):
                if abs(t1[i,j]) > tolerance:
                    operator = FermionOperator('{}^ {}'.format(i, j))
                    if not full_amplitudes:
                        operator_tot += t1[i, j] * operator - hermitian_conjugated(t1[i, j] * operator)
                    else:
                        operator_tot += t1[i, j] * operator

    # 2-excitation terms
    if t2 is not None:
        n_spin_orbitals = len(t2)
        for i in range(n_spin_orbitals):
            for j in range(n_spin_orbitals):
                for k in range(n_spin_orbitals):
                    for l in range(n_spin_orbitals):  # avoid duplicates
                        if np.mod(i, 2) + np.mod(k, 2) == np.mod(j, 2) + np.mod(l, 2):  # keep spin symmetry
                            if abs(t2[i, j, k, l]) > tolerance:
                                operator = FermionOperator('{}^ {} {}^ {}'.format(i, j, k, l))
                                if not full_amplitudes:
                                    operator_tot += 0.5 * t2[i, j, k, l] * operator - hermitian_conjugated(0.5 * t2[i, j, k, l] * operator)

                                else:
                                    operator_tot += 0.5 * t2[i, j, k, l] * operator

    assert normal_ordered(operator_tot + hermitian_conjugated(operator_tot)).isclose(FermionOperator.zero(), tolerance)

    operators = [operator_tot]
    coefficients = [1.0]

    if use_qubit:
        operators = [fermion_to_qubit(op) for op in operators]

    ansatz = OperatorList(operators, normalize=False, antisymmetrize=False)

    return coefficients, ansatz


def get_basis_change_exp(U_test, tolerance=1e-6, use_qubit=False):
    """
    get the generator of the unitary transformation of the basis change U_test

    :param U_test: rotation matrix [a_i^ a_j]
    :param tolerance: tolerance
    :return: sparse matrix representation of the generator
    """
    from openfermion import FermionOperator, hermitian_conjugated, get_sparse_operator, normal_ordered
    from numpy.testing import assert_almost_equal
    import scipy as sp

    kappa = -1. * sp.linalg.logm(U_test)  # convention

    assert_almost_equal(U_test, sp.sparse.linalg.expm(-kappa), err_msg='in generator K ')
    assert np.allclose(kappa + kappa.conj().T, 0), "kappa not anti-hermitian!"

    return get_ucc_ansatz(kappa, None, full_amplitudes=True, tolerance=tolerance, use_qubit=use_qubit)


def get_ucj_ansatz(t2, t1=None, tolerance=1e-20):
    """
    Get unitary coupled Jastrow ansatz from 1-excitation and 2-excitation CC amplitudes

    :param t1: 1-excitation amplitudes in spin-orbital basis ( n x n )
    :param t2: 1-excitation amplitudes in spin-orbital basis ( n x n x n x n )
    :param tolerance: amplitude cutoff to include term in the ansatz
    :return: coefficients as list and ansatz as OperatorsList
    """
    from jastrow.factor import double_factorized_t2
    from jastrow.basis import get_spin_matrix
    from jastrow.basis import get_t2_spinorbitals_absolute_full, get_t1_spinorbitals
    import numpy as np

    diag_coulomb_mats, orbital_rotations = double_factorized_t2(t2)  # a_j a_l a_i^ a_k^

    # print(orbital_rotations.shape)
    norb = orbital_rotations.shape[-1]
    n_qubits = norb * 2

    coefficients = []
    operators = []

    if t1 is not None:
        t1_spin = get_t1_spinorbitals(t1)  # a_j a_i^ -> a_i^ a_j
        operator_tot = FermionOperator()
        for i in range(n_qubits):
            for j in range(n_qubits):
                if abs(t1_spin[i,j]) > tolerance:
                    operator = FermionOperator('{}^ {}'.format(i, j))
                    operator_tot += t1_spin[i, j] * operator - hermitian_conjugated(t1_spin[i, j] * operator)

        coefficients.append(1.0)
        operators.append(operator_tot)

    for diag, U in zip(diag_coulomb_mats, orbital_rotations):
        for U_i, diag_i in zip(U, diag):

            z_mat = np.zeros((norb, norb, norb, norb), dtype=complex)
            for i in range(norb):
                for j in range(norb):
                    z_mat[i, i, j, j] = -1j * diag_i[i, j]  # a_i^ a_j a_k^ a_l

            # jastrow
            spin_jastrow = get_t2_spinorbitals_absolute_full(z_mat)  # a_i^ a_j a_k^ a_l -> a_i^ a_j a_k^ a_l
            coefficients_j, ansatz_j = get_ucc_ansatz(None, spin_jastrow, full_amplitudes=True)

            # basis change
            U_spin = get_spin_matrix(U_i.T)
            coefficients_u, ansatz_u = get_basis_change_exp(U_spin)  # a_i^ a_j

            # add to ansatz
            coefficients += [c for c in coefficients_u]
            operators += [op for op in ansatz_u]

            coefficients += coefficients_j
            operators += [op for op in ansatz_j]

            coefficients += [-c for c in coefficients_u]
            operators += [op for op in ansatz_u]

    operators = operators[::-1]
    coefficients = coefficients[::-1]
    ansatz = OperatorList(operators, normalize=False, antisymmetrize=False)

    return coefficients, ansatz


if __name__ == '__main__':

    from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
    from openfermionpyscf import run_pyscf
    from openfermion import MolecularData
    from vqemulti.utils import get_hf_reference_in_fock_space
    from vqemulti.energy import get_vqe_energy, get_adapt_vqe_energy
    from vqemulti.operators import n_particles_operator, spin_z_operator, spin_square_operator


    simulator = Simulator(trotter=False,
                          trotter_steps=1,
                          test_only=True,
                          hamiltonian_grouping=True,
                          use_estimator=True)

    hydrogen = MolecularData(geometry=[('H', [0.0, 0.0, 0.0]),
                                       ('H', [2.0, 0.0, 0.0]),
                                       ('H', [4.0, 0.0, 0.0]),
                                       ('H', [6.0, 0.0, 0.0])],
                        basis='sto-3g',
                        multiplicity=1,
                        charge=0,
                        description='molecule')

    # run classical calculation
    n_frozen_orb = 0
    n_total_orb = 4
    molecule = run_pyscf(hydrogen, run_fci=False, nat_orb=False, guess_mix=False, verbose=True,
                         frozen_core=n_frozen_orb, n_orbitals=n_total_orb, run_ccsd=True)

    hamiltonian = molecule.get_molecular_hamiltonian()

    n_electrons = molecule.n_electrons - n_frozen_orb * 2
    n_orbitals = n_total_orb - n_frozen_orb  # molecule.n_orbitals
    n_qubits = n_orbitals * 2

    hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, n_qubits)

    hf_energy = get_vqe_energy([], [], hf_reference_fock, hamiltonian, None)
    print('energy HF: ', hf_energy)

    coefficients, ansatz = get_ucc_ansatz(molecule.ccsd_single_amps, molecule.ccsd_double_amps)

    # simulator = None
    energy = get_adapt_vqe_energy(coefficients,
                                  ansatz,
                                  hf_reference_fock,
                                  hamiltonian,
                                  simulator)

    simulator.print_statistics()
    print('UCC ANSATZ energy: ', energy)

    ccsd = molecule._pyscf_data.get('ccsd', None)
    coefficients, ansatz = get_ucj_ansatz(ccsd.t2, t1=ccsd.t1)

    energy = get_adapt_vqe_energy(coefficients,
                                  ansatz,
                                  hf_reference_fock,
                                  hamiltonian,
                                  simulator)

    simulator.print_statistics()
    for circ in simulator.get_circuits():
        print(circ)
    print('Jastrow ANSATZ energy: ', energy)

    from vqemulti.energy.simulation import simulate_energy_sqd

    # energy = simulate_energy_sqd(coefficients, ansatz, hf_reference_fock, hamiltonian, simulator, n_electrons)

    n_particle = get_vqe_energy(coefficients,
                                ansatz,
                                hf_reference_fock,
                                n_particles_operator(n_orbitals),
                                simulator)

    spin_z = get_vqe_energy(coefficients,
                            ansatz,
                            hf_reference_fock,
                            spin_z_operator(n_orbitals),
                            simulator)

    spin_square = get_vqe_energy(coefficients,
                                 ansatz,
                                 hf_reference_fock,
                                 spin_square_operator(n_orbitals),
                                 simulator)

    print('n_particles: ', n_particle)
    print('Sz: ', spin_z)
    print('S2: ', spin_square)
