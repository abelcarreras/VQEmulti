from openfermion import FermionOperator, QubitOperator, hermitian_conjugated, normal_ordered
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


def jastrow_to_qubits(t2, tolerance=1e-6):
    """
    Get unitary coupled cluster ansatz from 1-excitation and 2-excitation CC amplitudes

    :param t2: 1-excitation amplitudes in spin-orbital basis ( n x n x n x n )
    :param tolerance: amplitude cutoff to include term in the ansatz
    :param use_qubit: define UCC ansatz in qubit operators
    :return: coefficients as list and ansatz as OperatorsList
    """

    operator_tot = QubitOperator()

    # 2-excitation terms
    n_spin_orbitals = len(t2)
    for i in range(n_spin_orbitals):
        for j in range(n_spin_orbitals):
                if abs(t2[i, i, j, j]) > tolerance:
                    operator = QubitOperator(())  # identity
                    operator -= QubitOperator(f'Z{i}')
                    operator -= QubitOperator(f'Z{j}')
                    operator += QubitOperator(f'Z{i} Z{j}')
                    operator_tot += 0.25 * t2[i, i, j, j] * operator

    # assert normal_ordered(operator_tot + hermitian_conjugated(operator_tot)).isclose(FermionOperator.zero(), tolerance)

    operators = [operator_tot]
    coefficients = [1.0]

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


def crop_local_amplitudes(amplitudes, n_neighbors=1):
    """
    remove interactions in the amplitudes between orbitals at larger distance than the set distance

    :param amplitudes: amplitudes matrix (occ x virt) / (occ x occ x virt x virt )
    :param n_neighbors: distance between orbitals
    :return: local amplitudes
    """

    if len(np.shape(amplitudes)) == 2:
        # a_j a_i^
        nocc, nvrt = amplitudes.shape
        local_amplitudes = np.array(amplitudes).copy()  # a_j a_i^
        for i in range(nocc):
            local_amplitudes[i, max(0, n_neighbors - nocc + i+1):] = 0

    elif len(np.shape(amplitudes)) == 4:
        # a_j a_l a_i^ a_k^
        # double electron amplitudes
        nocc, nvrt = amplitudes.shape[1:3]

        local_amplitudes = np.array(amplitudes).copy()  # a_j a_l a_i^ a_k^

        for i in range(nocc):
            # strict criteria, all pairs (i, j) (k, l) within distance
            local_amplitudes[i, :, max(0, n_neighbors - nocc + i + 1):, :] = 0
            local_amplitudes[:, i, :, max(0, n_neighbors - nocc + i + 1):] = 0

    else:
        raise Exception('Amplitudes format not accepted')

    return local_amplitudes


def get_mod_matrix(matrix, tolerance=2*np.pi, fix_antidiagonals=True):
    n_orb = len(matrix)
    matrix_mod = np.array(matrix)

    if matrix_mod[0, 0] < matrix_mod[n_orb-1, n_orb-1]:
        tolerance *= -1

    matrix_mod[0, 0] = tolerance
    matrix_mod[n_orb-1, n_orb-1] = -tolerance

    if fix_antidiagonals:
        matrix_mod[n_orb-1, 0] = 0.0
        matrix_mod[0, n_orb-1] = 0.0

    return matrix_mod


def get_ucj_ansatz(t2, t1=None, full_trotter=True, tolerance=1e-20, use_qubit=False, n_terms=None, local=False):
    """
    Get unitary coupled Jastrow ansatz from 1-excitation and 2-excitation CC amplitudes

    :param t1: 1-excitation amplitudes in spin-orbital basis ( n x n )
    :param t2: 1-excitation amplitudes in spin-orbital basis ( n x n x n x n )
    :param full_trotter: full trotter
    :param tolerance: amplitude cutoff to include term in the ansatz
    :param use_qubit: return operators as QubitOperators
    :param n_terms: number of terms (U^JU) to be included in the ansatz
    :param local: use local version of J operator
    :return: coefficients as list and ansatz as OperatorsList
    """
    from jastrow.factor import double_factorized_t2
    from jastrow.basis import get_spin_matrix, get_t2_spinorbitals_absolute_full, get_t1_spinorbitals
    from jastrow.rotation import change_of_basis_orbitals

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

    if n_terms is None:
        n_terms = len(diag_coulomb_mats) * 2

    def make_local(mat):
        local_mat = np.array(mat).copy()
        for i, row in enumerate(local_mat):
            row[i+2:] = 0
            row[:max(0, i-1)] = 0
        return local_mat

    i_term = 1
    operator_list = []
    for diag, U in zip(diag_coulomb_mats, orbital_rotations):
        for U_i, diag_i in zip(U, diag):
            if i_term > n_terms:
                break
            i_term += 1

            if local:
                # make local version
                diag_i = np.tril(np.triu(diag_i, -1), 1)

            #diag_i = get_mod_matrix(diag_i, tolerance=7*np.pi/4, fix_antidiagonals=False)

            # build Jastrow operator
            j_mat = np.zeros((norb, norb, norb, norb), dtype=complex)
            for i in range(norb):
                for j in range(norb):
                    j_mat[i, i, j, j] = -1j * diag_i[i, j]  # a_i^ a_j a_k^ a_l

            if full_trotter:

                # jastrow
                spin_jastrow = get_t2_spinorbitals_absolute_full(j_mat)  # a_i^ a_j a_k^ a_l -> a_i^ a_j a_k^ a_l
                coefficients_j, ansatz_j = get_ucc_ansatz(None, spin_jastrow, full_amplitudes=True, use_qubit=use_qubit)
                # coefficients_j, ansatz_j = jastrow_to_qubits(spin_jastrow)

                # basis change
                U_spin = get_spin_matrix(U_i.T)
                coefficients_u, ansatz_u = get_basis_change_exp(U_spin, use_qubit=use_qubit)  # a_i^ a_j

                # add to ansatz
                coefficients += [-c for c in coefficients_u]
                operators += [op for op in ansatz_u]

                coefficients += coefficients_j
                operators += [op for op in ansatz_j]

                coefficients += [c for c in coefficients_u]
                operators += [op for op in ansatz_u]

            else:
                orb_t2 = change_of_basis_orbitals(None, j_mat, U_i.T)[1]  # a_i^ a_j a_k^ a_l
                spin_t2 = get_t2_spinorbitals_absolute_full(orb_t2)
                coefficients_c, ansatz_c = get_ucc_ansatz(None, spin_t2, full_amplitudes=True)

                coefficients += coefficients_c
                operators += [op for op in ansatz_c]

    #from utils import get_operators_order
    #ordering = get_operators_order(operators)
    #print('ordering: ', ordering)
    #exit()
    #operators = np.array(operators)[ordering]
    #coefficients = np.array(coefficients)[ordering]

    ansatz = OperatorList(operators, normalize=False, antisymmetrize=False)

    return coefficients, ansatz


if __name__ == '__main__':

    from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
    from openfermionpyscf import run_pyscf
    from openfermion import MolecularData
    from vqemulti.utils import get_hf_reference_in_fock_space
    from vqemulti.energy import get_vqe_energy, get_adapt_vqe_energy
    from vqemulti.operators import n_particles_operator, spin_z_operator, spin_square_operator
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    from qiskit_aer import AerSimulator

    from vqemulti.preferences import Configuration
    #config = Configuration()
    #config.verbose = 2

    simulator = Simulator(trotter=False,
                          trotter_steps=1,
                          test_only=True,
                          hamiltonian_grouping=True,
                          use_estimator=True, shots=10000,
                          # backend=FakeTorino(),
                          # use_ibm_runtime=True
                          )

    simulator_jastrow = simulator.copy()
    #simulator_jastrow._backend = FakeTorino()
    #simulator_jastrow._use_ibm_runtime = True
    simulator_sqd = simulator.copy()
    simulator_sqd._backend = AerSimulator()
    simulator_sqd._use_ibm_runtime = True


    hydrogen = MolecularData(geometry=[('H', [0.0, 0.0, 0.0]),
                                       ('H', [2.0, 0.0, 0.0]),
                                       ('H', [4.0, 0.0, 0.0]),
                                       ('H', [6.0, 0.0, 0.0])],
                             basis='sto-3g',
                             multiplicity=1,
                             charge=0,
                             description='molecule')

    # run classical calculation
    n_frozen_orb = 0 # nothing
    n_total_orb = 4 # total orbitals
    molecule = run_pyscf(hydrogen, run_fci=False, nat_orb=False, guess_mix=False, verbose=True,
                         frozen_core=n_frozen_orb, n_orbitals=n_total_orb, run_ccsd=True)

    tol_ampl = 0.01

    from pyscf.fci import cistring
    mc = molecule._pyscf_data['casci']

    ncas = mc.ncas
    nelec = mc.nelecas

    # determinants α i β (representats com enters amb bits d’ocupació)
    na, nb = nelec
    alpha_det = cistring.make_strings(range(ncas), na)
    beta_det = cistring.make_strings(range(ncas), nb)


    def interleave_bits(a, b, ncas):
        """Return interleaved occupation string (αβ αβ ...)"""
        a_bits = [(a >> i) & 1 for i in range(ncas)]
        b_bits = [(b >> i) & 1 for i in range(ncas)]
        inter = []
        for i in reversed(range(ncas)):
            inter.append(str(a_bits[i]))
            inter.append(str(b_bits[i]))
        return ''.join(inter)[::-1]

    print('\namplitudes CASCI')
    for i, a in enumerate(alpha_det):
        for j, b in enumerate(beta_det):
            amp = mc.ci[i, j]
            if amp**2 > tol_ampl:
                # print(f"α={format(a, f'0{ncas}b')}  β={format(b, f'0{ncas}b')}  coef={amp:+.6f}")
                cfg = interleave_bits(a, b, ncas)
                print(f"{cfg}   {amp:+.6f}  ({amp**2:.6f}) ")

    hamiltonian = molecule.get_molecular_hamiltonian()
    from openfermion import QubitOperator

    # hamiltonian = QubitOperator('Z0 Z1 Z4 Z5 Z7')
    # hamiltonian = QubitOperator('Z0 Z2 Z3 Z6 Z7')

    # print(hamiltonian)

    n_electrons = molecule.n_electrons - n_frozen_orb * 2
    n_orbitals = n_total_orb - n_frozen_orb  # molecule.n_orbitals
    n_qubits = n_orbitals * 2
    print('n_qubits: ', n_qubits)

    hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, n_qubits)

    hf_energy = get_vqe_energy([], [], hf_reference_fock, hamiltonian, None)
    print('energy HF: ', hf_energy)

    print('\nUCC ansatz\n==========')
    coefficients, ansatz = get_ucc_ansatz(None, molecule.ccsd_double_amps, use_qubit=True)

    # simulator = None
    energy = get_adapt_vqe_energy(coefficients,
                                  ansatz,
                                  hf_reference_fock,
                                  hamiltonian,
                                  simulator)

    simulator.print_statistics()
    print('energy: ', energy)
    print(simulator.get_circuits()[-1])

    n_particle = get_adapt_vqe_energy(coefficients,
                                      ansatz,
                                      hf_reference_fock,
                                      n_particles_operator(n_orbitals),
                                      simulator)

    spin_z = get_adapt_vqe_energy(coefficients,
                                  ansatz,
                                  hf_reference_fock,
                                  spin_z_operator(n_orbitals),
                                  simulator)

    spin_square = get_adapt_vqe_energy(coefficients,
                                       ansatz,
                                       hf_reference_fock,
                                       spin_square_operator(n_orbitals),
                                       simulator)

    print('n_particles: {:8.4f}'.format(n_particle))
    print('Sz: {:8.4f}'.format(spin_z))
    print('S2: {:8.4f}'.format(spin_square))

    print('\nJASTROW ansatz\n==============')

    ccsd = molecule._pyscf_data.get('ccsd', None)
    t2 = crop_local_amplitudes(ccsd.t2, n_neighbors=3)

    coefficients, ansatz = get_ucj_ansatz(t2, full_trotter=False, use_qubit=False, n_terms=1, local=False)

    #simulator_jastrow = None
    energy = get_adapt_vqe_energy(coefficients,
                                  ansatz,
                                  hf_reference_fock,
                                  hamiltonian,
                                  simulator_jastrow)

    simulator_jastrow.print_statistics()
    print(simulator_jastrow.get_circuits()[-1])

    print('Jastrow energy: ', energy)

    n_particle = get_adapt_vqe_energy(coefficients,
                                      ansatz,
                                      hf_reference_fock,
                                      n_particles_operator(n_orbitals),
                                      simulator)

    spin_z = get_adapt_vqe_energy(coefficients,
                                  ansatz,
                                  hf_reference_fock,
                                  spin_z_operator(n_orbitals),
                                  simulator)

    spin_square = get_adapt_vqe_energy(coefficients,
                                       ansatz,
                                       hf_reference_fock,
                                       spin_square_operator(n_orbitals),
                                       simulator)

    print('n_particles: {:8.4f}'.format(n_particle))
    print('Sz: {:8.4f}'.format(spin_z))
    print('S2: {:8.4f}'.format(spin_square))


    def get_projections(n_qubits, n_electrons, tolerance=0.01):

        from vqemulti.operators import configuration_projector_operator
        from vqemulti.utils import configuration_generator

        print('\nAmplitudes square')
        amplitudes_square = []
        for configuration in configuration_generator(n_qubits, n_electrons):
            amplitude2 = get_adapt_vqe_energy(coefficients,
                                              ansatz,
                                              hf_reference_fock,
                                              configuration_projector_operator(configuration),
                                              simulator)

            amplitudes_square.append(amplitude2)

            if abs(amplitude2) > tolerance:
                print('{} {:8.5f} '.format(''.join([str(s) for s in configuration]), amplitude2.real))

        print('sum amplitudes square: ', np.sum(amplitudes_square))
        return amplitudes_square

    get_projections(n_qubits, n_electrons, tolerance=tol_ampl)

    # SQD
    from vqemulti.energy.simulation import simulate_energy_sqd
    energy, samples = simulate_energy_sqd(coefficients,
                                          ansatz,
                                          hf_reference_fock,
                                          hamiltonian,
                                          simulator_sqd,
                                          n_electrons,
                                          adapt=True,
                                          return_samples=True)
    print('SQD energy', energy)
    print(samples)
