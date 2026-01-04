import numpy as np
from openfermion import MolecularData
from openfermionpyscf import run_pyscf, PyscfMolecularData
import unittest


def get_sparse_cc(*params):
    from openfermion import get_sparse_operator
    from vqemulti.ansatz.generators.ucc import get_ucc_generator
    return get_sparse_operator(get_ucc_generator(*params)[0])


def check_similar(array_1, array_2):
    rel_error = np.linalg.norm(array_1 - array_2) / np.linalg.norm(array_1)
    print(f'check: {rel_error:.5e}   max:{np.max(np.linalg.norm(array_1 - array_2)):.5e} index:{np.argmax(np.linalg.norm(array_1 - array_2))}')


def print_max_values_2d(matrix, tol=1e-2, order=None, spin_notation=False):

    if order is not None:
        matrix = matrix.transpose(order)

    ni, nj = matrix.shape

    for i in range(ni):
        for j in range(nj):
            if abs(matrix[i, j]) > tol:
                if spin_notation is False:
                    print('{:3} {:3}  : {:10.5e}'.format(i, j, matrix[i, j]))
                else:
                    spin = lambda i: 'a' if np.mod(i, 2) == 0 else 'b'
                    print('{:3}{:1} {:3}{:1}: {:10.5e}'.format(i // 2, spin(i),
                                                               j // 2, spin(j),
                                                               matrix[i, j]))


def print_max_values_4d(matrix, tol=1e-2, order=None, spin_notation=False):
    # n_values = len(matrix)

    if order is not None:
        matrix = matrix.transpose(order)

    ni, nj, nk, nl = matrix.shape

    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                for l in range(nl):
                    if abs(matrix[i, j, k, l]) > tol:
                        if spin_notation is False:
                            print('{:3}  {:3}  {:3}  {:3}: {:10.5e}'.format(i, j, k, l, matrix[i, j, k, l]))
                        else:
                            spin = lambda i: 'a' if np.mod(i, 2) == 0 else 'b'
                            print('{:3}{:1} {:3}{:1} {:3}{:1} {:3}{:1}: {:10.5e}'.format(i//2, spin(i),
                                                                                         j//2, spin(j),
                                                                                         k//2, spin(k),
                                                                                         l//2, spin(l),
                                                                                         matrix[i, j, k, l]))



class OperationsTest(unittest.TestCase):

    def get_molecule_1(self):
        h4 = MolecularData(geometry=[('H', [0.0, 0.0, 0.0]),
                                     ('H', [2.0, 0.0, 0.0]),
                                     ('H', [4.0, 0.0, 0.0]),
                                     ('H', [6.0, 0.0, 0.0])
                                     ],
                           basis='sto-3g',
                           multiplicity=1,
                           charge=0,
                           description='molecule')

        # run classical calculation
        molecule = run_pyscf(h4, run_fci=False, nat_orb=False, guess_mix=False, verbose=True, run_ccsd=True)
        ccsd = molecule._pyscf_data.get('ccsd', None)
        hamiltonian = molecule.get_molecular_hamiltonian()  # a_i^ a_j / a_i^ a_j a_k^ a_l

        return {
            'name': 'H4/sto-3g',
            'n_qubits' : molecule.n_orbitals * 2,
            'n_orbitals': molecule.n_orbitals,
            'n_occupied': molecule.n_electrons // 2,
            'n_spin_occ': molecule.n_electrons,
            'hamiltonian': hamiltonian,
            'ccsd_amp_single_orb': ccsd.t1,
            'ccsd_amp_double_orb': ccsd.t2,
            'ccsd_amp_single_spin': molecule.ccsd_single_amps,
            'ccsd_amp_double_spin': molecule.ccsd_double_amps}

    def get_molecule_2(self):
        h4 = MolecularData(geometry=[('H', [0.0, 0.0, 0.0]),
                                     ('H', [2.0, 0.0, 0.0]),
                                     ],
                           basis='sto-3g',
                           multiplicity=1,
                           charge=0,
                           description='molecule')

        # run classical calculation
        molecule = run_pyscf(h4, run_fci=False, nat_orb=False, guess_mix=False, verbose=True, run_ccsd=True)
        ccsd = molecule._pyscf_data.get('ccsd', None)
        hamiltonian = molecule.get_molecular_hamiltonian()  # a_i^ a_j / a_i^ a_j a_k^ a_l

        return {
            'name': 'H2/sto-3g',
            'n_qubits' : molecule.n_orbitals * 2,
            'n_orbitals': molecule.n_orbitals,
            'n_occupied': molecule.n_electrons // 2,
            'n_spin_occ': molecule.n_electrons,
            'hamiltonian': hamiltonian,
            'ccsd_amp_single_orb': ccsd.t1,
            'ccsd_amp_double_orb': ccsd.t2,
            'ccsd_amp_single_spin': molecule.ccsd_single_amps,
            'ccsd_amp_double_spin': molecule.ccsd_double_amps}

    def get_molecule_3(self):
        h4 = MolecularData(geometry=[('H', [0.0, 0.0, 0.0]),
                                     ('H', [2.0, 0.0, 0.0]),
                                     ],
                           basis='3-21g',
                           multiplicity=1,
                           charge=0,
                           description='molecule')

        # run classical calculation
        molecule = run_pyscf(h4, run_fci=False, nat_orb=False, guess_mix=False, verbose=True, run_ccsd=True)
        ccsd = molecule._pyscf_data.get('ccsd', None)
        hamiltonian = molecule.get_molecular_hamiltonian()  # a_i^ a_j / a_i^ a_j a_k^ a_l

        return {
            'name': '3-21g',
            'n_qubits' : molecule.n_orbitals * 2,
            'n_orbitals': molecule.n_orbitals,
            'n_occupied': molecule.n_electrons // 2,
            'n_spin_occ': molecule.n_electrons,
            'hamiltonian': hamiltonian,
            'ccsd_amp_single_orb': ccsd.t1,
            'ccsd_amp_double_orb': ccsd.t2,
            'ccsd_amp_single_spin': molecule.ccsd_single_amps,
            'ccsd_amp_double_spin': molecule.ccsd_double_amps}

    def setUp(self):
        self._molecules = [self.get_molecule_1(),
                           self.get_molecule_2(),
                           self.get_molecule_3()]

    def test_orb2spin_conversion(self):
        from vqemulti.ansatz.generators.basis import get_absolute_orbitals
        from vqemulti.ansatz.generators.basis import get_t2_spinorbitals_absolute, get_t1_spinorbitals_absolute, get_t2_spinorbitals_absolute_full
        from vqemulti.ansatz.generators.basis import get_t2_spinorbitals, get_t1_spinorbitals
        for mol in self._molecules:

            print(mol['name'])

            print('n_occ: ', mol['n_occupied'])
            print('n_orb: ', mol['n_orbitals'])

            t1 = mol['ccsd_amp_single_orb'] # a_j  a_i^
            t2 = mol['ccsd_amp_double_orb'] # a_j a_l a_i^ a_k^

            print('T1_orb')
            T1_orb = get_absolute_orbitals(t1)  # a_j a_i^ -> a_j  a_i^
            print_max_values_2d(T1_orb, order=[1, 0])  #  a_i^ a_j

            print('T1_spin')
            T1_spin = get_t1_spinorbitals_absolute(T1_orb, n_occ=mol['n_occupied'])  # a_j a_i^ -> a_i^ a_j
            print_max_values_2d(T1_spin.real, order=[0, 1], spin_notation=True) # a_i^ a_j

            print('T2_orb')
            T2_orb = get_absolute_orbitals(t2)  # a_j a_l a_i^ a_k^ ->  a_j a_l a_i^ a_k^
            print_max_values_4d(T2_orb, order=[2, 0, 3, 1])  # a_i^ a_j a_k^ a_l

            print('T2_spin')
            T2_spin = get_t2_spinorbitals_absolute(T2_orb, n_occ=mol['n_occupied'])  # a_j a_l a_i^ a_k^ -> a_i^ a_j a_k^ a_l
            print_max_values_4d(T2_spin.real, order=[0, 1, 2, 3], spin_notation=True)  # a_i^ a_j a_k^ a_l

            check_similar(mol['ccsd_amp_single_spin'], T1_spin)
            check_similar(mol['ccsd_amp_double_spin'], T2_spin)

            np.testing.assert_almost_equal(mol['ccsd_amp_single_spin'], T1_spin)
            np.testing.assert_almost_equal(mol['ccsd_amp_double_spin'], T2_spin)

            # alternative version
            print('T1_spin')
            T1_spin = get_t1_spinorbitals(t1)  # a_j a_i^ -> a_i^ a_j
            print_max_values_2d(T1_spin.real, spin_notation=True)

            print('T2_spin')
            T2_spin = get_t2_spinorbitals(t2)  # a_j a_l a_i^ a_k^ -> a_i^ a_j a_k^ a_l
            print_max_values_4d(T2_spin.real, spin_notation=True)  # a_i^ a_j a_k^ a_l

            np.testing.assert_almost_equal(mol['ccsd_amp_single_spin'], T1_spin)
            np.testing.assert_almost_equal(mol['ccsd_amp_double_spin'], T2_spin)

            # alternative version 2
            T2_orb_t = T2_orb.transpose(2, 0, 3, 1)  # a_j a_l a_i^ a_k^ -> a_i^ a_j a_k^ a_l
            T2_spin = get_t2_spinorbitals_absolute_full(T2_orb_t)  # a_i^ a_j a_k^ a_l -> a_i^ a_j a_k^ a_l
            np.testing.assert_almost_equal(mol['ccsd_amp_double_spin'], T2_spin)

    def test_orb2spin_rotation(self):
        from vqemulti.ansatz.generators.basis import get_absolute_orbitals
        from vqemulti.ansatz.generators.basis import get_t2_spinorbitals_absolute, get_t2_spinorbitals_absolute_full, get_t1_spinorbitals_absolute_full
        from vqemulti.ansatz.generators.basis import get_t2_spinorbitals, get_t1_spinorbitals, get_t1_spinorbitals_absolute
        from vqemulti.ansatz.generators.rotation import change_of_basis_spin, random_rotation_qr, random_unitary_qr, change_of_basis_orbitals

        for mol in self._molecules:
            print(mol['name'])

            print('n_occ: ', mol['n_occupied'])
            print('n_orb: ', mol['n_orbitals'])

            n_occ = mol['n_occupied']

            t1 = mol['ccsd_amp_single_orb'] * 100  # a_j  a_i^
            t2 = mol['ccsd_amp_double_orb']  # a_j a_l a_i^ a_k^

            # amplitudes orbitals
            print('T1_orb')
            T1_orb = get_absolute_orbitals(t1)  # a_j a_i^ -> a_j a_i^
            print_max_values_2d(T1_orb, order=[1, 0])  #  a_i^ a_j

            print('T2_orb')
            T2_orb = get_absolute_orbitals(t2)  # a_j a_l a_i^ a_k^ ->  a_j a_l a_i^ a_k^
            T2_orb = T2_orb.transpose(2, 0, 3, 1)  # a_j a_l a_i^ a_k^ -> a_i^ a_j a_k^ a_l
            # print_max_values_4d(T2_orb)  # a_i^ a_j a_k^ a_l

            # amplitudes spinorbitals
            print('T1_spin')
            T1_spin = mol['ccsd_amp_single_spin']  #  a_i^ a_j
            print_max_values_2d(T1_spin.real, spin_notation=True)

            print('T2_spin')
            T2_spin = mol['ccsd_amp_double_spin']  #  a_i^ a_j a_k^ a_l
            # print_max_values_4d(T2_spin.real, order=[0, 1, 2, 3], spin_notation=True)  # a_i^ a_j a_k^ a_l

            U_test = random_rotation_qr(mol['n_orbitals'])  # a_i^ a_j
            # U_test = random_unitary_qr(mol['n_orbitals'])  # a_i^ a_j

            # T2 basis change 1
            T2_orb_rot = change_of_basis_orbitals(None, T2_orb, U_test, alternative=False)[1]
            T2_spin_rot_2 = get_t2_spinorbitals_absolute_full(T2_orb_rot)  # a_i^ a_j a_k^ a_l -> a_i^ a_j a_k^ a_l
            T2_spin_rev = change_of_basis_spin(None, T2_spin_rot_2, U_test.T.conj(), alternative=False)[1]

            # T2 basis change 2
            T2_spin = get_t2_spinorbitals(t2)  # a_j a_l a_i^ a_k^ -> a_i^ a_j a_k^ a_l
            T2_spin_rot = change_of_basis_spin(None, T2_spin, U_test, alternative=False)[1]

            np.testing.assert_almost_equal(T2_spin_rot, T2_spin_rot_2)
            np.testing.assert_almost_equal(T2_spin, T2_spin_rev)

            n_occ_spin = n_occ * 2

            # T1 basis change 1
            T1_orb_rot = change_of_basis_orbitals(T1_orb, None, U_test)[0]
            T1_spin_rot_2 = get_t1_spinorbitals_absolute_full(T1_orb_rot)  # a_j a_i^  -> a_i^ a_j
            T1_spin_rev = change_of_basis_spin(T1_spin_rot_2, None, U_test.T.conj())[0]

            # T1 basis change 2
            T1_spin = get_t1_spinorbitals(t1).T  # a_j a_i^ -> a_i^ a_j
            T1_spin_rot = change_of_basis_spin(T1_spin, None, U_test)[0]

            np.testing.assert_almost_equal(T1_spin_rot, T1_spin_rot_2)
            np.testing.assert_almost_equal(T1_spin, T1_spin_rev)

    def _test_basis_change_old(self):
        from vqemulti.ansatz.generators.basis import get_t1_spinorbitals, get_t2_spinorbitals, get_reduced_orbitals, get_t2_spinorbitals_absolute, get_absolute_orbitals, get_t2_spinorbitals_absolute_full
        from vqemulti.ansatz.generators.basis import get_t2_orbitals_absolute, get_t2_orbitals

        for mol in self._molecules:

            print(mol['name'])

            print('n_occ: ', mol['n_occupied'])
            print('n_orb: ', mol['n_orbitals'])

            t1 = mol['ccsd_amp_single_orb'] # a_j  a_i^
            t2 = mol['ccsd_amp_double_orb'] # a_j a_l a_i^ a_k^

            T1_spin = mol['ccsd_amp_single_spin'] # a_j  a_i^
            T2_spin = mol['ccsd_amp_double_spin'] # a_j a_l a_i^ a_k^

            n_total_orb = mol['n_orbitals']

            print('----')


            T1_spin = get_t1_spinorbitals(t1) # a_j a_i^ -> a_i^ a_j

            print('T1_spin_abs')
            print(T1_spin.shape)
            print_max_values_2d(T1_spin)

            T2_spin = get_t2_spinorbitals(t2)

            print('T2_spin_abs')
            print(T2_spin.shape)

            ccsd_t2_rec = get_t2_orbitals(T2_spin, n_occ= mol['n_occupied'])

            print('T2_orb')
            print(ccsd_t2_rec.shape)
            print(ccsd_t2_rec)

            check_similar(ccsd_t2_rec, t2)
            np.testing.assert_almost_equal(ccsd_t2_rec, t2)


            # exit()

            print('\n=======================\n')

            # absolute
            n_orbital = mol['n_orbitals']
            n_virt = n_orbital - mol['n_occupied']

            print('T2_spin_abs')
            print(T2_spin.shape)
            print_max_values_4d(T2_spin)

            ccsd_t2_abs = get_t2_orbitals_absolute(T2_spin, n_occ=mol['n_occupied'])

            print('T2_orb_abs')
            print(ccsd_t2_abs.shape)
            print_max_values_4d(ccsd_t2_abs)
            # exit()

            print('T2_orb')
            # ccsd_t2_trans = ccsd_t2_abs.transpose(1, 3, 0, 2)[:n_occ, :n_occ, n_occ:, n_occ:]
            ccsd_t2_trans = get_reduced_orbitals(ccsd_t2_abs, n_occ=mol['n_occupied'])
            print(ccsd_t2_trans.shape, t2.shape)
            print(ccsd_t2_trans)
            check_similar(ccsd_t2_trans, t2)

            T2_spin_abs = get_t2_spinorbitals_absolute(ccsd_t2_abs, n_occ=mol['n_occupied'])

            print('T2_spin_abs')
            print(T2_spin_abs.shape)
            print_max_values_4d(T2_spin_abs)

            check_similar(T2_spin_abs, T2_spin)

            # exit()

            # random unitary basis
            print('Test basis orbital and spinorbital')
            from vqemulti.ansatz.generators.rotation import random_rotation_qr
            U_test = random_rotation_qr(n_total_orb)

            print('U shape: ', U_test.shape)
            print('unitary: ', np.allclose(U_test.conj().T @ U_test, np.eye(U_test.shape[0])))

            # basis in spinorbitals
            U_test_sp = np.zeros((n_total_orb * 2, n_total_orb * 2), dtype=complex)
            for i in range(n_total_orb):
                for j in range(n_total_orb):
                    U_test_sp[2 * i, 2 * j] = U_test[i, j]
                    U_test_sp[2 * i + 1, 2 * j + 1] = U_test[i, j]

            # change of basis in orbitals
            T2_orb = get_absolute_orbitals(t2)
            T2_orb = T2_orb.transpose(3, 1, 2, 0)
            # print_max_values_4d(T2_orb)

            T2_orb_bc = np.einsum('ai, bj, abcd, ck, dl -> ijkl', U_test, U_test.conj(), T2_orb, U_test.conj(), U_test)
            print('----')
            # print_max_values_4d(T2_orb_bc)
            # print(np.round(T2_orb_bc, decimals=3)[:2, :2, :2, :2])

            print('full')
            print_max_values_4d(T2_orb)
            print('====')

            # T2_spin_bc_1 = get_t2_spinorbitals_absolute(T2_orb_bc, n_occ=n_occ)
            T2_spin_bc_1 = get_t2_spinorbitals_absolute_full(T2_orb_bc)
            # print_max_values_4d(T2_spin_bc_1)

            print(np.round(T2_spin_bc_1.real * 2, decimals=3)[:4, :4, :4, :4])

            print('----')

            # change of basis in spinorbitals
            T2_spin = get_t2_spinorbitals(t2)
            # print_max_values_4d(T2_spin.real*2)
            T2_spin_bc_2 = np.einsum('ai, bj, abcd, ck, dl -> ijkl', U_test_sp, U_test_sp.conj(), T2_spin, U_test_sp.conj(),
                                     U_test_sp)

            print(np.round(T2_spin_bc_2.real * 2, decimals=3)[:4, :4, :4, :4])

            # print_max_values_4d(T2_spin_bc_2.real*2)

            check_similar(T2_spin_bc_1, T2_spin_bc_2)

    def test_basis_change(self):
        from vqemulti.utils import get_sparse_ket_from_fock, get_hf_reference_in_fock_space
        from vqemulti.ansatz.generators.rotation import change_of_basis_spin, random_unitary_qr
        from openfermion import get_sparse_operator
        from vqemulti.ansatz.generators.basis import get_sparse_basis_change_exp, get_spin_matrix
        from numpy.testing import assert_almost_equal

        import scipy as sp

        for mol in self._molecules:

            print(mol['name'])

            print('n_occ: ', mol['n_occupied'])
            print('n_orb: ', mol['n_orbitals'])

            n_electrons = mol['n_spin_occ']
            n_orbitals = mol['n_orbitals']
            n_qubits = mol['n_qubits']

            ccsd_single_amps = mol['ccsd_amp_single_spin'] * 0  # a_i^ a_j
            ccsd_double_amps = mol['ccsd_amp_double_spin']      # a_i^ a_j a_k^ a_l

            hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, n_qubits)

            sparse_ham = get_sparse_operator(mol['hamiltonian'])  # a_i^ a_j / a_i^ a_j a_k^ a_l
            sparse_reference = get_sparse_ket_from_fock(hf_reference_fock)

            print('DIM hamiltonian:', sparse_ham.shape)
            print('DIM reference:', sparse_reference.shape)

            print('N electrons: ', n_electrons)
            print('N orbitals: ', n_orbitals)
            print('N spin-orbitals: ', n_qubits)

            print('HF energy (sparse):')
            print(sparse_reference.getH() @ sparse_ham @ sparse_reference)

            print('UCC energy (sparse):')
            sparse_cc_op = get_sparse_cc(ccsd_single_amps, ccsd_double_amps)  # a_i^ a_j / a_i^ a_j a_k^ a_l
            sparse_cc_state = sp.sparse.linalg.expm_multiply(sparse_cc_op, sparse_reference)
            energy_ucc = sparse_cc_state.T @ sparse_ham @ sparse_cc_state
            assert_almost_equal((sparse_cc_state.getH() @ sparse_cc_state)[0, 0], 1)
            print(energy_ucc)
            print('DIM UCC:', sparse_cc_op.shape)

            # basis change
            print('\n--------------\n' + 'BASIS CHANGE')

            # U_test = random_rotation_qr(n_orbitals)  # a_i^ a_j
            U_test = random_unitary_qr(n_orbitals)  # a_i^ a_j

            # theta = 0.2
            # U_test = np.array([[np.cos(theta), np.sin(theta)*np.exp(1j*theta)],
            #                    [-np.sin(theta)*np.exp(-1j*theta), np.cos(theta)]])


            print('U shape: ', U_test.shape)
            print('unitary: ', np.allclose(U_test.conj().T @ U_test, np.eye(U_test.shape[0])))
            print('Rotation: ', np.linalg.det(U_test) > 0)
            assert_almost_equal(np.linalg.det(U_test), 1)

            print('U_test')
            print(U_test)  # a_i^ a_j

            # print('ccsd_double_amps')
            # print_max_values_4d(ccsd_double_amps*2, spin_notation=True)

            # simulate basis change of exponent
            single_amps_bc, double_amps_bc = change_of_basis_spin(ccsd_single_amps, ccsd_double_amps, U_test)  # a_i^ a_j / a_i^ a_j a_k^ a_l
            sparse_cc_op_bc = get_sparse_cc(single_amps_bc, double_amps_bc)  # a_i^ a_j / a_i^ a_j a_k^ a_l

            print('UCC energy (basis change):')
            sparse_cc_state = sp.sparse.linalg.expm_multiply(sparse_cc_op_bc, sparse_reference)
            ucc_energy_bc = sparse_cc_state.getH() @ sparse_ham @ sparse_cc_state
            print(ucc_energy_bc)

            assert_almost_equal((sparse_cc_state.getH() @ sparse_cc_state)[0, 0], 1)

            # generate operator that reproduces U_test rotation on CC amplitudes
            U_test_sp = get_spin_matrix(U_test)  # a_i^ a_j
            sparse_K_sp = get_sparse_basis_change_exp(U_test_sp)
            sparse_U_sp = sp.sparse.linalg.expm(-sparse_K_sp)

            # check consistency of unitary transformations
            sparse_U_array = sparse_U_sp.toarray()
            sparse_U_array_inv = sp.sparse.linalg.expm(sparse_K_sp).toarray()
            print('sparse U shape: ', sparse_U_array.shape)
            print('sparse unitary: ', np.allclose(sparse_U_array.conj().T @ sparse_U_array, np.eye(sparse_U_array.shape[0])))
            print('sparse consisten: ', np.allclose(sparse_U_array_inv, sparse_U_array.conj().T))
            print('sparse Rotation: ', np.linalg.det(sparse_U_array) > 0)
            assert_almost_equal(np.linalg.det(sparse_U_array), 1)
            assert_almost_equal(sparse_U_array_inv, sparse_U_array.conj().T)

            # simulate basis change of exponent
            sparse_cc_op = get_sparse_cc(ccsd_single_amps, ccsd_double_amps)  # a_i^ a_j / a_i^ a_j a_k^ a_l
            sparse_cc_op_bc_2 = sparse_U_sp.getH() @ sparse_cc_op @ sparse_U_sp

            print('UCC energy (reproduce change with sparse_U):')
            sparse_cc_state = sp.sparse.linalg.expm_multiply(sparse_cc_op_bc_2, sparse_reference)
            ucc_energy_bc_2 = sparse_cc_state.getH() @ sparse_ham @ sparse_cc_state
            print(ucc_energy_bc_2)

            assert_almost_equal((sparse_cc_state.getH() @ sparse_cc_state)[0, 0], 1)
            self.assertAlmostEqual(ucc_energy_bc[0, 0], ucc_energy_bc_2[0, 0])

            # switch betwen exact and bc
            # sparse_cc_op_bc = sparse_cc_op_bc_2

            print('UCC energy (basis change inverse):')

            sparse_cc_exp_U = sparse_U_sp @ sp.sparse.linalg.expm(sparse_cc_op_bc) @ sparse_U_sp.getH()
            sparse_cc_state = sparse_cc_exp_U @ sparse_reference
            ucc_energy_recover = sparse_cc_state.getH() @ sparse_ham @ sparse_cc_state
            print(ucc_energy_recover)

            assert_almost_equal((sparse_cc_state.getH() @ sparse_cc_state)[0, 0], 1)
            self.assertAlmostEqual(ucc_energy_recover[0, 0], energy_ucc[0, 0])

    def test_rotation_op(self):
        from vqemulti.utils import get_sparse_ket_from_fock, get_hf_reference_in_fock_space
        from vqemulti.ansatz.generators.rotation import change_of_basis_spin, random_rotation_qr, random_unitary_qr
        from openfermion import get_sparse_operator
        from vqemulti.ansatz.generators.basis import get_sparse_basis_change_exp, get_spin_matrix
        from numpy.testing import assert_almost_equal

        import scipy as sp

        # get occupation vector
        def get_occupations(sparse_rot_state, n_qubits):
            from openfermion import FermionOperator

            occ_vect = []
            for i in range(n_qubits):
                op = get_sparse_operator(FermionOperator(f"{i}^ {i}"), n_qubits=n_qubits)
                occ_vect.append(float((sparse_rot_state.getH() @ op @ sparse_rot_state)[0, 0]))
            return np.round(occ_vect, decimals=4)

        # get projections
        def get_projections(sparse_state, n_qubits, n_electrons, tolerance=0.01):

            def generate_configurations(n, k):
                """
                generate configuration vectors of length n and k occupations

                :param n: number of sites
                :param k: number of occupied sites
                :return: vector
                """
                import itertools
                for positions in itertools.combinations(range(n), k):
                    vector = [0] * n
                    for pos in positions:
                        vector[pos] = 1
                    yield vector

            from openfermion import FermionOperator

            amplitudes = []
            for configuration in generate_configurations(n_qubits, n_electrons):

                sparse_reference = get_sparse_ket_from_fock(configuration)
                amplitude = float((sparse_state.getH() @ sparse_reference)[0, 0])
                amplitudes.append(amplitude.real)

                if np.linalg.norm(amplitude) > tolerance:
                    print(' {:8.5f} {} '.format(amplitude.real, ''.join([str(s) for s in configuration])))


            return amplitudes

        for mol in self._molecules:

            print(mol['name'])

            print('n_occ: ', mol['n_occupied'])
            print('n_orb: ', mol['n_orbitals'])

            n_electrons = mol['n_spin_occ']
            n_orbitals = mol['n_orbitals']
            n_qubits = mol['n_qubits']

            hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, n_qubits)

            sparse_ham = get_sparse_operator(mol['hamiltonian'])  # a_i^ a_j / a_i^ a_j a_k^ a_l
            sparse_reference = get_sparse_ket_from_fock(hf_reference_fock)
            occupations = get_occupations(sparse_reference, n_qubits)
            energy_ref = sparse_reference.getH() @ sparse_ham @ sparse_reference

            print('DIM hamiltonian:', sparse_ham.shape)
            print('DIM reference:', sparse_reference.shape)

            print('N electrons: ', n_electrons)
            print('N orbitals: ', n_orbitals)
            print('N spin-orbitals: ', n_qubits)

            print('HF energy (sparse):')
            print(energy_ref)

            print('occupations:', occupations)
            print('N electrons: ', sum(occupations))

            print('amplitudes')
            amplitudes_ref = get_projections(sparse_reference, n_qubits, n_electrons)

            # rotation
            print('\n=================================\n')

            if True:
                theta = np.pi/2
                R = np.array([[np.cos(theta), np.sin(theta)],
                              [-np.sin(theta), np.cos(theta)]])

                U_test = np.identity(n_orbitals)        # a_i^ a_j
                U_test[mol['n_occupied']-1: mol['n_occupied']+1, mol['n_occupied']-1: mol['n_occupied']+1] = R

            U_test = random_unitary_qr(n_orbitals)  # a_i^ a_j

            print('U shape: ', U_test.shape)
            print('unitary: ', np.allclose(U_test.conj().T @ U_test, np.eye(U_test.shape[0])))
            print('Rotation: ', np.linalg.det(U_test) > 0)
            assert_almost_equal(np.linalg.det(U_test), 1)

            print('U_test')
            print(U_test)  # a_i^ a_j

            print('ccsd_double_amps')
            # print_max_values_4d(ccsd_double_amps*2, spin_notation=True)

            # generate operator that reproduces U_test rotation on CC amplitudes
            U_test_sp = get_spin_matrix(U_test)  # a_i^ a_j
            sparse_K_sp = get_sparse_basis_change_exp(U_test_sp)

            # rotation straight
            print('transformed energy:')
            sparse_rot_state = sp.sparse.linalg.expm_multiply(-sparse_K_sp, sparse_reference)
            energy_rot = sparse_rot_state.getH() @ sparse_ham @ sparse_rot_state
            assert_almost_equal((sparse_rot_state.getH() @ sparse_rot_state)[0, 0], 1)
            print(energy_rot)

            occupations = get_occupations(sparse_rot_state, n_qubits)
            print('occupations:', occupations)
            print('N electrons: ', sum(occupations))

            print('amplitudes')
            amplitudes_rot = get_projections(sparse_rot_state, n_qubits, n_electrons)

            # rotation inversion (recover)
            print('recover energy:')
            sparse_rec_state = sp.sparse.linalg.expm_multiply(sparse_K_sp, sparse_rot_state)
            energy_rec = sparse_rec_state.getH() @ sparse_ham @ sparse_rec_state
            assert_almost_equal((sparse_rec_state.getH() @ sparse_rec_state)[0, 0], 1)
            print(energy_rec)

            occupations = get_occupations(sparse_rec_state, n_qubits)
            print('occupations:', occupations)
            print('N electrons: ', sum(occupations))

            print('amplitudes')
            amplitudes_rec = get_projections(sparse_rec_state, n_qubits, n_electrons)

            assert_almost_equal(energy_rec[0, 0], energy_ref[0, 0], err_msg='in recovered energies')
            assert_almost_equal(amplitudes_rec, amplitudes_ref, err_msg='in recovered amplitudes')


            # simple rotation
            sparse_cc_op_bc = get_sparse_cc(U_test_sp, None)  # a_i^ a_j / a_i^ a_j a_k^ a_l
            sparse_t1_state = sp.sparse.linalg.expm_multiply(sparse_cc_op_bc, sparse_reference)
            energy_t1 = sparse_t1_state.getH() @ sparse_ham @ sparse_t1_state
            print('final: ', energy_t1)

