from openfermionpyscf import run_pyscf
from openfermion import MolecularData
from openfermion import commutator
from openfermion import FermionOperator, get_fermion_operator
from openfermion import get_sparse_operator
from vqemulti.utils import get_truncated_fermion_operators
import numpy as np


def _separate_hamiltonian(hamiltonian):

    if not isinstance(hamiltonian, FermionOperator):
        hamiltonian = get_fermion_operator(hamiltonian)

    Hf = FermionOperator()
    Hv = FermionOperator()

    for terms in hamiltonian.terms.items():
        if len(terms[0]) == 2:
            Hf += FermionOperator(terms[0]) * terms[1]
        elif len(terms[0]) == 4:
            Hv += FermionOperator(terms[0]) * terms[1]
    return hamiltonian, Hf, Hv


def _get_ucc_interaction(t1, t2, core_orbitals, tolerance=1e-6, full_amplitudes=False):
    """
    Get subspace-environment interaction CC generator

    :param t1: 1-excitation amplitudes in spin-orbital basis ( n x n )
    :param t2: 1-excitation amplitudes in spin-orbital basis ( n x n x n x n )
    :param tolerance: amplitude cutoff to include term in the ansatz
    :return:
    """
    from openfermion import FermionOperator, hermitian_conjugated, normal_ordered

    core_qubits = core_orbitals * 2
    ext_op = FermionOperator()

    # 1-exciation terms
    if t1 is not None:
        n_spin_orbitals = len(t1)
        for i in range(n_spin_orbitals):
            for j in range(n_spin_orbitals):
                if (i >= core_qubits and j < core_qubits) or (i < core_qubits and j >= core_qubits):

                    if abs(t1[i,j]) > tolerance:
                        operator = FermionOperator('{}^ {}'.format(i, j))
                        if not full_amplitudes:
                            ext_op += t1[i, j] * operator - hermitian_conjugated(t1[i, j] * operator)
                        else:
                            ext_op += t1[i, j] * operator

    # 2-excitation terms
    if t2 is not None:
        n_spin_orbitals = len(t2)
        for i in range(n_spin_orbitals):
            for j in range(n_spin_orbitals):
                for k in range(n_spin_orbitals):
                    for l in range(n_spin_orbitals):  # avoid duplicates
                        if np.mod(i, 2) + np.mod(k, 2) == np.mod(j, 2) + np.mod(l, 2):  # keep spin symmetry

                            if (i >= core_qubits and j < core_qubits) or (
                                    i < core_qubits and j >= core_qubits) or (
                                    k >= core_qubits and l < core_qubits) or (
                                    k < core_qubits and l >= core_qubits):

                                if abs(t2[i, j, k, l]) > tolerance:
                                    operator = FermionOperator('{}^ {} {}^ {}'.format(i, j, k, l))
                                    if not full_amplitudes:
                                        ext_op += 0.5 * t2[i, j, k, l] * operator - hermitian_conjugated(0.5 * t2[i, j, k, l] * operator)

                                    else:
                                        ext_op += 0.5 * t2[i, j, k, l] * operator

    return ext_op


def get_ducc_hamiltonian(hamiltonian, T1, T2, max_orbitals, max_order=None):

    def commutator_trunc(operator_a, operator_b):
        operator_a = get_truncated_fermion_operators(operator_a, max_orbitals, max_order)
        operator_b = get_truncated_fermion_operators(operator_b, max_orbitals, max_order)

        operator_comm = commutator(operator_a, operator_b)

        return get_truncated_fermion_operators(operator_comm, max_orbitals, max_order)

    s_ext = _get_ucc_interaction(T1, T2, max_orbitals)

    Hn = hamiltonian - molecule.hf_energy

    Hn, Fn, Hv = _separate_hamiltonian(Hn)

    # A4
    Hext = Hn + commutator_trunc(Hn, s_ext) + 0.5 * (commutator_trunc(commutator_trunc(Fn, s_ext), s_ext)) + molecule.hf_energy

    # A7
    #Hext = Hn + commutator_trunc(Hn, s_ext) + 0.5 * (commutator_trunc(commutator_trunc(Hn, s_ext), s_ext)) + 1/6. * (commutator_trunc(commutator_trunc(commutator_trunc(Fn, s_ext), s_ext), s_ext)) + molecule.hf_energy

    # final truncation
    Hext = get_truncated_fermion_operators(Hext, max_orbitals, max_mb=max_order)

    return Hext


def get_hamiltonian(hamiltonian, T1, T2):

    import numpy as np
    import openfermion.chem.molecular_data
    from pyscf import ao2mo
    from pyscf.tools import fcidump


    # Load fcidump
    fci_dict = fcidump.read('fcidump_h3.txt')
    onebody_int_mo = fci_dict['H1']
    twobody_int_mo = fci_dict['H2']
    core_energy = fci_dict['ECORE']

    # Convert to full 4-index array
    twobody_int_full_mo = ao2mo.restore(1, twobody_int_mo, onebody_int_mo.shape[0])
    print("Two-body integrals restored from PySCF in MO basis:")
    #print(twobody_int_full_mo)
    print('-----------------------------')

    # The key correction: Reorder indices to match OpenFermion's expectation
    # PySCF gives (pq|rs) but OpenFermion's spinorb_from_spatial expects proper chemist ordering
    twobody_int_ordered = twobody_int_full_mo.transpose(0, 2, 3, 1)


    print("Two-body integrals fed to the spin basis change:")
    #print(twobody_int_ordered)
    print('-----------------------------')


    # Use spinorb_from_spatial
    onebody_coeffs, twobody_coeffs = openfermion.chem.molecular_data.spinorb_from_spatial(
        onebody_int_mo, twobody_int_ordered
    )

    # Apply the 1/2 factor
    twobody_coeffs = twobody_coeffs / 2

    print("Two-body coefficients found after changing the basis to spin orbitals and dividing by 2 "
          "this is fed to the InterationOperator():")
    #print(twobody_coeffs)
    print('-----------------------------')

    op_ham_secq = openfermion.ops.InteractionOperator(core_energy, onebody_coeffs, twobody_coeffs)


    #print("\nFinal Hamiltonian:")
    #print(op_ham_secq)
    #print('-----------------------------')

    from openfermion import get_fermion_operator, get_sparse_operator
    H_fermion = get_fermion_operator(op_ham_secq)
    print("\nFermion operator:")
    print(H_fermion)
    print('-----------------------------')
    H_sparse = get_sparse_operator(H_fermion).toarray()

    diag = np.linalg.eig(H_sparse)
    print('The diagonal hamiltonian is:')
    print(diag[0])
    print('With the lowest eigenvalue being the ground state energy:', min(diag[0]).real)






if __name__ == '__main__':

    dist = 3.0
    molecule = MolecularData(geometry=[('H', [0.0, 0.0, 0.0]),
                                       ('H', [dist*2, 0.0, 0.0]),
                                       ('H', [dist*3, 0.0, 0.0]),
                                       ('H', [dist*4, 0.0, 0.0]),
                                       ],
                             basis='sto-3g',
                             multiplicity=1,
                             charge=0,
                             description='H4')

    molecule = run_pyscf(molecule, run_fci=True, verbose=True, frozen_core=0, n_orbitals=4, run_ccsd=True)

    hamiltonian = molecule.get_molecular_hamiltonian()
    print('n_orbitals: ', molecule.n_orbitals)
    print('HF energy: ', molecule.hf_energy)
    print('FCI energy: ', molecule.fci_energy)

    T2 = molecule.ccsd_double_amps
    T1 = molecule.ccsd_single_amps

    # get GS of DUCC hamiltonian (truncated to 3 orbitals)
    hamiltonian_ducc = get_ducc_hamiltonian(hamiltonian, T1, T2, max_orbitals=3, max_order=4)
    sparse_ham = get_sparse_operator(hamiltonian_ducc).toarray()
    print('lowest eigenvalues (DUCC):', np.sort(np.linalg.eigvals(sparse_ham).real)[0])

    # get GS of original hamiltonian
    sparse_ham = get_sparse_operator(hamiltonian).toarray()
    print('lowest eigenvalues (Real):', np.sort(np.linalg.eigvals(sparse_ham).real)[0])

    # molecule, n_frozen_orb, n_total_orb = get_LiH()
    molecule = MolecularData(geometry=[('H', [0.0, 0.0, 0.0]),
                                       ('H', [dist*2, 0.0, 0.0]),
                                       ('H', [dist*3, 0.0, 0.0]),
                                       ('H', [dist*4, 0.0, 0.0]),
                                       ],
                             basis='sto-3g',
                             multiplicity=1,
                             charge=0,
                             description='H4')

    # run CASCI truncated calculation
    molecule = run_pyscf(molecule, run_fci=False, verbose=True, frozen_core=0, n_orbitals=3)
    print('CASCI energy (truncated {} orb): {}'.format(3, molecule.casci_energy))
