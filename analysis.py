import numpy as np
from openfermion.transforms.opconversions import get_fermion_operator
from openfermion.linalg import jw_hartree_fock_state, jordan_wigner_sparse


def compute_JK(density_matrix, Vee):
    density_matrix = np.array(density_matrix)
    n_basis_functions = len(density_matrix)
    J = np.zeros((n_basis_functions, n_basis_functions))
    K = np.zeros((n_basis_functions, n_basis_functions))

    def J_element(i, j):
        J_sum = 0.0
        K_sum = 0.0

        for k in range(n_basis_functions):
            for l in range(n_basis_functions):
                density = density_matrix[k, l]
                J = Vee[i, j, k, l]
                K = Vee[i, l, k, j]
                J_sum += density * J
                K_sum += density * K

        return J_sum, K_sum  # handle double counting

    for i in range(n_basis_functions):
        for j in range(n_basis_functions):
            J[i, j], K[i, j] = J_element(i, j)

    return J, K


def get_info(molecule, check_HF_data=False):
    # check calculation properties

    print('Molecule description {}\n'.format(molecule.description))
    print('HF energy', molecule.hf_energy)

    electronNumber = molecule.n_electrons
    orbitalNumber = molecule.n_orbitals
    qubitNumber = molecule.n_qubits

    print('orbitalNumber: ', orbitalNumber)
    print('qubitNumber: ', qubitNumber)
    print('electronNumber: ', electronNumber)

    # binary numbers encode fock space
    jw_hf_state = jw_hartree_fock_state(electronNumber, qubitNumber)
    print("JW HF reference")
    print(jw_hf_state)

    # get fermionic hamiltonian
    hamiltonian = get_fermion_operator(molecule.get_molecular_hamiltonian())
    print('Fermionic Hamiltonian:\n', hamiltonian)

    sparse_hamiltonian = jordan_wigner_sparse(hamiltonian)
    print('JW Sparse Hamiltonian:\n', sparse_hamiltonian)

    if check_HF_data:

        print('Mono electronic integrals (MO)')
        print(h2molecule.one_body_integrals)

        print('Bi electronic integrals (MO)')
        print(h2molecule.two_body_integrals)

        s = h2molecule.overlap_integrals
        coeff = h2molecule.canonical_orbitals

        density_matrix = 2*np.outer(coeff.T[0], coeff.T[0])
        dm_mo = coeff.T @ s @ density_matrix @ s @ coeff

        print('density (MO)')
        print(dm_mo)
        J, K = compute_JK(dm_mo, h2molecule.two_body_integrals)

        print('J (MO)')
        print(J)
        print('K (MO)')
        print(K)

        coulomb_energy = np.sum(dm_mo * J/2)
        exchange_energy = -0.5 * np.sum(dm_mo * K/2)
        print('Computed Properties\n')
        print('1e energy:', np.sum(dm_mo * h2molecule.one_body_integrals))
        print('Total Coulomb:', coulomb_energy)
        print('HF Exchange:', exchange_energy)

        print('======================')


if __name__ == '__main__':
    # molecule definition
    from openfermion import MolecularData
    from openfermionpyscf import run_pyscf

    geometry = [['H',[0, 0, 0]],
                ['H',[0, 0, 0.74]]]

    basis = 'sto-3g'
    multiplicity = 1
    charge = 0
    h2molecule = MolecularData(geometry, basis, multiplicity, charge, description='H2')


    # run classical calculation
    h2molecule = run_pyscf(h2molecule, run_fci=True, run_ccsd=True)
    print(h2molecule.orbital_energies)

    print('HF energy', h2molecule.hf_energy)
