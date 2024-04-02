from posym import SymmetryMolecule, SymmetryGaussianLinear
from posym.tools import get_basis_set_pyscf, build_density, build_orbital
from scipy.linalg import expm
from scipy.optimize import minimize
from itertools import combinations
import numpy as np


def cost_fuction(values):
    return np.sum([np.prod(pair) for pair in combinations(values, 2)])


def generate_unitary(n, params):
    anti_symmetric = np.zeros((n, n))
    k = 0
    for i in range(0, n):
        for j in range(i+1, n):
            anti_symmetric[i, j] = params[k]
            anti_symmetric[j, i] = -params[k]
            k += 1

    return expm(anti_symmetric)


def symmetrize_orbitals(mo_coefficients, molecule, group, skip_first=0, skip=False):

    # get geometric data
    symbols = [atom[0] for atom in molecule.geometry]
    coordinates = molecule._pyscf_data['mol'].atom_coords()

    geom_sym = SymmetryMolecule(group, coordinates, symbols)
    print('geom_sym: ', geom_sym)

    # get electronic data
    basis_set = get_basis_set_pyscf(molecule._pyscf_data['mol'])

    print('\nMO symmetry (initial)')
    for i, orbital_vect in enumerate(mo_coefficients.T):
        orb = build_orbital(basis_set, orbital_vect)

        sym_orb = SymmetryGaussianLinear(group, orb,
                                         orientation_angles=geom_sym.orientation_angles,
                                         center=geom_sym.center
                                         )
        values = sym_orb.get_ir_representation().values
        cost = cost_fuction(values)

        print('orbital {:3}: {:6.2f} ({})'.format(i, cost, sym_orb))

    if skip:
        return mo_coefficients

    def minimize_function(a, restrict, group):

        n_dim = len(restrict.T)
        u_matrix = generate_unitary(n_dim, a)

        restrict = restrict @ u_matrix

        total_sum = 0
        for i, orbital_vect in enumerate(restrict.T[:1]):
            orb = build_orbital(basis_set, orbital_vect)

            sym_orb = SymmetryGaussianLinear(group, orb,
                                             orientation_angles=geom_sym.orientation_angles,
                                             center=geom_sym.center
                                             )

            values = sym_orb.get_ir_representation().values
            total_sum += cost_fuction(values)

        return total_sum

    n_orbitals = len(mo_coefficients.T)
    for n_orb_chunk in range(skip_first, n_orbitals-1):
        print('Symmetry iteration: ', n_orb_chunk)
        restrict = mo_coefficients[:, n_orb_chunk:]

        n_dim = len(restrict.T)
        n_params = (n_dim ** 2 - n_dim) // 2
        params = np.zeros(n_params)

        res = minimize(minimize_function, params, args=(restrict, group))

        u_matrix = generate_unitary(n_dim, res.x)

        restrict = restrict @ u_matrix
        mo_coefficients[:, n_orb_chunk:] = restrict[:, :]

    # final check
    print('\nMO symmetry (final)')
    for i, orbital_vect in enumerate(mo_coefficients.T):
        orb = build_orbital(basis_set, orbital_vect)

        sym_orb = SymmetryGaussianLinear(group, orb,
                                         orientation_angles=geom_sym.orientation_angles,
                                         center=geom_sym.center
                                         )

        values = sym_orb.get_ir_representation().values
        cost = np.sum([np.prod(pair) for pair in combinations(values, 2)])

        print('orbital {:3}: {:6.2f} ({})'.format(i, cost, sym_orb))
        # print('values: ', values)

    return mo_coefficients


if __name__ == '__main__':

    from openfermionpyscf import prepare_pyscf_molecule
    from openfermion import MolecularData
    from openfermionpyscf import run_pyscf
    import matplotlib.pyplot as plt

    def square_h4_mol(distance, basis='sto-3g'):
        mol = MolecularData(geometry=[['H', [0, 0, 0]],
                                      ['H', [distance, 0, 0]],
                                      ['H', [0, distance, 0]],
                                      ['H', [distance, distance, 0]]],
                            basis=basis,
                            multiplicity=1,
                            charge=0,
                            description='H4')
        return mol

    pyscf_molecule = square_h4_mol(distance=1.0, basis='3-21g')

    # run classical calculation
    molecule = run_pyscf(pyscf_molecule, run_fci=True, run_ccsd=True, nat_orb=True, guess_mix=False)

    # get FullCI density matrix in natural orbitals basis (for fidelity calculation)
    mo_coefficients = molecule.canonical_orbitals
    mo_energies = molecule.orbital_energies

    symbols = [atom[0] for atom in molecule.geometry]
    coordinates = molecule._pyscf_data['mol'].atom_coords()

    mo_coefficients = symmetrize_orbitals(mo_coefficients, coordinates, symbols, 'd4h', skip=False)

    mol = prepare_pyscf_molecule(pyscf_molecule)
    n_electrons = pyscf_molecule.n_electrons
    n_orbitals = pyscf_molecule.n_orbitals

    fullci_list = []
    casci_list = []
    cisd_list = []
    range_orbitals = list(range(n_electrons//2+1, n_orbitals+1))
    for i_orb in range_orbitals:

        print('\n========', i_orb, '========')
        print('n_elect', n_electrons)
        print('n_orbitals', n_orbitals)
        print('included orb', i_orb)

        rhf_calc = mol.RHF() #.run()

        # substitute orbitals
        rhf_calc.mo_coeff = mo_coefficients
        rhf_calc.mo_energy = mo_energies
        rhf_calc.mo_occ = np.array([2.] * (n_electrons//2) + [0.] * (n_orbitals - n_electrons//2))
        rhf_calc.converged = True

        # cas calculation
        cas_calc = rhf_calc.CASCI(i_orb, n_electrons)
        cas_calc.kernel()
        print('CAS: ', i_orb, n_electrons)
        print('Energy CASCI: {:.8f}'.format(cas_calc.e_tot))
        print('Error CASCI: {:.8f}'.format(cas_calc.e_tot - molecule.fci_energy))
        casci_list.append(cas_calc.e_tot - molecule.fci_energy)

        # CISD calculation
        list_orb = list(range(i_orb, n_orbitals))
        print('freeze: ', list_orb)
        if len(list_orb) == 0:
            list_orb = None

        cis_calc = rhf_calc.CISD(frozen=list_orb)
        cis_calc.kernel()
        print('Energy CISD: {:.8f}'.format(cis_calc.e_tot))
        print('Error CISD: {:.8f}'.format(cis_calc.e_tot - molecule.fci_energy))
        print('Correlation CISD: {:.8f}'.format(cis_calc.e_corr))
        cisd_list.append(cis_calc.e_tot - molecule.fci_energy)

    plt.plot(range_orbitals, casci_list, label='CASCI')
    plt.plot(range_orbitals, cisd_list, label='CISD')
    plt.ylim(0, None)
    plt.legend()
    plt.show()
