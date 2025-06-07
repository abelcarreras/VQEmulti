from posym import SymmetryMolecule, SymmetryGaussianLinear, SymmetryObject
from posym.tools import get_basis_set_pyscf, build_density, build_orbital
from posym.algebra import dot as sym_dot
from scipy.linalg import expm
from scipy.optimize import minimize
from itertools import combinations
from vqemulti.pool.tools import OperatorList
import numpy as np


def plot_orbital(orbital):
    import matplotlib.pyplot as plt

    x = np.linspace(-2, 5, 50)
    y = np.linspace(-2, 5, 50)

    X, Y = np.meshgrid(x, y)

    Z = orbital(X, Y, np.zeros_like(X))
    plt.imshow(Z, interpolation='bilinear', origin='lower', cmap='seismic')
    plt.figure()
    plt.axes().set_aspect('equal')
    plt.contour(X, Y, Z, colors='k')
    plt.show()


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


def symmetrize_molecular_orbitals(molecule, group, skip_first=0, skip=False, frozen_core=0, n_orbitals=None):

    #

    # get geometric data
    symbols = [atom[0] for atom in molecule.geometry]
    coordinates = molecule._pyscf_data['mol'].atom_coords()

    # get electronic data
    mo_coefficients = molecule.canonical_orbitals

    if n_orbitals is None:
        n_orbitals = len(mo_coefficients)

    mo_coefficients = mo_coefficients[:, frozen_core: n_orbitals]

    geom_sym = SymmetryMolecule(group, coordinates, symbols)
    print('geom_sym {} : {}'.format(group, geom_sym))
    # print(geom_sym.orientation_angles)
    # print(geom_sym.center)

    # get electronic data
    basis_set = get_basis_set_pyscf(molecule._pyscf_data['mol'])

    print('\nMO symmetry (initial)')
    sym_orbitals = []
    for i, orbital_vect in enumerate(mo_coefficients.T):
        orb = build_orbital(basis_set, orbital_vect)

        sym_orb = SymmetryGaussianLinear(group, orb,
                                         # orientation_angles=geom_sym.orientation_angles,
                                         # center=geom_sym.center
                                         )
        values = sym_orb.get_ir_representation().values
        cost = cost_fuction(values)
        sym_orbitals.append(sym_orb)

        print('orbital {:3}: {:6.2f} ({})'.format(i, cost, sym_orb))

    if skip:
        return sym_orbitals

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

    mo_coefficients = np.array(mo_coefficients, copy=True)
    n_orbitals = len(mo_coefficients.T)
    for n_orb_chunk in range(skip_first, n_orbitals-1):
        # print('Symmetry iteration: ', n_orb_chunk)
        restrict = mo_coefficients[:, n_orb_chunk:]

        n_dim = len(restrict.T)
        n_params = (n_dim ** 2 - n_dim) // 2
        params = np.zeros(n_params)

        res = minimize(minimize_function, params, args=(restrict, group), method='BFGS', options={'gtol': 1e-3})

        u_matrix = generate_unitary(n_dim, res.x)

        restrict = restrict @ u_matrix
        mo_coefficients[:, n_orb_chunk:] = restrict[:, :]

    # final check
    print('\nMO symmetry (final)')
    sym_orbitals = []
    for i, orbital_vect in enumerate(mo_coefficients.T):
        orb = build_orbital(basis_set, orbital_vect)

        sym_orb = SymmetryGaussianLinear(group, orb,
                                         orientation_angles=geom_sym.orientation_angles,
                                         center=geom_sym.center
                                         )
        values = sym_orb.get_ir_representation().values
        cost = np.sum([np.prod(pair) for pair in combinations(values, 2)])
        sym_orbitals.append(sym_orb)

        print('orbital {:3}: {:6.2f} ({})'.format(i, cost, sym_orb))
        # print('values: ', values)

    from openfermionpyscf import set_mo_coefficients
    set_mo_coefficients(molecule, mo_coefficients)

    return sym_orbitals


def get_hamiltonian_terms_symmetry(hamiltonian, sym_orbitals):

    tot_sum = []
    for term in hamiltonian.unique_iter():
        if len(term) == 0:
            continue

        prod_list = []
        for op in term:
            orb_index = op[0]//2
            prod_list.append(sym_orbitals[orb_index])
        # print(term, np.prod(prod_list))
        tot_sum.append(np.prod(prod_list))

    return np.sum(tot_sum)


def get_pool_symmetry(pool, sym_orbitals):

    ferm_op_list = []
    for full_op in pool:
        tot_sum = []
        for term in full_op.terms:
            if len(term) == 0:
                continue

            prod_list = []
            for op in term:
                orb_index = op[0]//2
                prod_list.append(sym_orbitals[orb_index])

            tot_sum.append(np.prod(prod_list))

        ferm_op_list.append(np.sum(tot_sum))

    return ferm_op_list


def get_symmetry_reduced_pool(pool, sym_orbitals, hamiltonian_sym=None, threshold=1e-2):

    rep = sym_orbitals[0].get_point_group().ir_labels[0]
    group = sym_orbitals[0].get_point_group().label

    # assuming hamiltonian symmetry is most symmetric IR
    if hamiltonian_sym is None:
        hamiltonian_sym = SymmetryObject(group=group, rep=rep)

    # assuming state symmetry is most symmetric IR
    state_sym = SymmetryObject(group=group, rep=rep)

    pool_sym = get_pool_symmetry(pool, sym_orbitals)

    print('\npool symmetry')
    new_pool = []
    for i, op_sym in enumerate(pool_sym):
        dot_sym = abs(sym_dot(op_sym * hamiltonian_sym, state_sym))
        if dot_sym > threshold:
            new_pool.append(pool[i])
            print('operator {:3} : {:6.2f} {:6}  ({})'.format(i, dot_sym, str(True), op_sym))
        else:
            print('operator {:3} : {:6.2f} {:6}  ({})'.format(i, dot_sym, str(False), op_sym))

    return OperatorList(new_pool, antisymmetrize=False)


def compare_operators(operator1, operator2):

    def apply_rotation(p):
        rot_p = list(p)
        for i, qubit in enumerate(rot_p):
            if qubit[1] == 'Y':
                rot_p[i] = (qubit[0], 'Z')
            if qubit[1] == 'Z':
                rot_p[i] = (qubit[0], 'Y')
        yield tuple(rot_p)

        rot_p = list(p)
        for i, qubit in enumerate(rot_p):
            if qubit[1] == 'Z':
                rot_p[i] = (qubit[0], 'Y')
            if qubit[1] == 'Z':
                rot_p[i] = (qubit[0], 'Y')
        yield tuple(rot_p)

        rot_p = list(p)
        for i, qubit in enumerate(rot_p):
            if qubit[1] == 'X':
                rot_p[i] = (qubit[0], 'Y')
            if qubit[1] == 'Y':
                rot_p[i] = (qubit[0], 'X')
        yield tuple(rot_p)

    for (k1, v1), (k2, v2) in zip(operator1.terms.items(), operator2.terms.items()):
        for ki in apply_rotation(k1):
            if ki == k2:
                return True

    return False


def get_pauli_symmetry_reduced_pool(pool):

    print('\npool commutation')
    new_pool = []
    for i, op_1 in enumerate(pool):

        included = False
        for op_2 in new_pool:
            included = included or compare_operators(op_1, op_2)

        if not included:
            new_pool.append(op_1)

    return OperatorList(new_pool, antisymmetrize=False)


if __name__ == '__main__':

    from openfermionpyscf import prepare_pyscf_molecule
    from openfermion import MolecularData, hermitian_conjugated
    from openfermionpyscf import run_pyscf
    import matplotlib.pyplot as plt

    from openfermion import QubitOperator


    def compare_operators(operator1, operator2):

        def apply_rotation(p):
            rot_p = list(p)
            for i, qubit in enumerate(rot_p):
                if qubit[1] == 'Y':
                    rot_p[i] = (qubit[0], 'Z')
                if qubit[1] == 'Z':
                    rot_p[i] = (qubit[0], 'Y')
            yield tuple(rot_p)

            rot_p = list(p)
            for i, qubit in enumerate(rot_p):
                if qubit[1] == 'Z':
                    rot_p[i] = (qubit[0], 'Y')
                if qubit[1] == 'Z':
                    rot_p[i] = (qubit[0], 'Y')
            yield tuple(rot_p)

            rot_p = list(p)
            for i, qubit in enumerate(rot_p):
                if qubit[1] == 'X':
                    rot_p[i] = (qubit[0], 'Y')
                if qubit[1] == 'Y':
                    rot_p[i] = (qubit[0], 'X')
            yield tuple(rot_p)

        for (k1, v1), (k2, v2) in zip(operator1.terms.items(), operator2.terms.items()):
            for ki in apply_rotation(k1):
                if ki == k2:
                    return True

        return False

    pauli_1 = QubitOperator('X0 Y1 Y2 Y3')
    pauli_2 = QubitOperator('Y0 X1 X2 X3')



    print(compare_operators(pauli_1, pauli_2))

    # print(pauli_1 + hermitian_conjugated(pauli_2))
    # print(pauli_1 - hermitian_conjugated(pauli_2))


    exit()

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

    mo_coefficients = symmetrize_molecular_orbitals(mo_coefficients, coordinates, symbols, 'd4h', skip=False)

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
