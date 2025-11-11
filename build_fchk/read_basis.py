import numpy as np

def basis_format(basis_set_name,
                 atomic_numbers,
                 atomic_symbols,
                 shell_type,
                 n_primitives,
                 atom_map,
                 p_exponents,
                 c_coefficients,
                 p_c_coefficients):
    """
    Function to generate standard basis dictionary

    :param basis_set_name:
    :param atomic_numbers:
    :param atomic_symbols:
    :param shell_type:
    :param n_primitives:
    :param atom_map:
    :param p_exponents:
    :param c_coefficients:
    :param p_c_coefficients:
    :return:
    """

    class TypeList:
        def __getitem__(self, item):
            typeList = {'0': ['s', 1],
                        '1': ['p', 3],
                        '2': ['d', 6],
                        '3': ['f', 10],
                        '4': ['g', 15],
                        '-1': ['sp', 4],
                        '-2': ['d_', 5],
                        '-3': ['f_', 7],
                        '-4': ['g_', 9]}
            if item in typeList:
                return typeList[item]


    type_list = TypeList()

    atomic_numbers = [int(an) for an in atomic_numbers]
    atom_map = np.array(atom_map, dtype=int)
    # print(atom_map)
    basis_set = {'name': basis_set_name,
                 'primitive_type': 'gaussian'}

    shell_type_index = [0] + np.cumsum([type_list['{}'.format(s)][1]
                                        for s in shell_type]).tolist()
    prim_from_shell_index = [0] + np.cumsum(np.array(n_primitives, dtype=int)).tolist()

    # print(shell_type_index)
    # print(prim_from_shell_index)

    atoms_data = []
    for iatom, atomic_number in enumerate(atomic_numbers):
        symbol = str(atomic_symbols[iatom])

        shell_from_atom_counts = np.unique(atom_map, return_counts=True)[1]
        shell_from_atom_index = np.unique(atom_map, return_index=True)[1]

        shells_data = []
        for ishell in range(shell_from_atom_counts[iatom]):
            st = type_list['{}'.format(shell_type[shell_from_atom_index[iatom] + ishell])]
            ini_prim = prim_from_shell_index[shell_from_atom_index[iatom] + ishell]
            fin_prim = prim_from_shell_index[shell_from_atom_index[iatom] + ishell+1]

            shells_data.append({
                'shell_type': st[0],
                'functions': st[1],
                'p_exponents': p_exponents[ini_prim: fin_prim],
                'con_coefficients': c_coefficients[ini_prim: fin_prim],
                'p_con_coefficients': p_c_coefficients[ini_prim: fin_prim],
            })

        atoms_data.append({'shells': shells_data,
                           'symbol': symbol,
                           'atomic_number': atomic_number})

    basis_set['atoms'] = atoms_data

    return basis_set


def get_structure_pyscf(pyscf_mol):
    coords = pyscf_mol.atom_coords()
    n_alpha, n_beta = pyscf_mol.nelec

    n_atoms = len(coords)
    atomic_symbols = [pyscf_mol.atom_symbol(i) for i in range(n_atoms)]

    charge = pyscf_mol.charge
    atomic_numbers = [int(pyscf_mol.atom_charge(i)) for i in range(n_atoms)]

    class Structure:
        def __init__(self, coords, atomic_symbols, atomic_numbers, charge, n_alpha, n_beta, multiplicity=0):
            self._coords = coords
            self._symbols = atomic_symbols
            self._atomic_numbers = atomic_numbers
            self.charge = charge
            self.multiplicity = multiplicity

            # to be modified
            self.alpha_electrons = n_alpha
            self.beta_electrons = n_beta

        def get_atomic_numbers(self):
            return self._atomic_numbers

        def get_coordinates(self):
            return self._coords

        def get_number_of_atoms(self):
            return len(self._coords)

    return Structure(coords, atomic_symbols, atomic_numbers, charge, n_alpha, n_beta)


def get_basis_pyscf(pyscf_mol):
    """
    get basis functions from pyscf molecule object
    :param pyscf_mol: pyscf molecule
    :return: list of basis functions
    """

    coords = pyscf_mol.atom_coords().flatten()
    n_atoms = len(coords)//3
    atomic_symbols = [pyscf_mol.atom_symbol(i) for i in range(n_atoms)]


    atomic_numbers = [int(pyscf_mol.atom_charge(i)) for i in range(n_atoms)]

    basis_set_name = pyscf_mol.basis

    shell_type_match =[0, 1, -2, -3, -4]
    shell_type = []
    n_primitives = []
    p_exponents = []
    c_coefficients = []
    p_c_coefficients = []
    atom_map = []
    for i in range(n_atoms):
        bas_ids = pyscf_mol.atom_shell_ids(i)

        atom_map += [i+1] * len(bas_ids)

        n_prim = 0
        for bas_id in bas_ids:
            shell_type.append(shell_type_match[pyscf_mol.bas_angular(bas_id)])
            n_prim += len(pyscf_mol.bas_exp(bas_id))
            pyscf_mol.bas_exp(bas_id)
            p_exponents += list(pyscf_mol.bas_exp(bas_id).flatten())
            c_coefficients += list(pyscf_mol.bas_ctr_coeff(bas_id).flatten())
            p_c_coefficients += [0] * len(pyscf_mol.bas_ctr_coeff(bas_id))

            n_primitives.append(len(pyscf_mol.bas_ctr_coeff(bas_id)))

    return basis_format(basis_set_name,
                        atomic_numbers,
                        atomic_symbols,
                        shell_type,
                        n_primitives,
                        atom_map,
                        p_exponents,
                        c_coefficients,
                        p_c_coefficients)




def get_fchk_from_pyscf(mol, mo_coeff, mo_energy, dm):
    basis = get_basis_pyscf(mol)
    structure = get_structure_pyscf(mol)

    density_alpha = density_beta = list(np.array(density)/2)
    parsed_data = {'structure': structure,
                   'basis': basis,
                   'coefficients': {'alpha': mo_coeff},
                   'mo_energies': {'alpha': mo_energy},
                   'scf_density': {'alpha': density_alpha, 'beta': density_beta}}

    from pyscf_fchk import build_fchk

    return build_fchk(parsed_data)


from pyscf import scf
from pyscf import gto

mol = gto.M(atom=[["H", 0., 0., 0.],
                  ["O", 0., 0., 0.8]], basis='3-21g', verbose=0, charge=-1)
rhf = scf.RHF(mol).run()
uhf = scf.UHF(mol)
density = rhf.make_rdm1()


mo_coeff = rhf.mo_coeff.T.tolist()
mo_energies = rhf.mo_energy.tolist()

fchk_txt = get_fchk_from_pyscf(mol, mo_coeff, mo_energies, density)

with open('test.fchk', 'w') as f:
    f.write(fchk_txt)
