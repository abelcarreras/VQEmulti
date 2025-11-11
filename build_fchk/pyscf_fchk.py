import numpy as np
angstrom_to_bohr = 1/0.529177249


def mat_to_vect(matrix):
    matrix = np.array(matrix)
    tril_matrix = matrix[np.tril_indices_from(matrix)]
    return tril_matrix.tolist()


def get_array_txt(label, type, array, row_size=5):

    formats = {'R': '15.8e',
               'I': '11'}

    array = np.array(array, dtype=float if type == 'R' else int).flatten()
    n_elements = len(array)
    rows = int(np.ceil(n_elements/row_size))

    txt_fchk = '{:40}   {}   N=       {:5}\n'.format(label, type, n_elements)

    # print(rows)
    for i in range(rows):
        if (i+1)*row_size > n_elements:
            txt_fchk += (' {:{fmt}}'* (n_elements - i*row_size) + '\n').format(*array[i * row_size:n_elements],
                                                                               fmt=formats[type])
        else:
            txt_fchk += (' {:{fmt}}'* row_size + '\n').format(*array[i * row_size: (i+1)*row_size],
                                                              fmt=formats[type])

    return txt_fchk


def write_to_fchk(parsed_data, filename):
    """
    write a FCHK file from electronic structure dictionary

    :param parsed_data: electronic structure dictionary
    :param filename: file name
    :return: None
    """
    txt = build_fchk(parsed_data)
    with open(filename, 'w') as f:
        f.write(txt)




def build_fchk(parsed_data):
    """
    build a string containing in FCHK file format from an electronic structure dictionary

    :param parsed_data: electronic structure dictionary
    :return: string in FCHK file format
    """

    structure = parsed_data['structure']
    basis = parsed_data['basis']
    alpha_mo_coeff = parsed_data['coefficients']['alpha']
    alpha_mo_energies = parsed_data['mo_energies']['alpha']

    number_of_basis_functions = len(alpha_mo_coeff)
    number_of_electrons = np.sum(structure.get_atomic_numbers()) - structure.charge
    if structure.multiplicity > 1:
        # raise Exception('1> multiplicity not yet implemented')
        pass

    alpha_mo_coeff = np.array(alpha_mo_coeff).flatten().tolist()

    shell_type_list = {'s':  {'type':  0, 'angular_momentum': 0},
                       'p':  {'type':  1, 'angular_momentum': 1},
                       'd':  {'type':  2, 'angular_momentum': 2},
                       'f':  {'type':  3, 'angular_momentum': 3},
                       'g':  {'type':  4, 'angular_momentum': 4},
                       'sp': {'type': -1, 'angular_momentum': 1},  # hybrid
                       'd_': {'type': -2, 'angular_momentum': 2},  # pure
                       'f_': {'type': -3, 'angular_momentum': 3},  # pure
                       'g_': {'type': -4, 'angular_momentum': 4}}  # pure

    shell_type = []
    p_exponents = []
    c_coefficients = []
    p_c_coefficients = []
    n_primitives = []
    atom_map = []

    largest_degree_of_contraction = 0
    highest_angular_momentum = 0
    number_of_contracted_shells = 0

    for i, atoms in enumerate(basis['atoms']):
        for shell in atoms['shells']:
            number_of_contracted_shells += 1
            st = shell['shell_type']
            shell_type.append(shell_type_list[st]['type'])
            n_primitives.append(len(shell['p_exponents']))
            atom_map.append(i+1)
            if highest_angular_momentum < shell_type_list[st]['angular_momentum']:
                highest_angular_momentum = shell_type_list[st]['angular_momentum']

            if len(shell['con_coefficients']) > largest_degree_of_contraction:
                    largest_degree_of_contraction = len(shell['con_coefficients'])

            for p in shell['p_exponents']:
                p_exponents.append(p)
            for c in shell['con_coefficients']:
                c_coefficients.append(c)
            for pc in shell['p_con_coefficients']:
                p_c_coefficients.append(pc)

    coordinates_list = angstrom_to_bohr*np.array(structure.get_coordinates()).flatten()

    txt_fchk = '{}\n'.format('filename')
    txt_fchk += 'SP        R                             {}\n'.format(basis['name'] if 'name' in basis else 'no_name')
    txt_fchk += 'Number of atoms                            I               {}\n'.format(structure.get_number_of_atoms())
    txt_fchk += 'Charge                                     I               {}\n'.format(structure.charge)
    txt_fchk += 'Multiplicity                               I               {}\n'.format(structure.multiplicity)
    txt_fchk += 'Number of electrons                        I               {}\n'.format(number_of_electrons)
    txt_fchk += 'Number of alpha electrons                  I               {}\n'.format(structure.alpha_electrons)
    txt_fchk += 'Number of beta electrons                   I               {}\n'.format(structure.beta_electrons)

    txt_fchk += get_array_txt('Atomic numbers', 'I', structure.get_atomic_numbers(), row_size=6)
    txt_fchk += get_array_txt('Current cartesian coordinates', 'R', coordinates_list)
    txt_fchk += get_array_txt('Nuclear charges', 'R', structure.get_atomic_numbers())

    txt_fchk += 'Number of basis functions                  I               {}\n'.format(number_of_basis_functions)
    txt_fchk += 'Number of contracted shells                I               {}\n'.format(number_of_contracted_shells)
    txt_fchk += 'Number of primitive shells                 I               {}\n'.format(np.sum(n_primitives))
    txt_fchk += 'Highest angular momentum                   I               {}\n'.format(highest_angular_momentum)
    txt_fchk += 'Largest degree of contraction              I               {}\n'.format(largest_degree_of_contraction)

    txt_fchk += get_array_txt('Shell types', 'I', shell_type, row_size=6)
    txt_fchk += get_array_txt('Number of primitives per shell', 'I', n_primitives, row_size=6)
    txt_fchk += get_array_txt('Shell to atom map', 'I', atom_map, row_size=6)
    txt_fchk += get_array_txt('Primitive exponents', 'R', p_exponents)
    txt_fchk += get_array_txt('Contraction coefficients', 'R', c_coefficients)
    txt_fchk += get_array_txt('P(S=P) Contraction coefficients', 'R', p_c_coefficients)
    # txt_fchk += get_array_txt('Coordinates of each shell', 'R', coor_shell) #
    # txt_fchk += get_array_txt('Overlap Matrix', 'R', overlap)
    #txt_fchk += get_array_txt('Core Hamiltonian Matrix', 'R', core_hamiltonian)
    txt_fchk += get_array_txt('Alpha Orbital Energies', 'R', alpha_mo_energies)
    # txt_fchk += get_array_txt('Beta Orbital Energies', 'R', beta_mo_energies)

    if 'scf_density' in parsed_data:
        total_density = np.array(parsed_data['scf_density']['alpha']) + np.array(parsed_data['scf_density']['beta'])
        scf_density = mat_to_vect(total_density)
        txt_fchk += get_array_txt('Total SCF Density', 'R', scf_density)

    if 'spin_density' in parsed_data:
        spin_density = mat_to_vect(parsed_data['spin_density'])
        txt_fchk += get_array_txt('Spin SCF Density', 'R', spin_density)

    txt_fchk += get_array_txt('Alpha MO coefficients', 'R', alpha_mo_coeff)

    if 'beta' in parsed_data['coefficients']:
        beta_mo_coeff = parsed_data['coefficients']['beta']
        beta_mo_coeff = np.array(beta_mo_coeff).flatten().tolist()
        beta_mo_energies = parsed_data['mo_energies']['beta']

        txt_fchk += get_array_txt('Beta MO coefficients', 'R', beta_mo_coeff)
        txt_fchk += get_array_txt('Beta Orbital Energies', 'R', beta_mo_energies)

    # Fractional occupation density
    if 'fractional_occupation_density' in parsed_data:
        fod = mat_to_vect(parsed_data['fractional_occupation_density'])
        txt_fchk += get_array_txt('Fractional occupation density', 'R', fod)


    # Natural orbitals
    if 'nato_coefficients' in parsed_data:
        for type, nato_coeff  in parsed_data['nato_coefficients'].items():
            nato_coeff = np.array(nato_coeff).flatten().tolist()
            trans_dict = {'alpha': 'Alpha', 'beta': 'Beta'}
            txt_fchk += get_array_txt('{} NATO coefficients '.format(trans_dict[type]), 'R', nato_coeff)

    if 'nato_occupancies' in parsed_data:
        for type, nato_occ  in parsed_data['nato_occupancies'].items():
            nato_occ = np.array(nato_occ).flatten().tolist()
            trans_dict = {'alpha': 'Alpha', 'beta': 'Beta'}
            txt_fchk += get_array_txt('{} Natural Orbital occupancies '.format(trans_dict[type]), 'R', nato_occ)

    return txt_fchk

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
        def __init__(self, coords, atomic_symbols, atomic_numbers, charge, n_alpha, n_beta):
            self._coords = coords
            self._symbols = atomic_symbols
            self._atomic_numbers = atomic_numbers
            self.charge = charge
            self.multiplicity = np.abs(n_alpha - n_beta)+1

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

    return build_fchk(parsed_data)


if __name__ == '__main__':
    # simple example

    from pyscf import scf
    from pyscf import gto

    # prepare pySCF calculation
    mol = gto.M(atom=[["H", 0., 0., 0.],
                      ["H", 0., 0., 0.8]], basis='3-21g', verbose=0)
    rhf = scf.RHF(mol).run()
    density = rhf.make_rdm1()

    # get MO orbitals & energies
    mo_coeff = rhf.mo_coeff.T.tolist()
    mo_energies = rhf.mo_energy.tolist()

    # convert to fchk
    fchk_txt = get_fchk_from_pyscf(mol, mo_coeff, mo_energies, density)

    # write fchk to file
    with open('test2.fchk', 'w') as f:
        f.write(fchk_txt)


    from pyscf.tools import molden

    molden.from_mo(mol, 'test.molden', rhf.mo_coeff)
    from iodata import load_one, dump_one


    mol = load_one('test.molden')
    print(mol)  # print coordinates in Bohr.
    dump_one(mol, 'test.fchk')


    #mol = load_fchk('test.fchk')  # XYZ files contain atomic coordinates in Angstrom
    #print(mol.atcoords)  # print coordinates in Bohr.