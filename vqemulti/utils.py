from openfermion.utils import count_qubits
from openfermion.ops.representations import InteractionOperator
from openfermion.transforms import jordan_wigner, bravyi_kitaev, parity_code, binary_code_transform, get_fermion_operator
from openfermion import get_sparse_operator as get_sparse_operator_openfermion
from vqemulti.preferences import Configuration
import openfermion
import numpy as np
import scipy
import warnings


def string_to_matrix(pauli_string):
    """
    Converts a Pauli string to its matrix form

    :param pauli_string: the Pauli string (e.g. 'IXYIZ')
    :return: the corresponding matrix, in the computational basis
    """

    # Define necessary Pauli operators (two-dimensional) as matrices
    pauli_x = np.array([[0, 1],
                       [1, 0]],
                      dtype=complex)
    pauli_z = np.array([[1, 0],
                       [0, -1]],
                      dtype=complex)
    pauli_y = np.array([[0, -1j],
                       [1j, 0]],
                      dtype=complex)

    matrix = np.array([1])

    # Iteratively construct the matrix, going through each single qubit Pauli term
    for pauli in pauli_string:
        if pauli == "I":
            matrix = np.kron(matrix, np.identity(2))
        elif pauli == "X":
            matrix = np.kron(matrix, pauli_x)
        elif pauli == "Y":
            matrix = np.kron(matrix, pauli_y)
        elif pauli == "Z":
            matrix = np.kron(matrix, pauli_z)

    return matrix


def ansatz_to_matrix(ansatz, n_qubits):
    """
    generate a matrix representation of the operator that corresponds to ansatz

    :param ansatz: list of operators
    :param n_qubits: number of quibits
    :return: the matrix
    """

    identity = scipy.sparse.identity(2, format='csc', dtype=complex)
    matrix = identity

    for _ in range(n_qubits - 1):
        matrix = scipy.sparse.kron(identity, matrix, 'csc')

    for operator in ansatz:
        # Get corresponding the operator matrix (exponent)
        operator_matrix = get_sparse_operator_openfermion(operator, n_qubits)

        # Add unitary operator to matrix as exp(operator_matrix)
        matrix = scipy.sparse.linalg.expm(operator_matrix) * matrix

    return matrix


def ansatz_to_matrix_list(ansatz, n_qubits):
    """
    generate a list of matrix representations of the operators that corresponds to each element of the ansatz list

    :param ansatz: list of operators
    :param n_qubits: number of quibits
    :return: list of matrices
    """

    matrix_list = []
    for operator in ansatz:
        # Get corresponding the operator matrix (exponent)
        operator_matrix = get_sparse_operator_openfermion(operator, n_qubits)

        # Add unitary operator to matrix as exp(operator_matrix)
        matrix_list.append(scipy.sparse.linalg.expm(operator_matrix))

    return matrix_list


def fock_to_bk(fock_vector):
    bk_vector = []
    for i, occ in enumerate(fock_vector):
        if np.mod(i, 2) == 0:
            bk_vector.append(occ)
        else:
            bk_vector.append(int(np.mod(np.sum(fock_vector[:i + 1]), 2)))

    return bk_vector


def fock_to_parity(fock_vector):
    parity_vector = []
    for i, occ in enumerate(fock_vector):
        parity_vector.append(int(np.mod(np.sum(fock_vector[:i + 1]), 2)))

    return parity_vector


def get_sparse_ket_from_fock(fock_vector):
    """
    Transforms a state represented in Fock space to sparse vector.

    :param fock_vector: a list representing the Fock vector
    :return: the corresponding sparse vector
    """

    state_vector = [1]

    # Iterate through the ket, calculating the tensor product of the qubit states
    for i in fock_vector:
        qubit_vector = [not i, i]
        state_vector = np.kron(state_vector, qubit_vector)

    return scipy.sparse.csc_matrix(state_vector, dtype=complex).transpose()


def get_hf_reference_in_fock_space(electron_number, qubit_number, frozen_core=0):
    """
    Get the Hartree Fock reference in Fock space vector
    The order is: [orbital_1-alpha, orbital_1-beta, orbital_2-alpha, orbital_2-beta, orbital_3-alpha.. ]

    :param electron_number: number of electrons
    :param qubit_number: the number of qubits necessary to represent the molecule
    :param frozen_core: number of orbitals that are frozen and not explicitly defined in Fock space
    :param mapping: mapping of fermions to qubits (jw: Jordan-Wigner, bk: Bravyi-Kitaev)
    :return: the vector in the Fock space
    """

    # This considers occupied the lower energy orbitals
    hf_reference = np.zeros(qubit_number, dtype=int)
    for i in range(electron_number - frozen_core * 2):
        hf_reference[i] = 1

    if Configuration().mapping == 'bk':
        hf_reference = fock_to_bk(hf_reference)
    if Configuration().mapping == 'pc':
        hf_reference = fock_to_parity(hf_reference)

    return list(hf_reference)


def find_sub_strings(mainString, hamiltonian, checked=()):
    """
    Finds and groups all the strings in a Hamiltonian that only differ from
    mainString by identity operators.

    :param mainString: a Pauli string (e.g. "XZ)
    :param hamiltonian: a Hamiltonian (with Pauli strings as keys and their coefficients as values)
    :param checked: a list of the strings in the Hamiltonian that have already been inserted in another group
    :return: groupedOperators (dict): a dictionary whose keys are boolean strings
             representing substrings of the mainString (e.g. if mainString = "XZ",
             "IZ" would be represented as "01"). It includes all the strings in the
             hamiltonian that can be written in this form (because they only differ
             from mainString by identities), except for those that were in checked
             (because they are already part of another group of strings).
             checked (list):  the same list passed as an argument, with extra values
             (the strings that were grouped in this function call).
    """
    grouped_operators = {}

    # Go through the keys in the dictionary representing the Hamiltonian that
    # haven't been grouped yet, and find those that only differ from mainString
    # by identities
    for pauliString in hamiltonian:

        if pauliString not in checked:
            # The string hasn't been grouped yet

            if all((op1 == op2 or op2 == "I") for op1, op2 in zip(mainString, pauliString)):
                # The string only differs from mainString by identities

                # Represent the string as a substring of the main one
                booleanString = "".join([str(int(op1 == op2)) for op1, op2 in \
                                         zip(mainString, pauliString)])

                # Add the boolean string representing this string as a key to
                # the dictionary of grouped operators, and associate its
                # coefficient as its value
                grouped_operators[booleanString] = hamiltonian[pauliString]

                # Mark the string as grouped, so that it's not added to any
                # other group
                checked.append(pauliString)

    return (grouped_operators, checked)


def group_hamiltonian(hamiltonian):
    """
    Organizes a Hamiltonian into Abelian commutative groups where strings only differ from
    identities, so that the expectation values of all the strings in each
    group can be calculated from the same measurement array.

    :param hamiltonian: a dictionary representing a Hamiltonian, with Pauli strings as keys and their coefficients as values
    :return: grouped_hamiltonian (dict): a dictionary of subhamiltonians, each of
             which includes Pauli strings that only differ from each other by
             identities.
             The keys of grouped_hamiltonian are the main strings of each group: the
             ones with least identity terms. The value associated to a main string is
             a dictionary, whose keys are boolean strings representing substrings of
             the respective main string (with 1 where the Pauli is the same, and 0
             where it's identity instead). The values are their coefficients
    """
    grouped_hamiltonian = {}
    checked = []

    # Go through the hamiltonian, starting by the terms that have less
    # identity operators
    for main_string in \
            sorted(hamiltonian, key=lambda pauliString: pauliString.count("I")):

        # Call findSubStrings to find all the strings in the dictionary that
        # only differ from main_string by identities, and organize them as a
        # dictionary (grouped_operators)
        grouped_operators, checked = find_sub_strings(main_string, hamiltonian, checked)

        # Use the dictionary as a value for the main_string key in the
        # grouped_hamiltonian dictionary
        grouped_hamiltonian[main_string] = grouped_operators

        # If all the strings have been grouped, exit the for cycle
        if len(checked) == len(hamiltonian.keys()):
            break

    return grouped_hamiltonian


def convert_hamiltonian(qubit_hamiltonian):
    """
    Separates a qubit Hamiltonian in dictionary entries, so that it's a suitable
    argument for functions such as measure_expectation_estimation.

    :param qubit_hamiltonian: hamiltonian in qubits
    :return: formatted hamiltonian
    """

    formatted_hamiltonian = {}
    qubit_number = count_qubits(qubit_hamiltonian)

    # Iterate through the terms in the Hamiltonian
    for term in qubit_hamiltonian.get_operators():

        operators = []
        coefficient = list(term.terms.values())[0]
        pauli_string = list(term.terms.keys())[0]
        previous_qubit = -1

        for (qubit, operator) in pauli_string:

            # If there are qubits in which no operations are performed, add identities
            # as necessary, to make sure that the length of the string will match the
            # number of qubits
            identities = (qubit - previous_qubit - 1)

            if identities > 0:
                operators.append('I' * identities)

            operators.append(operator)
            previous_qubit = qubit

        # Add final identity operators if the string still doesn't have the
        # correct length (because no operations are performed in the last qubits)
        operators.append('I' * (qubit_number - previous_qubit - 1))

        formatted_hamiltonian["".join(operators)] = coefficient

    return formatted_hamiltonian


def generate_reduced_hamiltonian(hamiltonian, n_orbitals, frozen_core=0):
    """
    get truncated hamiltonian given a number of orbitals

    :param hamiltonian: hamiltonian in fermionic operators
    :param n_orbitals: number of orbitals to keep
    :param frozen_core: number of orbitals to freeze
    :return: truncated hamiltonian in fermionic operators
    """
    frozen_spin_orbitals = frozen_core * 2
    n_spin_orbitals = n_orbitals * 2

    reduced_one = hamiltonian.one_body_tensor[frozen_spin_orbitals: n_spin_orbitals,
                                              frozen_spin_orbitals: n_spin_orbitals]
    reduced_two = hamiltonian.two_body_tensor[frozen_spin_orbitals: n_spin_orbitals,
                                              frozen_spin_orbitals: n_spin_orbitals,
                                              frozen_spin_orbitals: n_spin_orbitals,
                                              frozen_spin_orbitals: n_spin_orbitals]

    energy_inactive = 0
    if frozen_core > 0:
        # compute the core energy
        for j in range(frozen_spin_orbitals):
            energy_inactive += hamiltonian.one_body_tensor[j, j]

        for j in range(frozen_spin_orbitals):
            for i in range(frozen_spin_orbitals):
               energy_inactive += hamiltonian.two_body_tensor[j, i, i, j] - hamiltonian.two_body_tensor[j, j, i, i]

        # print('inactive Ham', energy_inactive)

        # construct effective hamiltonian
        def core_effective(p, q):

            effective_energy = 0
            pp = p + frozen_spin_orbitals
            qq = q + frozen_spin_orbitals

            for i in range(frozen_spin_orbitals):
                effective_energy += hamiltonian.two_body_tensor[i, qq, pp, i] - hamiltonian.two_body_tensor[i, i, pp, qq]
            return effective_energy*2

        for i in range(n_spin_orbitals - frozen_spin_orbitals):
            for j in range(n_spin_orbitals - frozen_spin_orbitals):
                reduced_one[i, j] += core_effective(i, j)

    return InteractionOperator(hamiltonian.constant + energy_inactive, reduced_one, reduced_two)


def get_uccsd_operators(n_electrons, n_orbitals, frozen_core=0):
    """
    get all UCCSD operators with coefficients as ones

    :param n_electrons: number of electrons
    :param n_orbitals: number of orbitals
    :param frozen_core: number of orbitals to freeze
    :return: UCCSD operators in fermion representation
    """
    warnings.warn('To be deprecated. Use pool singlet_sd')

    n_occupied = int(np.ceil(n_electrons / 2)) - frozen_core
    n_virtual = n_orbitals - n_occupied - frozen_core

    singles = []
    doubles_1 = []
    doubles_2 = []
    import itertools
    for p, q in itertools.product(range(n_virtual), range(n_occupied)):
        singles.append(1)
        doubles_1.append(1)
    for (p, q), (r, s) in itertools.combinations(
            itertools.product(range(n_virtual), range(n_occupied)), 2):
        doubles_2.append(1)

    packed_amplitudes = singles + doubles_1 + doubles_2

    return -openfermion.uccsd_singlet_generator(packed_amplitudes,
                                               (n_occupied + n_virtual) * 2,
                                               n_occupied * 2)


def get_hf_energy_core(mol_h2, n_core_orb=0):
    from pyscf import ao2mo

    nuclear = mol_h2.nuclear_repulsion
    hf_orb = mol_h2.n_electrons//2
    print('hf_orb: ', hf_orb)

    rhf_h2 = mol_h2._pyscf_data['scf']
    mol_h2 = mol_h2._pyscf_data['mol']

    #rhf_h2 = mol_h2.RHF()
    #e_rhf_h2 = rhf_h2.kernel()

    Fao = rhf_h2.get_fock()
    Fmo = rhf_h2.mo_coeff.T @ Fao @ rhf_h2.mo_coeff

    print('Fock matrix (MO)')
    print(Fmo)

    Jao = rhf_h2.get_j()
    Jmo = rhf_h2.mo_coeff.T @ Jao @ rhf_h2.mo_coeff
    print('J matrix (MO)')
    print(Jmo)
    #print(Jao)

    Kao = rhf_h2.get_k()
    Kmo = rhf_h2.mo_coeff.T @ Kao @ rhf_h2.mo_coeff
    print('K matrix (MO)')
    print(Kmo)
    # print(Kao)

    print('Overlap')
    eri = mol_h2.intor('int1e_ovlp', aosym='s1')
    print(eri)

    print('Bi-electronic (MO)')
    eri = mol_h2.intor('int2e', aosym='s1')
    eri_mo = ao2mo.kernel(eri, rhf_h2.mo_coeff, aosym='s1')
    print(eri_mo)

    print('Mono-electronic (MO)')
    eri = mol_h2.intor('int1e_nuc', aosym='s1') + \
          mol_h2.intor('int1e_kin', aosym='s1')
    h_mo = rhf_h2.mo_coeff.T @ eri @ rhf_h2.mo_coeff
    print(h_mo)

    d_mo = np.zeros_like(h_mo)
    for i in range(n_core_orb):
        d_mo[i, i] = 2.0
    print('Density matrix (mo)')
    print(d_mo)

    h_energy = np.sum(d_mo * h_mo)
    coulomb_energy = np.sum(d_mo * Jmo / 2)
    exchange_energy = -0.5 * np.sum(d_mo * Kmo / 2)

    print('Computed Properties\n')
    print('1e energy:', h_energy)
    print('Total Coulomb:', coulomb_energy)
    print('HF Exchange:', exchange_energy)
    print('HF Total Electronic: {:12.8f}'.format(h_energy + coulomb_energy + exchange_energy))
    print('HF Total:', h_energy + coulomb_energy + exchange_energy + nuclear)
    print('--------------------')

    energy = 0
    for i in range(n_core_orb):
        energy += 2*h_mo[i, i]

    def F_effect(p, q):
        tot_j = 0
        for i in range(n_core_orb):
            tot_j += 2*eri_mo[i, i, p, q] - eri_mo[i, q, p, i]

        return tot_j

    for i in range(n_core_orb):
        # energy += Jmo[i, i] - 0.5*Kmo[i, i]
        energy += F_effect(i, i)

    print('HF Test  Electronic (base): {:12.8f}'.format(energy))

    def F_effect_2(p, q):
        tot_j = 0
        for i in range(n_core_orb, hf_orb):
            tot_j += 2*eri_mo[i, i, p, q] - eri_mo[i, q, p, i]

        return tot_j
    for i in range(n_core_orb):
        # energy += Jmo[i, i] - 0.5*Kmo[i, i]
        energy += F_effect_2(i, i)

    print('HF Test  Electronic (effective): {:12.8f}'.format(energy))


def normalize_operator(operator):
    """
    normalize an operator

    :param operator:
    :return: the normalized operator
    """

    coeff = 0
    for t in operator.terms:
        coeff_t = operator.terms[t]
        coeff += np.conj(coeff_t) * coeff_t

    if operator.many_body_order() > 0:
        return operator / np.sqrt(coeff)

    raise Exception('Cannot normalize 0 operator')


def fermion_to_qubit(operator):
    """
    transform fermions to qubits

    :param operator: fermion operator
    :return: qubit operator
    """

    if Configuration().mapping == 'jw':
        return jordan_wigner(operator)
    elif Configuration().mapping == 'bk':
        return bravyi_kitaev(operator)
    elif Configuration().mapping == 'pc':
        if isinstance(operator, InteractionOperator):
            operator = get_fermion_operator(operator)
        return binary_code_transform(operator, parity_code(count_qubits(operator)))

    raise Exception('{} mapping not implemented'.format(Configuration().mapping))


def sort_tuples(tuples):
    """
    auxilar function to order a list of tuples of 2 items along the first index.
    To be used in sorting fermion operators

    :param tuples: list of tuples. Ex. ((3, 1), (4, 1), (2, 0), (1, 0))
    :return: ordered tuples
    """
    tuples = list(tuples)
    count = 1

    for index in range(len(tuples)):
        idx = np.argmax(np.array(tuples).T[0][index:])

        if index != idx + index:
            tuples[index], tuples[idx + index] = tuples[idx + index], tuples[index]
            count *= -1

    return tuple(tuples), count


def proper_order(ansatz):
    """
    reorder fermions to proper order (from higher to lower orbital)

    :param ansatz: Fermion operators
    :return: ordered Fermion operators
    """
    from openfermion import normal_ordered

    total = openfermion.FermionOperator()
    for term in ansatz:
        for i, v in term.terms.items():
            t, c = sort_tuples(i)
            total += c*v*openfermion.FermionOperator(t)

    return normal_ordered(total)


def cache_operator(func):
    cache_dict = {}
    import openfermion

    def wrapper_cache(*args, **kwargs):

        operator = args[0]
        if isinstance(operator, openfermion.InteractionOperator):
            operator = openfermion.get_fermion_operator(operator)

        hash_key = frozenset(operator.terms.items())
        if len(args) > 1:
            hash_key = (hash_key, args[1])

        if len(kwargs) > 1:
            hash_key = (hash_key, frozenset(kwargs.items()))

        if hash_key in cache_dict:
            return cache_dict[hash_key]

        cache_dict[hash_key] = func(*args, **kwargs)
        return cache_dict[hash_key]

    return wrapper_cache


@cache_operator
def get_sparse_operator(operator, n_qubits=None, trunc=None, hbar=1.):
    """
    wrapper over openfermion's get_sparse_operator for convenience

    :param operator: Currently supported operators include: FermionOperator,
                     QubitOperator, DiagonalCoulombHamiltonian, PolynomialTensor, BosonOperator, QuadOperator.
    :param n_qubits: Number qubits in the system Hilbert space. Applicable only to fermionic systems.
    :param trunc: The size at which the Fock space should be truncated. Applicable only to bosonic systems.
    :param hbar: the value of hbar to use in the definition of the canonical commutation
                  relation [q_i, p_j] = \delta_{ij} i hbar. Applicable only to the QuadOperator.

    :return:
    """
    if Configuration().mapping == 'bk':
        if isinstance(operator, (openfermion.FermionOperator, openfermion.InteractionOperator)):
            operator = bravyi_kitaev(operator)
    if Configuration().mapping == 'pc':
        if isinstance(operator, (openfermion.FermionOperator, openfermion.InteractionOperator)):
            operator = binary_code_transform(operator, parity_code(count_qubits(operator)))

    return get_sparse_operator_openfermion(operator, n_qubits, trunc, hbar)


def get_string_from_fermionic_operator(operator):
    """
    return a string representation of a fermionic unitary operator

    :param operator: operator
    :return: string
    """
    operator_string = ''
    total_spins_string = ''
    for term, coefficient in operator.terms.items():

        spins_term = ''
        operator_string = '('
        for ti in term:
            operator_string += str(int(ti[0] / 2))
            if ti[1] == 0:
                operator_string += "  "
            elif ti[1] == 1:
                operator_string += "' "

            spins_term += str(ti[0] % 2)

        operator_string += ')'
        if coefficient > 0:
            total_spins_string += spins_term + ' '
        else:
            total_spins_string += '[' + spins_term + '] '

    return ' {:>18} : {}'.format(operator_string, total_spins_string)

