from openfermion.utils import count_qubits
from openfermion.ops.representations import InteractionOperator
from openfermion.transforms import jordan_wigner, bravyi_kitaev, parity_code, binary_code_transform
from openfermion.transforms import reorder, get_interaction_operator, get_fermion_operator
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
        operator_matrix = get_sparse_operator(operator, n_qubits)

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
        operator_matrix = get_sparse_operator(operator, n_qubits)

        # Add unitary operator to matrix as exp(operator_matrix)
        matrix_list.append(scipy.sparse.linalg.expm(operator_matrix))

    return matrix_list


def should_include_in_parity(qubit_idx, fermion_idx):
    """
    Determine if fermion_idx should be included in the parity calculation for qubit_idx.
    This implements the Fenwick tree structure for BK transformation.
    """
    # Convert to 1-indexed for easier bit manipulation
    i = qubit_idx + 1
    j = fermion_idx + 1

    # Check if j is in the "update set" of i in Fenwick tree
    # This happens when j is in the subtree rooted at i

    # First check: j must be <= i
    if j > i:
        return False

    # Get the binary representations
    # For BK transformation, we need to check if j is in the Fenwick update set of i

    # Find the rightmost set bit of i (this determines the range)
    rightmost_bit = i & (-i)  # LSB operation

    # j is included if it's in the range [i - rightmost_bit + 1, i]
    return (i - rightmost_bit + 1) <= j <= i


def fock_to_bk(fock_vector):
    """
    Convert Fock state vector to Bravyi-Kitaev encoded vector.

    The BK transformation uses a Fenwick tree structure where:
    - Some qubits store occupation numbers directly
    - Others store parities of specific subsets determined by binary tree structure

    :param fock_vector: List of occupation numbers (0 or 1)
    :return: List representing BK encoded state
    """
    n = len(fock_vector)
    bk_vector = [0] * n

    for i in range(n):
        # Calculate the parity for qubit i
        parity = 0

        # Get the set of fermionic modes that contribute to qubit i
        # This follows the Fenwick tree structure
        for j in range(n):
            if should_include_in_parity(i, j):
                parity ^= fock_vector[j]

        bk_vector[i] = parity

    return bk_vector




def bk_to_fock(bk_vector):
    """
    Convert Bravyi-Kitaev encoded vector back to Fock state vector.

    This is the inverse of the BK transformation. We reconstruct the original
    occupation numbers by using the Fenwick tree structure in reverse.

    :param bk_vector: List representing BK encoded state
    :return: List of occupation numbers (0 or 1) in Fock basis
    """
    n = len(bk_vector)
    fock_vector = [0] * n

    # Process from left to right (lowest to highest index)
    for i in range(n):
        # To find fock_vector[i], we need to determine what value
        # would produce the observed bk_vector[i] given the current
        # partial fock_vector

        # Calculate what the parity should be at position i
        # based on the already determined fock values
        current_parity = 0

        # Calculate parity from fermionic modes that contribute to qubit i
        # but exclude the current mode i itself
        for j in range(i):  # Only consider already processed modes
            if should_include_in_parity(i, j):
                current_parity ^= fock_vector[j]

        # The occupation at position i is determined by:
        # bk_vector[i] = current_parity XOR fock_vector[i]
        # Therefore: fock_vector[i] = bk_vector[i] XOR current_parity
        fock_vector[i] = bk_vector[i] ^ current_parity

    return fock_vector


def fock_to_parity(fock_vector):
    parity_vector = []
    for i, occ in enumerate(fock_vector):
        parity_vector.append(int(np.mod(np.sum(fock_vector[:i + 1]), 2)))

    return parity_vector


def parity_to_fock(parity_vector):
    parity_vector = np.array(parity_vector, dtype=int)
    fock_vector = np.zeros_like(parity_vector)

    fock_vector[0] = parity_vector[0]
    fock_vector[1:] = np.bitwise_xor(parity_vector[1:], parity_vector[:-1])

    return fock_vector.tolist()


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


def get_fock_space_vector(vector):
    """
    get vector in Fock space from mapped vector

    :param vector: mapped vector
    :return: vector in Fock space
    """

    if Configuration().mapping == 'bk':
        vector = bk_to_fock(vector)
    if Configuration().mapping == 'pc':
        vector = parity_to_fock(vector)

    return list(vector)


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
    for main_string in sorted(hamiltonian, key=lambda pauliString: pauliString.count("I")):

        # Call findSubStrings to find all the strings in the dictionary that
        # only differ from main_string by identities, and organize them as a
        # dictionary (grouped_operators)
        grouped_operators, checked = find_sub_strings(main_string, hamiltonian, checked)

        # Use the dictionary as a value for the main_string key in the
        # grouped_hamiltonian dictionary
        if len(grouped_operators) > 0:
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

    print('n_orbitals: ', n_orbitals)
    print('n_spin_orbitals: ', n_spin_orbitals)

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

    # print('UCCSD info: ', n_occupied, n_virtual)

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


def get_norm_operator(operator):

    coeff = 0
    for t in operator.terms:
        coeff_t = operator.terms[t]
        coeff += np.conj(coeff_t) * coeff_t

    return np.sqrt(coeff)


def normalize_operator(operator, phase_sign=False):
    """
    normalize an operator

    :param operator:
    :param phase_sign:
    :return: the normalized operator
    """

    coeff = 0
    sum_coeff = 0
    for t in operator.terms:
        coeff_t = operator.terms[t]
        coeff += np.conj(coeff_t) * coeff_t
        sum_coeff += coeff_t

    if phase_sign:
        sign = np.sign(sum_coeff)
    else:
        sign = 1

    if operator.many_body_order() > 0:
        return operator / np.sqrt(coeff)*sign

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

        #if hash_key in cache_dict:
        #    return cache_dict[hash_key]

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
        if isinstance(operator, openfermion.InteractionOperator):
            operator = get_fermion_operator(operator)
            operator = binary_code_transform(operator, parity_code(count_qubits(operator)))
        if isinstance(operator, openfermion.FermionOperator):
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
        if np.linalg.norm(coefficient) > 0:
            total_spins_string += spins_term + ' '
        else:
            total_spins_string += '[' + spins_term + '] '

    return ' {:>18} : {}'.format(operator_string, total_spins_string)


def store_hamiltonian(hamiltonian, file='hamiltonian.npz'):
    """
    store the hamiltonian in a file

    :param hamiltonian: Hamiltonian as InteractionOperator object
    :param file: filename
    :return:
    """

    one_body = hamiltonian.one_body_tensor
    two_body = hamiltonian.two_body_tensor
    constant = hamiltonian.constant

    np.savez(file, one_body, two_body, constant)


def load_hamiltonian(file='hamiltonian.npz'):
    """
    load Hamiltonian from a file

    :param file: filename
    :return: Hamiltonian as InteractionOperator object
    """

    npzfile = np.load(file)

    one_body = npzfile['arr_0']
    two_body = npzfile['arr_1']
    constant = float(npzfile['arr_2'])

    return InteractionOperator(constant, one_body, two_body)


def store_wave_function(coefficients, ansatz, filename='wf.yml'):
    """
    store the hamiltonian in a file

    :param coefficients: list of coefficients
    :param ansatz: list of operators
    :param filename: filename
    :return:
    """

    import yaml

    total_terms = []
    for op in ansatz:
        total_terms.append(op.terms)

    total_terms.append(list(coefficients))

    with open(filename, 'w') as f:
        yaml.dump(total_terms, f)


def load_wave_function(filename='wf.yml', qubit_op=False):
    """
    load wave function from file

    :param filename: file name
    :param qubit_op: True if loading QubitOperators, False if loading FermionOperators
    :return: coefficients, ansatz(list of operators)
    """

    import yaml
    from openfermion import FermionOperator, QubitOperator
    from vqemulti.pool.tools import OperatorList

    op_list = []
    with open(filename, 'r') as f:

        dump = yaml.load(f, yaml.Loader)
        coefficients = dump[-1]
        op_data = dump[:-1]

        for op_dict in op_data:
            op_single = QubitOperator() if qubit_op else FermionOperator()
            for k, v in op_dict.items():
                if qubit_op:
                    op_single += v * QubitOperator(k)
                else:
                    op_single += v * FermionOperator(k)

            op_list.append(op_single)

    return coefficients, OperatorList(op_list)


def reorder_qubits(orbitals_order, hamiltonian, hf_reference_fock, pool=None):
    """
    reorder hamiltonian, reference and pool (optional)

    :param orbitals_order: list of indices of the new orbitals order
    :param hamiltonian: hamiltonian operator
    :param hf_reference_fock: reference in Fock space
    :param pool: pool of operators
    :return: reordered Hamiltonian, reference and pool
    """

    n_qubits = len(hf_reference_fock)

    # define reorder function
    def order_function(mode_idx, num_modes):
        spin = mode_idx % 2
        spatial_idx = mode_idx // 2
        new_spatial_idx = orbitals_order.index(spatial_idx)
        return 2 * new_spatial_idx + spin

    # reorder hamiltonian
    if isinstance(hamiltonian, InteractionOperator):
        hamiltonian = get_fermion_operator(hamiltonian)
        reordered_hamiltonian = reorder(hamiltonian, order_function)
        reordered_hamiltonian = get_interaction_operator(reordered_hamiltonian)
    else:
        reordered_hamiltonian = reorder(hamiltonian, order_function)

    # reorder reference
    reordered_reference = [0] * n_qubits
    for old_idx in range(n_qubits):
        new_idx = order_function(old_idx, n_qubits)
        reordered_reference[new_idx] = hf_reference_fock[old_idx]

    if pool is not None:
        from vqemulti.pool.tools import OperatorList

        # reorder operator pool
        reordered_pool = []
        for op in pool:
            reordered_pool.append(reorder(op, order_function))

        reordered_pool = OperatorList(reordered_pool, normalize=False, antisymmetrize=False, spin_symmetry=False)

        return reordered_hamiltonian, reordered_reference, reordered_pool

    return reordered_hamiltonian, reordered_reference


def build_interaction_matrix(hamiltonian, show_plot=False):
    n_spatial_orbitals = count_qubits(hamiltonian)//2

    # hamiltonian = get_fermion_operator(hamiltonian)
    # get Interaction hamiltonian
    if not isinstance(hamiltonian, InteractionOperator):
        hamiltonian = get_interaction_operator(hamiltonian)

    interaction_matrix = np.zeros((n_spatial_orbitals, n_spatial_orbitals))

    # For spin orbitals indexing: 2*i + spin (0=alpha,1=beta)
    for i in range(n_spatial_orbitals):
        for j in range(n_spatial_orbitals):
            # Sum over spins
            for spin_i in [0,1]:
                for spin_j in [0,1]:
                    p = 2 * i + spin_i
                    q = 2 * j + spin_j

                    # One-body terms h_{pq}
                    interaction_matrix[i,j] += abs(hamiltonian.one_body_tensor[p, q])

                    # Two-body terms h_{pqrs} - simplified summation (example)
                    for r in range(2*n_spatial_orbitals):
                        for s in range(2*n_spatial_orbitals):
                             interaction_matrix[i,j] += abs(hamiltonian.two_body_tensor[p, q, r, s])
    if show_plot:
        # plot interaction matrix
        import matplotlib.pyplot as plt
        plt.imshow(interaction_matrix, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Interaction strength')
        plt.xlabel('Orbital index')
        plt.ylabel('Orbital index')
        plt.title('Interaction matrix heatmap')
        plt.show()

    return interaction_matrix


def get_clustering_order(interaction_matrix, show_plot=False):

    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram

    d_condensed = squareform(np.max(interaction_matrix) - interaction_matrix, checks=False)

    Z = linkage(d_condensed, method='average')
    order = leaves_list(Z)

    print("Optimal clustering order:", order)

    if show_plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,4))
        dendrogram(Z, labels=np.arange(len(interaction_matrix)))
        plt.title('Orbital clustering dendrogram')
        plt.xlabel('Orbital')
        plt.ylabel('Distance')
        plt.show()


def spinorbital_to_spatial_1e(h1_spin):
    n_spin = h1_spin.shape[0]
    n_orb = n_spin // 2
    h1_spatial = np.zeros((n_orb, n_orb))
    for i in range(n_orb):
        for j in range(n_orb):
            # average over spin blocks (alpha-alpha and beta-beta)
            h1_spatial[i, j] = 0.5 * (h1_spin[2*i, 2*j] + h1_spin[2*i+1, 2*j+1])
    return h1_spatial


def spinorbital_to_spatial_2e(two_body_spin, n_orb):
    """
    Convert two-electron integrals from spin-orbitals to spatial orbitals.

    Args:
        two_body_spin (np.ndarray): (2n, 2n, 2n, 2n) tensor of integrals in spin-orbital basis.
        n_orb (int): number of spatial orbitals (n).

    Returns:
        np.ndarray: (n, n, n, n) tensor of integrals in spatial orbital basis.
    """
    two_body_spatial = np.zeros((n_orb, n_orb, n_orb, n_orb))

    for i in range(n_orb):
        for j in range(n_orb):
            for k in range(n_orb):
                for l in range(n_orb):
                    # Sum over spin indices: s_p, s_q, s_r, s_s ∈ {0,1}
                    # delta(sp, sr) and delta(sq, ss) => s_p == s_r and s_q == s_s
                    val = 0.0
                    for sp in (0, 1):
                        for sq in (0, 1):
                            sr = sp
                            ss = sq
                            p = 2 * i + sp
                            q = 2 * j + sq
                            r = 2 * k + sr
                            s = 2 * l + ss
                            val += two_body_spin[p, q, r, s]
                    two_body_spatial[i, j, k, l] = val
    # Optional: divide by 2 to average over spins, depèn de la definició
    # Per la majoria dels formats FCIDUMP es fa així:
    return two_body_spatial / 2


def create_fcidump_file(hamiltonian, filename='FCIDUMP'):
    """
    write FCIDUMP file from an openFermion Hamiltonian using pyscf tools

    :param hamiltonian: InterationOperator
    :param filename: custom file name for FCIDUMP file
    """

    from pyscf.tools import fcidump

    # 1-e integrals
    h1 = hamiltonian.one_body_tensor
    h1 = spinorbital_to_spatial_1e(h1)

    norb = h1.shape[0]

    # 2-e integrals
    inv = np.argsort((0, 2, 3, 1))
    # Hamiltonian absorbs 1/2 into the coefficients. To recover 2e integrals should be multiplied by 2
    h2 = hamiltonian.two_body_tensor.transpose(inv)*2
    h2 = spinorbital_to_spatial_2e(h2, norb)
    norb = h1.shape[0]

    # effective nuclear potential (nuc-nuc + effective core electrons)
    nuc = hamiltonian.constant

    fcidump.from_integrals(
        filename,
        h1,
        h2,
        norb,
        nelec=4,  # total electrons
        ms=0,  # spin multiplicity
        # orbsym=None,
        # tol=1e-8,
        nuc=nuc
    )


def create_fcidump_file_fermion(hamiltonian, configuration_list, filename='FCIDUMP'):

    nelec = np.sum(configuration_list[0])
    n_orb = len(configuration_list[0])//2
    ms2 = 0

    def all_even_indices(index_list):
        return all(i % 2 == 0 for i in index_list)

    with open(filename, "w") as f:
        f.write(f" &FCI NORB={n_orb}, NELEC={nelec}, MS2={ms2}, \nISYM=1, \n &END\n")
        for key in hamiltonian:
            indices = [ind[0] for ind in key]

            if all_even_indices(indices):
                indices = [i//2+1 for i in indices]
            else:
                continue

            if len(indices) == 4:
                value = 0
                for sp in (0, 1):
                    for sq in (0, 1):
                        sr = sp
                        ss = sq
                        p = indices[0] + sp
                        q = indices[1] + sq
                        r = indices[2] + sr
                        s = indices[3] + ss

                        key = ((p, 1), (q, 1), (r, 0), (s, 0))

                        value += hamiltonian[key]

                # Hamiltonian absorbs 1/2 into the coefficients. To recover 2e integrals should be multiplied by 2
                # f.write('{:25.20e} '.format(hamiltonian[key]*2) + '{} {} {} {}\n'.format(*indices))
                f.write('{:25.20e} '.format(value*2) + '{} {} {} {}\n'.format(*indices))

            if len(indices) == 2:
                f.write('{:25.20e} '.format(hamiltonian[key]) + '{} {}  0  0\n'.format(*indices))

            if len(indices) == 0:
                f.write('{:25.20e} '.format(hamiltonian[key]) + '0  0  0  0\n')


def create_input_file_dice(configuration_list, davidsontol=1e-8, dE=1e-10, filename='input.dat'):
    n_orbs = len(configuration_list[0]) // 2
    n_electrons = np.sum(configuration_list[0])

    def get_num_conf(conf):
        final_conf = []
        for i, c in enumerate(conf):
            if c == 1:
                final_conf.append(str(i))
        return final_conf

    with open(filename, 'w') as f:
        #f.write(f"orbitals FCIDUMP\n")

        # diagonalization parameters
        f.write(f"davidsontol {davidsontol}\n")
        f.write(f"dE {dE}\n")
        f.write(f"nroots 1\n")
        f.write(f"noio\n")  # no intermediate files

        # active orbital range
        f.write(f"ncore 0\n")
        f.write(f"nact {n_orbs}\n")

        # schedule
        f.write(f"schedule\n")
        f.write(f"0 1e20\n")
        f.write(f"end\n")

        # perturbation
        f.write(f"maxiter 1\n")
        f.write(f"epsilon2 1e20\n")
        f.write(f"nPTiter 0\n")
        f.write(f"sampleN 200\n")

        # configurations
        # f.write(f"sampleN 200\n")
        f.write(f"nocc {n_electrons}\n")
        for configuration in configuration_list:
            # f.write(f"0 1 2 3\n")  # TODO arreglar aixo
            f.write(' '.join(get_num_conf(configuration)) + '\n')
        f.write(f"end\n")


def get_selected_ci_energy_dice(configuration_list, hamiltonian):
    import tempfile
    from subprocess import Popen, PIPE

    dice_path = '/Users/abel/bin/Dice'
    data_dir = 'temp_dir/'
    mpirun_options = None

    # create input files
    create_fcidump_file(hamiltonian, filename=data_dir + 'FCIDUMP')
    create_input_file_dice(configuration_list, filename=data_dir + 'input.dat')

    # run Dice
    if mpirun_options:
        if isinstance(mpirun_options, str):
            mpirun_options = mpirun_options.split()
        dice_call = ["mpirun"] + list(mpirun_options) + [dice_path]
    else:
        dice_call = ["mpirun", dice_path]

    qchem_process = Popen(dice_call, stdout=PIPE, stdin=PIPE, stderr=PIPE, cwd=data_dir)
    (output, err) = qchem_process.communicate()
    qchem_process.wait()
    output = output.decode(errors='ignore')
    err = err.decode()

    print(err)
    print(output)

    enum = output.find('Variational calculation result')
    #print(output[enum: enum+500].split()[7])
    sci_energy = float(output[enum: enum+500].split()[7])

    # print('final parsed energy: ', sci_energy)
    return sci_energy


def get_selected_ci_energy_qiskit_alternative(configuration_list, hamiltonian):

    from qiskit_addon_sqd.fermion import solve_fermion

    up_list = []
    down_list = []
    for fock_vector in configuration_list:
        # print(fock_vector)
        bitstring = ''.join(str(bit) for bit in fock_vector[::-1])

        up_str = ''.join(
            '0' if i % 2 == 0 else bit
            for i, bit in enumerate(bitstring)
        )

        down_str = ''.join(
            bit if i % 2 == 0 else '0'
            for i, bit in enumerate(bitstring)
        )

        # if up_str.count('1') == alpha_electrons and down_str.count('1') == beta_electrons:
        up_list.append(int(up_str, 2))
        down_list.append(int(down_str, 2))

    up = np.array(up_list, dtype=np.uint32)
    down = np.array(down_list, dtype=np.uint32)

    inv = np.argsort((0, 2, 3, 1))
    # Hamiltonian absorbs 1/2 into the coefficients. To recover 2e integrals should be multiplied by 2
    two_body_tensor_restored = hamiltonian.two_body_tensor.transpose(inv) * 2

    if Configuration().verbose:
        print('Number of SQD configurations:', len(up))

    energy_sci, coeffs_sci, avg_occs, spin = solve_fermion((up, down),
                                                           hamiltonian.one_body_tensor,
                                                           two_body_tensor_restored,
                                                           open_shell=True,
                                                           # spin_sq=0,
                                                           )

    return energy_sci + hamiltonian.constant


def get_selected_ci_energy_qiskit(configuration_list, hamiltonian):

    from qiskit_addon_sqd.fermion import solve_fermion

    # 1-e integrals
    h1 = hamiltonian.one_body_tensor
    h1 = spinorbital_to_spatial_1e(h1)

    norb = h1.shape[0]

    # 2-e integrals
    inv = np.argsort((0, 2, 3, 1))
    # Hamiltonian absorbs 1/2 into the coefficients. To recover 2e integrals should be multiplied by 2
    h2 = hamiltonian.two_body_tensor.transpose(inv)*2
    h2 = spinorbital_to_spatial_2e(h2, norb)

    # setup configurations
    up_list = []
    down_list = []
    for fock_vector in configuration_list:
        bitstring = ''.join(str(bit) for bit in fock_vector[::-1])
        up_list.append(int(bitstring[::2], 2))
        down_list.append(int(bitstring[1::2], 2))

    up = np.array(up_list, dtype=np.uint32)
    down = np.array(down_list, dtype=np.uint32)

    if Configuration().verbose:
        print('Number of SQD configurations:', len(up))

    energy_sci, coeffs_sci, avg_occs, spin = solve_fermion((up, down),
                                                           h1,
                                                           h2,
                                                           open_shell=False,
                                                           # spin_sq=0,
                                                           )

    return energy_sci + hamiltonian.constant


def get_dmrg_energy(hamiltonian,
                    n_electrons,
                    symmetry=None,
                    spin=0,
                    maxM=500,
                    schedule='default',
                    maxiter=200):

    import tempfile
    from subprocess import Popen, PIPE

    dice_path = '/Users/abel/bin/block2main'
    data_dir = 'temp_dir/'
    mpirun_options = None

    # create input files
    create_fcidump_file(hamiltonian, filename=data_dir + 'FCIDUMP')

    with open(data_dir + 'dmrg.conf', 'w') as f:

        if symmetry is not None:
            f.write(f"sym d2h {symmetry}\n")

        f.write(f"orbitals FCIDUMP\n")
        f.write(f"nelec {n_electrons}\n")
        f.write(f"spin {spin}\n")

        # f.write(f"irrep 1\n")
        f.write(f"hf_occ integral\n")

        # schedule
        f.write(f"schedule {schedule}\n")
        f.write(f"maxM {maxM}\n")

        # perturbation
        f.write(f"maxiter {maxiter}\n")
        f.write(f"onepdm\n")
        f.write(f"irrep_reorder\n")

    # run block2
    if mpirun_options:
        if isinstance(mpirun_options, str):
            mpirun_options = mpirun_options.split()
        dice_call = ["mpirun"] + list(mpirun_options) + [dice_path, 'dmrg.conf']
    else:
        dice_call = [dice_path, 'dmrg.conf']

    qchem_process = Popen(dice_call, stdout=PIPE, stdin=PIPE, stderr=PIPE, cwd=data_dir)
    (output, err) = qchem_process.communicate()
    qchem_process.wait()
    output = output.decode(errors='ignore')
    err = err.decode()

    # print(err)
    # print(output)

    enum = output.find('Final canonical form')
    # print(output[enum: enum+500].split())
    sci_energy = float(output[enum: enum+500].split()[9])

    # print('final parsed energy: ', sci_energy)
    return sci_energy
