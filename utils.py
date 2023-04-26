from openfermion.utils import count_qubits
from openfermion.ops.representations import InteractionOperator
import numpy as np
import scipy


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


def get_sparse_ket_from_fock(fock_vector):
    """
    Transforms a state represented in Fock space to sparse vector.
    The Jordan-Wigner Transform is used.

    :param fock_vector: a list representing the Fock vector
    :return: the corresponding sparse vector
    """

    state_vector = [1]

    # Iterate through the ket, calculating the tensor product of the qubit states
    for i in fock_vector:
        qubit_vector = [not i, i]
        state_vector = np.kron(state_vector, qubit_vector)

    return scipy.sparse.csc_matrix(state_vector, dtype=complex).transpose()


def get_hf_reference_in_fock_space(electron_number, qubit_number):
    """
    Get the Hartree Fock reference in Fock space vector
    The order is: [orbital_1-alpha, orbital_1-beta, orbital_2-alpha, orbital_2-beta, orbital_3-alpha.. ]

    :param electron_number: number of electrons
    :param qubit_number: the number of qubits necessary to represent the molecule
    :return: the vector in the Fock space
    """

    # This considers occupied the lower energy orbitals
    hf_reference = np.zeros(qubit_number, dtype=int)
    for i in range(electron_number):
        hf_reference[i] = 1

    return hf_reference.tolist()


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
    Organizes a Hamiltonian into groups where strings only differ from
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


def generate_reduced_hamiltonian(hamiltonian, n_orbitals):
    """
    get truncated hamiltonian given a number of orbitals

    :param hamiltonian: hamiltonian in fermionic operators
    :param n_orbitals: number of orbitals to keep
    :return: truncated hamiltonian in fermionic operators
    """

    n_spin_orbitals = n_orbitals * 2

    reduced_one = hamiltonian.one_body_tensor[:n_spin_orbitals, :n_spin_orbitals]
    reduced_two = hamiltonian.two_body_tensor[:n_spin_orbitals, :n_spin_orbitals, :n_spin_orbitals, :n_spin_orbitals]

    return InteractionOperator(hamiltonian.constant, reduced_one, reduced_two)
