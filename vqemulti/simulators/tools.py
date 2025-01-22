import numpy as np

previous_row = []

def set_previous_row(row):
    global previous_row
    previous_row = row


def get_cnot_inversion_mat(ordered_terms, n_qubits):
    """
    get the matrix that tells if a CNOT gates are inverted or not in staircase algorithm
    for a given pair of qubits and exponential operator

    :param ordered_terms: openfermion qubit operator list (as for each exponential operator)
    :param n_qubits: number of qubits in the circuit
    :return: matrix of n_operators x (n_qubits - 1) containing whether CNOT should be inverted or not
    """

    #print(ordered_terms)
    n_row = len(ordered_terms)
    n_col = n_qubits
    # print(n_row, n_col)

    gate_matrix = np.chararray((n_row, n_col))
    gate_matrix[:] = 'I'

    for i, row in enumerate(ordered_terms):
        for element in row:
            gate_matrix[i, element[0]] = element[1]

    # print('previous: ', previous_row)
    # print(gate_matrix)

    mat = np.zeros((n_row, n_col), dtype=int)

    for pauli_string, m_row in zip(gate_matrix, mat):
        for i, gate in enumerate(pauli_string):
            if gate == b'X':
                m_row[i] = -1
            elif gate == b'Z':
                m_row[i] = +1
            elif gate == b'Y' or gate == b'I':
                count = 0
                max_count = 1 if i == 0 or i == n_col-1 else 2
                if i > 0:
                    if pauli_string[i-1] == b'X':
                        count += 1
                if i < n_col-1:
                    if pauli_string[i+1] == b'X':
                        count += 1
                if count > max_count//2:
                    m_row[i] = -1
            else:
                m_row[i] = 0

    if len(previous_row) == n_col:
        mat = np.vstack((previous_row, mat))

    control_val = np.sum(np.abs(mat))
    randomize_next = False
    while np.sum(np.abs(mat)) < n_col * len(mat):
        # print(np.sum(np.abs(mat)), n_col * len(mat))
        for i, m_row in enumerate(mat):
            for j, element in enumerate(m_row):
                #if element == 0:
                    n_sum = 0 # mat[i, j]
                    for k1, k2 in zip([1, -1, 0, 0], [0, 0, 1, -1]):
                        # print('try: ', k1, k2)
                        try:
                            n_sum += mat[i + k1, j + k2]

                        except IndexError:
                            pass

                    # print('     counts: ', n_count_plus, n_count_minus)
                    if n_sum > 0:
                        mat[i, j] = 1
                    elif n_sum < 0:
                        mat[i, j] = -1
                    elif n_sum == 0 and randomize_next:
                        mat[i, j] = np.random.choice([1, -1])
                        randomize_next = False

        if control_val == np.sum(np.abs(mat)):
            randomize_next = True

        control_val = np.sum(np.abs(mat))

    # print('final:', mat)

    mat = mat[-n_row:]
    set_previous_row(mat[-1])
    #print('end:', mat)
    #print('\n')

    # exit()
    bool_mat = []
    for row in mat:
        bool_row = []
        for j in range(len(row)-1):
            if row[j] == row[j+1]:
                bool_row.append(bool(row[j]-1))
            else:
                bool_row.append(True)
        bool_mat.append(bool_row)

    return bool_mat
