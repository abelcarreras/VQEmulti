from vqemulti.utils import get_sparse_ket_from_fock, get_sparse_operator
from vqemulti.genetic_tools.probs_cheating import generate_new_operator, delete_an_operator, generate_reduced_pool
from openfermion.utils import count_qubits
import numpy as np
import scipy
from copy import deepcopy
import math
import random



def add_new_excitation_first_it(hf_reference_fock, hamiltonian, ansatz, coefficients, operators_pool, einstein_index,cheat_param, selected_already):

    #random_operator = generate_new_operator(operators_pool, einstein_index, cheat_param)

    list_possible = []
    for i in range(len(operators_pool)):
        list_possible.append(i)
    for i in range(len(selected_already)):
        index = list_possible.index(selected_already[i])
        list_possible.pop(index)

    #random_operator = np.random.randint(0, len(list_possible))
    random_operator = generate_reduced_pool(operators_pool, selected_already, einstein_index, cheat_param)
    #random_operator = list_possible[random_operator]



    einstein_index_game = deepcopy(einstein_index)

    if len(ansatz) == 0:
        new_ansatz = []
    else:
        new_ansatz = deepcopy(ansatz)

    new_ansatz.append(operators_pool[random_operator])

    if len(coefficients) == 0:
        new_coefficients = []
        new_coefficients.append(0.05)
    else:
        new_coefficients = deepcopy(coefficients)
        random_number = np.random.rand()
        if random_number < 0.5:
            new_coeff = 0.05
            #print('POSITIVE')
            #new_coeff = coefficients[-1] - math.pi * random_number * coefficients[-1]
            #new_coeff = math.pi * random_number
            #new_coeff = coefficients[-1] - 0.2 * coefficients[-1]
            new_coefficients.append(new_coeff)
        else:
            new_coeff = -0.05
            #print('NEGATIVE')
            #new_coeff = coefficients[-1] - math.pi * random_number * coefficients[-1]
            #new_coeff = math.pi * random_number
            #new_coeff = coefficients[-1] + 0.2 * coefficients[-1]
            new_coefficients.append(new_coeff)


    fitness = fitness_evaluation(new_coefficients, new_ansatz, hf_reference_fock, hamiltonian)
    einstein_index_game.append(random_operator)
    selected = random_operator
    #print('Added', random_operator, 'coefficients', new_coefficients,'indeces',einstein_index_game, 'fitness', fitness)
    return new_ansatz, new_coefficients, fitness, einstein_index_game, selected


def add_new_excitation_double(hf_reference_fock, hamiltonian, ansatz, coefficients, operators_pool, einstein_index,cheat_param, selected_already):

    list_possible = []
    for i in range(len(operators_pool)):
        list_possible.append(i)
    for i in range(len(selected_already)):
        index = list_possible.index(selected_already[i])
        list_possible.pop(index)

    #random_operator = np.random.randint(0, len(list_possible))
    random_operator = generate_reduced_pool(operators_pool, selected_already, einstein_index, cheat_param)
    #random_operator = list_possible[random_operator]



    einstein_index_game_1 = deepcopy(einstein_index)
    einstein_index_game_2 = deepcopy(einstein_index)

    if len(ansatz) == 0:
        new_ansatz_1 = []
        new_ansatz_2 = []
    else:
        new_ansatz_1 = deepcopy(ansatz)
        new_ansatz_2 = deepcopy(ansatz)

    new_ansatz_1.append(operators_pool[random_operator])
    new_ansatz_2.append(operators_pool[random_operator])

    if len(coefficients) == 0:
        new_coefficients_1 = []
        new_coefficients_2 = []
        new_coefficients_1.append(0.05)
        new_coefficients_2.append(-0.05)
    else:
        new_coefficients_1 = deepcopy(coefficients)
        new_coefficients_2 = deepcopy(coefficients)
        new_coefficients_1.append(0.05)
        new_coefficients_2.append(-0.05)


    fitness_1 = fitness_evaluation(new_coefficients_1, new_ansatz_1, hf_reference_fock, hamiltonian)
    einstein_index_game_1.append(random_operator)
    fitness_2 = fitness_evaluation(new_coefficients_2, new_ansatz_2, hf_reference_fock, hamiltonian)
    einstein_index_game_2.append(random_operator)
    selected = random_operator
    #print('Added', random_operator, 'coefficients', new_coefficients,'indeces',einstein_index_game, 'fitness', fitness)
    return new_ansatz_1, new_coefficients_1, fitness_1, einstein_index_game_1, selected, new_ansatz_2, new_coefficients_2, fitness_2, einstein_index_game_2







def add_new_excitation(hf_reference_fock, hamiltonian, ansatz, coefficients, operators_pool, einstein_index,cheat_param):

    random_operator = generate_new_operator(operators_pool, einstein_index, cheat_param)
    einstein_index_game = deepcopy(einstein_index)

    if len(ansatz) == 0:
        new_ansatz = []
    else:
        new_ansatz = deepcopy(ansatz)

    new_ansatz.append(operators_pool[random_operator])

    if len(coefficients) == 0:
        new_coefficients = []
        new_coefficients.append(0.05)
    else:
        new_coefficients = deepcopy(coefficients)
        random_number = np.random.rand()
        if random_number < 0.5:
            new_coeff = 0.05
            new_coefficients.append(new_coeff)
        else:
            new_coeff = -0.05
            new_coefficients.append(new_coeff)


    fitness = fitness_evaluation(new_coefficients, new_ansatz, hf_reference_fock, hamiltonian)
    einstein_index_game.append(random_operator)
    #print('Added', random_operator, 'coefficients', new_coefficients,'indeces',einstein_index_game, 'fitness', fitness)
    return new_ansatz, new_coefficients, fitness, einstein_index_game




def delete_excitation(hf_reference_fock, hamiltonian, ansatz, coefficients, einstein_index):

    if len(ansatz) == 0 or len(ansatz) == 1:
        fitness = fitness_evaluation(coefficients, ansatz, hf_reference_fock, hamiltonian)
        return ansatz, coefficients, fitness, einstein_index

    einstein_index_game = deepcopy(einstein_index)
    random_operator = delete_an_operator(einstein_index, coefficients)
    #random_operator = np.random.randint(0, len(ansatz))
    new_ansatz = deepcopy(ansatz)
    new_ansatz.pop(random_operator)
    new_coefficients = deepcopy(coefficients)
    new_coefficients.pop(random_operator)

    fitness = fitness_evaluation(new_coefficients, new_ansatz, hf_reference_fock, hamiltonian)
    einstein_index_game.pop(random_operator)
    #print('Deleted', random_operator, 'coefficients', new_coefficients,'indeces',einstein_index_game,'fitness', fitness)

    return new_ansatz, new_coefficients, fitness, einstein_index_game


def change_excitation(hf_reference_fock, hamiltonian, ansatz, coefficients, operators_pool, einstein_index, cheat_param):

    if len(ansatz) == 0:
        fitness = fitness_evaluation(coefficients, ansatz, hf_reference_fock, hamiltonian)
        return ansatz, coefficients, fitness, einstein_index

    einstein_index_game = deepcopy(einstein_index)
    #random_operator_change = np.random.randint(0, len(ansatz))
    random_operator_change = delete_an_operator(einstein_index, coefficients)
    new_ansatz = deepcopy(ansatz)
    new_coefficients = deepcopy(coefficients)
    #random_operator_add = np.random.randint(0, len(operators_pool))
    random_operator_add = generate_new_operator(operators_pool, einstein_index, cheat_param)
    new_ansatz[random_operator_change] = operators_pool[random_operator_add]
    einstein_index_game[random_operator_change] = random_operator_add
    random_number = np.random.rand()
    if random_number < 0.5:
        new_coefficients[random_operator_change] = 0.05
    else:
        new_coefficients[random_operator_change] = -0.05


    fitness = fitness_evaluation(coefficients, new_ansatz, hf_reference_fock, hamiltonian)
    #print('Deleated', random_operator_change, 'coefficients', coefficients,'indeces final',einstein_index_game,'fitness',fitness)

    return new_ansatz, coefficients, fitness, einstein_index_game












def fitness_evaluation(coefficients, ansatz, hf_reference_fock, hamiltonian):
    """
    Calculates the energy of the state prepared by applying an ansatz (of the
    type of the Adapt VQE protocol) to a reference state.
    Applies Trotter approach (n=1) -> psi> = prod(e^(theta_i*T_i)) |0>

    :param coefficients: the list of coefficients of the ansatz operators
    :param ansatz: ansatz expressed in qubit/fermion operators
    :param hf_reference_fock: HF reference in Fock space vector
    :param hamiltonian: Hamiltonian in FermionOperator/InteractionOperator
    :return: exact energy
    """

    # Transform Hamiltonian to matrix representation
    sparse_hamiltonian = get_sparse_operator(hamiltonian)

    # Find the number of qubits of the system (2**n_qubit = dimension)
    n_qubit = count_qubits(hamiltonian)

    # Transform reference vector into a Compressed Sparse Column matrix
    ket = get_sparse_ket_from_fock(hf_reference_fock)

    # Apply e ** (coefficient * operator) to the state (ket) for each operator in
    # the ansatz, following the order of the list
    for coefficient, operator in zip(coefficients, ansatz):

        # Get the operator matrix representation of the operator
        sparse_operator = coefficient * get_sparse_operator(operator, n_qubit)

        # Exponentiate the operator and update ket t
        ket = scipy.sparse.linalg.expm_multiply(sparse_operator, ket)

    # Get the corresponding bra and calculate the energy: |<bra| H |ket>|
    bra = ket.transpose().conj()
    energy = np.sum(bra * sparse_hamiltonian * ket).real

    return energy


