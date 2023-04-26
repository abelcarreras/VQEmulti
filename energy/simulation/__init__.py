from utils import convert_hamiltonian, group_hamiltonian
from openfermion.utils import count_qubits
from energy.simulation.trotter import get_preparation_gates_trotter, trotterizeOperator
from energy.simulation.tools_cirq import measure_expectation, get_exact_state_evaluation, get_preparation_gates
from energy.simulation.tools_penny import measure_expectation, get_exact_state_evaluation, get_preparation_gates


def get_sampled_energy(qubitHamiltonian, shots, statePreparationGates):
    """
    Obtains the expectation value in a state by sampling (using the CIRQ
    simulator).

    Arguments:
      groupedHamiltonian (dict): the Hamiltonian, grouped by terms that only differ
        by identities (as done by groupHamiltonian).
      shots (int): the number of circuit repetitions to be used.
      statePreparationGates (list): a list of CIRQ gates that prepare the state.
      qubits (list): a list of cirq.LineQubit to apply the gates on.

    Returns:
      energy (float): the energy (with sampling noise).
    """

    qubitNumber = count_qubits(qubitHamiltonian)

    # print(qubitHamiltonian)
    formattedHamiltonian = convert_hamiltonian(qubitHamiltonian)
    # print(formattedHamiltonian)
    groupedHamiltonian = group_hamiltonian(formattedHamiltonian)
    # for group in groupedHamiltonian:
    #    print(group, groupedHamiltonian[group])

    # Obtain the experimental expectation value for each Pauli string by
    # calling the measureExpectation function, and perform the necessary weighed
    # sum to obtain the energy expectation value

    energy = 0
    for main_string, sub_hamiltonian in groupedHamiltonian.items():
        expectation_value = measure_expectation(main_string,
                                                sub_hamiltonian,
                                                shots,
                                                statePreparationGates,
                                                qubitNumber)
        energy += expectation_value

    assert energy.imag < 1e-5

    return energy.real


def simulate_vqe_energy(coefficients, ansatz, hf_reference_fock, qubitHamiltonian, shots,
                        trotter=True, trotter_steps=1, sample=True):
    """
    Obtains the energy of the state prepared by applying an ansatz (of the
    type of the Adapt VQE protocol) to a reference state, using the CIRQ simulator.

    Arguments:
      coefficients ([float]): the list of coefficients of the ansatz operators.
      ansatz ([openfermion.FermionOperator]): the list of ansatz operators
        (fermionic ladder operators, pre exponentiation). The index of the
        operators in the list should be in accordance with the index of the
        respective coefficients in coefVect.
      shots (int): the number of circuit repetitions to use to get the
        expectation value. The precision of the estimate goes like 1/sqrt(shots)
      trotter (bool): whether to trotterize the ansatz. If False, the corresponding
        matrix will be applied directly using a cirq.MatrixGate.
      trotter_steps (int): the number of trotter steps to be used in the trotterization.
      sample (bool): whether to sample from the circuit. If False, the full
        wavevector will be simulated, and the final energy will be calculated
        analytically.

      Returns:
        energy (float): the expectation value of the Hamiltonian in the state
          prepared by applying the ansatz to the reference state, with trotter
          error (if trotter = True) and sampling noise (if sample = True)
    """

    # Format and group the Hamiltonian, so as to save measurements by using
    # the same data for Pauli strings that only differ by identities
    # formattedHamiltonian = convertHamiltonian(qubitHamiltonian)
    # groupedHamiltonian = groupHamiltonian(formattedHamiltonian)

    # Count the number of qubits the Hamiltonian acts on
    n_qubits = count_qubits(qubitHamiltonian)

    # If the trotter flag isn't set, use matrix exponentiation to get the exact
    # matrix representing the action of the ansatz, and apply it directly on the
    if trotter:
        statePreparationGates = get_preparation_gates_trotter(coefficients,
                                                              ansatz,
                                                              trotter_steps,
                                                              hf_reference_fock,
                                                              n_qubits)
    else:
        statePreparationGates = get_preparation_gates(coefficients,
                                                      ansatz,
                                                      hf_reference_fock,
                                                      n_qubits)


    # from energy.simulation.tools import get_circuit_depth
    # circuit_depth = get_circuit_depth(statePreparationGates)
    # print('circuit_depth', circuit_depth)

    if sample:
        # Obtain the energy expectation value by sampling from the circuit using
        # the CIRQ simulator
        energy = get_sampled_energy(qubitHamiltonian, shots, statePreparationGates)
    else:
        # Calculate the exact energy in this state
        energy = get_exact_state_evaluation(qubitHamiltonian, statePreparationGates)
        # print('Exact energy ener (no sample): ', energy)

    return energy
