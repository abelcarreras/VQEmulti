from vqemulti.simulators import SimulatorBase
import numpy as np
import pennylane as qml


def trotter_step(qubit_operator, time):
    """
    Creates the circuit for applying e^(-j*operator*time), simulating the time
    evolution of a state under the Hamiltonian 'operator'.

    :param qubit_operator: qubit operator
    :param time: the evolution time
    :return: trotter_gates
    """

    # Initialize list of gates
    trotter_gates = []

    # Order the terms the same way as done by OpenFermion's
    # trotter_operator_grouping function (sorted keys) for consistency.
    ordered_terms = sorted(list(qubit_operator.terms.keys()))

    # Add to trotter_gates the gates necessary to simulate each Pauli string,
    # going through them by the defined order
    for pauliString in ordered_terms:

        # Get real part of the coefficient (the immaginary one can't be simulated,
        # as the exponent would be real and the operation would not be unitary).
        # Multiply by time to get the full multiplier of the Pauli string.
        coefficient = float(np.real(qubit_operator.terms[pauliString])) * time

        # Keep track of the qubit indices involved in this particular Pauli string.
        # It's necessary so as to know which are included in the sequence of CNOTs
        # that compute the parity
        involved_qubits = []

        # Perform necessary basis rotations
        for pauli in pauliString:

            # Get the index of the qubit this Pauli operator acts on
            qubit_index = pauli[0]
            involved_qubits.append(qubit_index)

            # Get the Pauli operator identifier (X,Y or Z)
            pauli_operator = pauli[1]

            if pauli_operator == "X":
                # Rotate to X basis
                trotter_gates.append(qml.Hadamard(wires=qubit_index))

            if pauli_operator == "Y":
                # Rotate to Y Basis
                trotter_gates.append(qml.RX(np.pi / 2, wires=qubit_index))

        # Compute parity and store the result on the last involved qubit
        for i in range(len(involved_qubits) - 1):
            control = involved_qubits[i]
            target = involved_qubits[i + 1]
            trotter_gates.append(qml.CNOT(wires=[control, target]))

        # Apply e^(-i*Z*coefficient) = Rz(coefficient*2) to the last involved qubit
        last_qubit = max(involved_qubits)
        trotter_gates.append(qml.RZ((2 * coefficient), wires=last_qubit))

        # Uncompute parity
        for i in range(len(involved_qubits) - 2, -1, -1):
            control = involved_qubits[i]
            target = involved_qubits[i + 1]
            trotter_gates.append(qml.CNOT(wires=[control, target]))

        # Undo basis rotations
        for pauli in pauliString:

            # Get the index of the qubit this Pauli operator acts on
            qubit_index = pauli[0]

            # Get the Pauli operator identifier (X,Y or Z)
            pauli_operator = pauli[1]

            if pauli_operator == "X":
                # Rotate to Z basis from X basis
                trotter_gates.append(qml.Hadamard(qubit_index))

            if pauli_operator == "Y":
                # Rotate to Z basis from Y Basis
                trotter_gates.append(qml.RX(-np.pi / 2, wires = qubit_index))

    return trotter_gates


class PennylaneSimulator(SimulatorBase):

    def _get_state_vector(self, state_preparation_gates, n_qubits):

        # Initialize circuit.
        dev_unique_wires = qml.device('default.qubit', wires=[i for i in range(n_qubits)])

        # add gates to circuit
        def circuit_function():
            for gate in state_preparation_gates:
                qml.apply(gate)
            return qml.state()

        # create and run circuit
        circuit = qml.QNode(circuit_function, dev_unique_wires, analytic=None)

        return circuit()

    def _get_matrix_operator_gates(self, hf_reference_fock, matrix):

        # Initialize qubits
        n_qubits = len(hf_reference_fock)

        # Add gates for HF reference
        state_preparation_gates = self._build_reference_gates(hf_reference_fock)

        # Append the ansatz directly as a matrix
        state_preparation_gates.append(qml.QubitUnitary(matrix.toarray(), wires=list(range(n_qubits))))

        return state_preparation_gates

    def _measure_expectation(self, main_string, sub_hamiltonian, state_preparation_gates, n_qubits):
        """
        Measures the expectation value of a sub_hamiltonian (pauli string) using the Quiskit simulator.
        By construction, all the expectation values of the strings in subHamiltonian can be
        obtained from the same measurement array. This reduces quantum computer simulations

        :param main_string: hamiltonian base Pauli string ex: (XXYY)
        :param sub_hamiltonian: partial hamiltonian interactions ex: {'0000': -0.4114, '1111': -0.0222}
        :param state_preparation_gates: list of gates in simulation library format that represents the state
        :param n_qubits: number of qubits
        :return:
        """

        # Initialize circuit.
        dev_unique_wires = qml.device('default.qubit', wires=[i for i in range(n_qubits)], shots=self._shots)

        # Build circuit from preparation gates
        @qml.qnode(dev_unique_wires)
        def circuit():
            # apply preparation gates
            for gate in state_preparation_gates:
                qml.apply(gate)

            # apply operators to measure each qubit in the basis given by main strings
            for i, op in enumerate(main_string):
                if op == "X":
                    qml.Hadamard(wires=[i])

                elif op == "Y":
                    qml.RX(np.pi / 2, wires=[i])

            # sample measurements in PauliZ
            # return [qml.sample(qml.PauliZ(wires=k)) for k in range(n_qubits)]
            return qml.counts()

        # draw circuit
        # print(qml.draw(circuit)())
        self._get_circuit_stat_data(circuit)

        def str_to_bit(string):
            return 1 if string == '1' else -1

        # Get function return from measurements in Z according to sub_hamiltonian
        total_expectation_value = 0
        for measure_string, counts in circuit().items():
            for sub_string, coefficient in sub_hamiltonian.items():

                prod_function = 1
                for i, measure_z in enumerate([str_to_bit(k) for k in measure_string[::-1]]):
                    if main_string[i] != "I":
                        prod_function *= measure_z ** int(sub_string[i])

                total_expectation_value += prod_function * coefficient * counts/self._shots

        return total_expectation_value

    def _build_reference_gates(self, hf_reference_fock):
        """
        Create the gates for preparing the Hartree Fock ground state, that serves
        as a reference state the ansatz

        :param hf_reference_fock: HF reference in fock space
        :return: reference gates
        """

        reference_gates = []
        for i, occ in enumerate(hf_reference_fock):
            if bool(occ):
                reference_gates.append(qml.PauliX(wires=[i]))
            else:
                reference_gates.append(qml.Identity(wires=[i]))
        return reference_gates
        # return [qml.PauliX(wires=[i]) for i, occ in enumerate(hf_reference_fock) if bool(occ)]

    def _trotterize_operator(self, qubit_operator, time, trotter_steps, n_qubits):
        """
        Creates the circuit for applying e^(-j*operator*time), simulating the time
        evolution of a state under the Hamiltonian 'operator', with the given
        number of steps.
        Increasing the number of steps increases precision (unless the terms in the
        operator commute, in which case steps = 1 is already exact).
        For the same precision, a greater time requires a greater step number
        (again, unless the terms commute)

        :param qubit_operator: qubit operator
        :param time: the evolution time
        :param trotter_steps: number of trotter steps
        :return: the number of trotter steps to split the time evolution into
        """

        # Divide time into steps and apply the evolution operator the necessary
        # number of times
        trotter_gates = []
        for step in range(1, trotter_steps + 1):
            trotter_gates += trotter_step(qubit_operator, time / trotter_steps)

        return trotter_gates

    def _get_circuit_stat_data(self, circuit):

        specs_func = qml.specs(circuit)

        # depth
        self._circuit_count.append(specs_func()['depth']-1)

        # gates
        gates_dict = specs_func()['gate_types']
        for k, v in gates_dict.items():
            self._circuit_gates[k] += v


    def get_circuit_info(self, coefficients, ansatz, hf_reference_fock):

        ansatz_qubit = ansatz.transform_to_scaled_qubit(coefficients)

        state_preparation_gates = self.get_preparation_gates(ansatz_qubit, hf_reference_fock)

        # Initialize circuit.
        n_qubits = len(hf_reference_fock)
        dev_unique_wires = qml.device('default.qubit', wires=[i for i in range(n_qubits)])

        # add gates to circuit
        def circuit_function():
            for gate in state_preparation_gates:
                qml.apply(gate)
            return qml.state()

        # create and run circuit
        circuit = qml.QNode(circuit_function, dev_unique_wires, analytic=None)

        specs_func = qml.specs(circuit)
        return specs_func()


if __name__ == '__main__':
    simulator = PennylaneSimulator(trotter=True,
                                   trotter_steps=1,
                                   test_only=True,
                                   shots=100)

    print(simulator.get_preparation_gates)