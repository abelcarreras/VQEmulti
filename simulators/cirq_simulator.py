from simulators import SimulatorBase
from utils import transform_to_scaled_qubit
from openfermion.utils import count_qubits
import numpy as np
import cirq


def trotter_step(operator, time):
    """
    Creates the circuit for applying e^(-j*operator*time), simulating the time
    evolution of a state under the Hamiltonian 'operator'.

    :param operator: qubit operator
    :param time: the evolution time
    :return: trotter_gates
    """

    # Get the number of qubits the operator acts on and define circuit architecture
    n_qubits = count_qubits(operator)
    qubits = cirq.LineQubit.range(n_qubits)

    # Initialize list of gates
    trotter_gates = []

    # Order the terms the same way as done by OpenFermion's
    # trotter_operator_grouping function (sorted keys) for consistency.
    ordered_terms = sorted(list(operator.terms.keys()))

    # Add to trotter_gates the gates necessary to simulate each Pauli string,
    # going through them by the defined order
    for pauli_string in ordered_terms:

        # Get real part of the coefficient (the immaginary one can't be simulated,
        # as the exponent would be real and the operation would not be unitary).
        # Multiply by time to get the full multiplier of the Pauli string.
        coefficient = float(np.real(operator.terms[pauli_string])) * time

        # Keep track of the qubit indices involved in this particular Pauli string.
        # It's necessary so as to know which are included in the sequence of CNOTs
        # that compute the parity
        involved_qubits = []

        # Perform necessary basis rotations
        for pauli in pauli_string:

            # Get the index of the qubit this Pauli operator acts on
            qubit_index = pauli[0]
            involved_qubits.append(qubit_index)

            # Get the Pauli operator identifier (X,Y or Z)
            pauli_operator = pauli[1]

            if pauli_operator == "X":
                # Rotate to X basis
                trotter_gates.append(cirq.H(qubits[qubit_index]))

            if pauli_operator == "Y":
                # Rotate to Y Basis
                trotter_gates.append(cirq.rx(np.pi / 2).on(qubits[qubit_index]))

        # Compute parity and store the result on the last involved qubit
        for i in range(len(involved_qubits) - 1):
            control = involved_qubits[i]
            target = involved_qubits[i + 1]

            trotter_gates.append(cirq.CX(qubits[control], qubits[target]))

        # Apply e^(-i*Z*coefficient) = Rz(coefficient*2) to the last involved qubit
        last_qubit = max(involved_qubits)
        trotter_gates.append(cirq.rz(2 * coefficient).on(qubits[last_qubit]))

        # Uncompute parity
        for i in range(len(involved_qubits) - 2, -1, -1):
            control = involved_qubits[i]
            target = involved_qubits[i + 1]

            trotter_gates.append(cirq.CX(qubits[control], qubits[target]))

        # Undo basis rotations
        for pauli in pauli_string:

            # Get the index of the qubit this Pauli operator acts on
            qubit_index = pauli[0]

            # Get the Pauli operator identifier (X,Y or Z)
            pauli_operator = pauli[1]

            if pauli_operator == "X":
                # Rotate to Z basis from X basis
                trotter_gates.append(cirq.H(qubits[qubit_index]))

            if pauli_operator == "Y":
                # Rotate to Z basis from Y Basis
                trotter_gates.append(cirq.rx(-np.pi / 2).on(qubits[qubit_index]))

    return trotter_gates


class CirqSimulator(SimulatorBase):

    def _get_state_vector(self, state_preparation_gates, *args):

        # Initialize circuit.
        circuit = cirq.Circuit(state_preparation_gates)

        simulation = cirq.Simulator()

        # Access the exact final state vector
        results = simulation.simulate(circuit)

        return results.final_state_vector

    def _get_matrix_operator_gates(self, hf_reference_fock, matrix):

        # Initialize qubits
        n_qubits = len(hf_reference_fock)
        qubits = cirq.LineQubit.range(n_qubits)

        # Initialize the state preparation gates with the Hartree Fock preparation
        state_preparation_gates = self._build_reference_gates(hf_reference_fock)

        # Append the ansatz directly as a matrix
        state_preparation_gates.append(cirq.MatrixGate(matrix.toarray()).on(*qubits))

        return state_preparation_gates

    def _measure_expectation(self, main_string, sub_hamiltonian, shots, state_preparation_gates, n_qubits):
        """
        Measures the expectation value of a sub_hamiltonian (pauli string) using the Cirq simulator.
        By construction, all the expectation values of the strings in subHamiltonian can be
        obtained from the same measurement array. This reduces quantum computer simulations

        :param main_string: hamiltonian base Pauli string ex: (XXYY)
        :param sub_hamiltonian: partial hamiltonian interactions ex: {'0000': -0.4114, '1111': -0.0222}
        :param shots: number of samples to simulate
        :param state_preparation_gates: list of gates in simulation library format that represents the state
        :param n_qubits: number of qubits
        :return:
        """

        # Initialize circuit.
        circuit = cirq.Circuit()

        # Append to the circuit the gates that prepare the state corresponding to
        # the received parameters.
        circuit.append(state_preparation_gates)

        # optimize circuit
        circuit = cirq.eject_z(circuit)
        circuit = cirq.drop_negligible_operations(circuit)

        # define qubits
        qubits = cirq.LineQubit.range(n_qubits)

        # parse string
        # apply operators to measure each qubit in the basis given by main strings
        for i, qubit in enumerate(qubits):
            op = main_string[i]

            if op == "X":
                circuit.append(cirq.H(qubit))

            elif op == "Y":
                circuit.append(cirq.rx(np.pi / 2).on(qubit))

            if op != "I":
                circuit.append(cirq.measure(qubit, key=str(i)))

        # Sample the desired number of repetitions from the circuit, unless
        # there are no measurements (identity term).
        if main_string != "I" * n_qubits:
            simulation = cirq.Simulator()
            results = simulation.run(circuit, repetitions=shots)
        else:
            raise Exception('Nothing to run')

        # For each substring, initialize the sum of all measurements as zero
        measurements = {}
        for sub_string in sub_hamiltonian:
            measurements[sub_string] = 0

        indices = np.array(results.data.T.index, dtype=int)
        for vector in results.data.values:
            for sub_string in sub_hamiltonian:
                prod_function = 1
                for i, v in zip(indices, vector):
                    if main_string[i] != "I":
                        prod_function *= int(1 - 2 * v) ** int(sub_string[i])

                measurements[sub_string] += prod_function

        # Calculate the expectation value of the subHamiltonian, by multiplying
        # the expectation value of each substring by the respective coefficient
        total_expectation_value = 0
        for sub_string, coefficient in sub_hamiltonian.items():
            # Get the expectation value of this substring by taking the average
            # over all the repetitions
            expectation_value = measurements[sub_string] / shots

            # Add this value to the measurements expectation value, weighed by its
            # coefficient
            total_expectation_value += expectation_value * coefficient

        return total_expectation_value

    def _build_reference_gates(self, hf_reference_fock, mapping='jw'):
        """
        Create the gates for preparing the Hartree Fock ground state, that serves
        as a reference state the ansatz

        :param hf_reference_fock: HF reference in fock space
        :param mapping: mapping transform
        :return: reference gates
        """

        # Initialize qubits
        n_qubits = len(hf_reference_fock)
        qubits = cirq.LineQubit.range(n_qubits)

        # Create the gates for preparing the Hartree Fock ground state, that serves
        # as a reference state the ansatz will act on
        reference_gates = []
        if mapping == 'jw':
            for i, occ in enumerate(hf_reference_fock):
                if bool(occ):
                    reference_gates.append(cirq.X(qubits[i]))
                else:
                    reference_gates.append(cirq.I(qubits[i]))
            return reference_gates
            # return [cirq.X(qubits[i]) for i, occ in enumerate(hf_reference_fock) if bool(occ)]

        raise Exception('{} mapping not implemented'.format(mapping))

    def _trotterize_operator(self, qubit_operator, time, trotter_steps):
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

        # Divide time into steps and apply the evolution operator the necessary number of times
        trotter_gates = []
        for step in range(1, trotter_steps + 1):
            trotter_gates += trotter_step(qubit_operator, time / trotter_steps)

        return trotter_gates

    def get_circuit_info(self, coefficients, ansatz, hf_reference_fock):
        ansatz_qubit = transform_to_scaled_qubit(ansatz, coefficients)
        state_preparation_gates = self.get_preparation_gates(ansatz_qubit, hf_reference_fock)
        circuit = cirq.Circuit(state_preparation_gates)

        return {'depth': len(cirq.Circuit(circuit.all_operations()))}


if __name__ == '__main__':
    simulator = CirqSimulator(trotter=True,
                              trotter_steps=1,
                              test_only=True,
                              shots=100)

    print(simulator.get_preparation_gates)