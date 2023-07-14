from vqemulti.simulators import SimulatorBase
from vqemulti.preferences import Configuration
from vqemulti.utils import convert_hamiltonian, group_hamiltonian
from openfermion.utils import count_qubits
from qiskit.quantum_info import SparsePauliOp, Operator
from qiskit.circuit import CircuitInstruction
from qiskit.circuit.library import HGate, RXGate, RZGate, XGate, MCXGate
import qiskit
import numpy as np


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
                trotter_gates.append(CircuitInstruction(HGate(), [qubit_index]))

            if pauli_operator == "Y":
                # Rotate to Y Basis
                trotter_gates.append(CircuitInstruction(RXGate(np.pi / 2), [qubit_index]))

        # Compute parity and store the result on the last involved qubit
        for i in range(len(involved_qubits) - 1):
            control = involved_qubits[i]
            target = involved_qubits[i + 1]
            # trotter_gates.append(qml.CNOT(wires= [control, target]))
            trotter_gates.append(CircuitInstruction(MCXGate(1), [control, target]))

            # Apply e^(-i*Z*coefficient) = Rz(coefficient*2) to the last involved qubit
        last_qubit = max(involved_qubits)
        # trotter_gates.append(qml.RZ((2 * coefficient), wires=last_qubit))
        trotter_gates.append(CircuitInstruction(RZGate((2 * coefficient)), [last_qubit]))

        # Uncompute parity
        for i in range(len(involved_qubits) - 2, -1, -1):
            control = involved_qubits[i]
            target = involved_qubits[i + 1]
            # trotter_gates.append(qml.CNOT(wires=[control, target]))
            trotter_gates.append(CircuitInstruction(MCXGate(1), [control, target]))

        # Undo basis rotations
        for pauli in pauliString:

            # Get the index of the qubit this Pauli operator acts on
            qubit_index = pauli[0]

            # Get the Pauli operator identifier (X,Y or Z)
            pauli_operator = pauli[1]

            if pauli_operator == "X":
                # Rotate to Z basis from X basis
                # trotter_gates.append(qml.Hadamard(qubit_index))
                trotter_gates.append(CircuitInstruction(HGate(), [qubit_index]))

            if pauli_operator == "Y":
                # Rotate to Z basis from Y Basis
                # trotter_gates.append(qml.RX(-np.pi / 2, wires = qubit_index))
                trotter_gates.append(CircuitInstruction(RXGate(-np.pi / 2), [qubit_index]))

    return trotter_gates


class QiskitSimulator(SimulatorBase):

    def __init__(self,
                 trotter=False,
                 trotter_steps=1,
                 test_only=False,
                 shots=1000,
                 backend=qiskit.Aer.get_backend('aer_simulator'),
                 use_estimator=False,
                 session=None,
                 ):

        """
        :param trotter: Trotterize ansatz operators
        :param trotter_steps: number of trotter steps (only used if trotter=True)
        :param test_only: If true resolve QC circuit analytically instead of simulation (for testing circuit)
        :param shots: number of samples to perform in the simulation
        :param backend: qiskit backend to run the calculations
        :param use_estimator: use qiskit estimator instead of VQEmulti implementation
        :param session: IBM runtime session to run jobs on IBM computers (estimator)
        """

        self._backend = backend
        self._session = session
        self._use_estimator = use_estimator

        if session is True:
            self._use_estimator = True

        super().__init__(trotter, trotter_steps, test_only, shots)

    def _get_state_vector(self, state_preparation_gates, n_qubits):

        # Initialize circuit.
        circuit = qiskit.QuantumCircuit(n_qubits)
        for gate in state_preparation_gates:
            circuit.append(gate)

        # print(circuit)

        backend = qiskit.Aer.get_backend('statevector_simulator')
        result = backend.run(circuit).result()

        return np.array(result.get_statevector())

    def _get_matrix_operator_gates(self, hf_reference_fock, matrix):

        # Initialize qubits
        n_qubits = len(hf_reference_fock)

        # Add gates for HF reference
        state_preparation_gates = self._build_reference_gates(hf_reference_fock)

        # Append the ansatz directly as a matrix
        matrix_gate = Operator(np.real(matrix.toarray()))
        state_preparation_gates.append(CircuitInstruction(matrix_gate, list(range(n_qubits))))

        return state_preparation_gates

    def _get_sampled_state_evaluation(self, qubit_hamiltonian, state_preparation_gates):
        """
        Obtains the expectation value in a state by sampling (using a simulator)

        :param qubit_hamiltonian: hamiltonian in qubits
        :param state_preparation_gates: list of gates in simulation library format that represents the state
        :param shots: number of samples
        :return: the expectation value of the energy
        """

        n_qubits = count_qubits(qubit_hamiltonian)

        # Format and group the Hamiltonian, so as to save measurements by using
        # the same data for Pauli strings that only differ by identities
        formatted_hamiltonian = convert_hamiltonian(qubit_hamiltonian)
        grouped_hamiltonian = group_hamiltonian(formatted_hamiltonian)

        if self._use_estimator is False:
            # Obtain the expectation value for each Pauli string
            expectation_value = 0
            for main_string, sub_hamiltonian in grouped_hamiltonian.items():
                expectation_value += self._measure_expectation(main_string,
                                                               sub_hamiltonian,
                                                               self._shots,
                                                               state_preparation_gates,
                                                               n_qubits)

        else:
            expectation_value = self._measure_expectation_estimator(formatted_hamiltonian,
                                                                    state_preparation_gates,
                                                                    n_qubits,
                                                                    self._session)

        assert expectation_value.imag < 1e-5
        return expectation_value.real

    def _measure_expectation(self, main_string, sub_hamiltonian, state_preparation_gates, n_qubits):
        """
        Measures the expectation value of a sub_hamiltonian (pauli string) using the Qiskit simulator.
        By construction, all the expectation values of the strings in subHamiltonian can be
        obtained from the same measurement array. This reduces quantum computer simulations

        :param main_string: hamiltonian base Pauli string ex: (XXYY)
        :param sub_hamiltonian: partial hamiltonian interactions ex: {'0000': -0.4114, '1111': -0.0222}
        :param state_preparation_gates: list of gates in simulation library format that represents the state
        :param n_qubits: number of qubits
        :return: expectation value
        """

        # Initialize circuit and apply hamiltonian gates according to main string
        circuit = qiskit.QuantumCircuit(n_qubits)
        for gate in state_preparation_gates:
            circuit.append(gate)

        # apply operators to measure each qubit in the basis given by main strings
        for i, op in enumerate(main_string):
            if op == "X":
                circuit.h([i])

            elif op == "Y":
                circuit.rx(np.pi / 2, [i])

        self._get_circuit_stat_data(circuit)

        circuit.measure_all()


        result = self._backend.run(circuit, shots=self._shots, memory=True).result()
        # memory = result.get_memory()
        counts_total = result.get_counts()

        # draw circuit
        # print(qml.draw(circuit)())
        def str_to_bit(string):
            return 1 if string == '1' else -1

        # Get function return from measurements in Z according to sub_hamiltonian
        total_expectation_value = 0
        for measure_string, counts in counts_total.items():
            for sub_string, coefficient in sub_hamiltonian.items():

                prod_function = 1
                for i, measure_z in enumerate([str_to_bit(k) for k in measure_string[::-1]]):
                    if main_string[i] != "I":
                        prod_function *= measure_z ** int(sub_string[i])

                total_expectation_value += prod_function * coefficient * counts/self._shots

        return total_expectation_value

    def _measure_expectation_estimator(self, formatted_hamiltonian, state_preparation_gates, n_qubits, session):
        """
        get the expectation value of the full Hamiltonian

        :param formatted_hamiltonian: hamiltonian in dictionary of Pauli strings and coefficients
        :param state_preparation_gates: list of gates in simulation library format that represents the state
        :param n_qubits: number of qubits
        :param session: quiskit IBM runtime session
        :return: expectation value of the full hamiltonian
        """

        # apply hamiltonian gates according to main string
        circuit = qiskit.QuantumCircuit(n_qubits)
        for gate in state_preparation_gates:
            circuit.append(gate)

        if Configuration().verbose:
            print('circuit:')
            print(circuit)

        list_strings = []
        for pauli_string, coefficient in formatted_hamiltonian.items():
            list_strings.append((pauli_string, coefficient))

        measure_op = SparsePauliOp.from_list(list_strings)

        if Configuration().verbose:
            print('measure operator:')
            print(measure_op)

        if self._session is None:
            from qiskit_aer.primitives import Estimator
            estimator = Estimator(abelian_grouping=False)
        else:
            from qiskit_ibm_runtime import Estimator, Options
            options = Options(optimization_level=3)
            estimator = Estimator(session=session, options=options)

        # circuit stats
        for _ in list_strings:
            self._get_circuit_stat_data(circuit)

        # estimate [ <psi|H|psi)> ]
        job = estimator.run(circuits=[circuit], observables=[measure_op], shots=self._shots)#, abelian_grouping=True)

        if Configuration().verbose:
            print('Expectation value: ', job.result().values[0])

        return job.result().values[0]

    def _build_reference_gates(self, hf_reference_fock):
        """
        Create the gates for preparing the Hartree Fock ground state, that serves
        as a reference state the ansatz

        :param hf_reference_fock: HF reference in fock space
        :param mapping: mapping transform
        :return: reference gates
        """

        n_qubits = len(hf_reference_fock)

        reference_gates = []
        for i, occ in enumerate(hf_reference_fock):
            if bool(occ):
                reference_gates.append(CircuitInstruction(XGate(), [n_qubits-i-1]))

        return reference_gates


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

        # Divide time into steps and apply the evolution operator the necessary
        # number of times
        trotter_gates = []
        for step in range(1, trotter_steps + 1):
            trotter_gates += trotter_step(qubit_operator, time / trotter_steps)

        return trotter_gates

    def _get_circuit_stat_data(self, circuit):

        gates_name = {'x': 'PauliX', 'y': 'PauliY', 'z': 'PauliZ',
                      'rx': 'RX', 'ry': 'RY', 'rz': 'RZ',
                      'i': 'Identity', 'h': 'Hadamard', 'cx': 'CNOT',
                      'unitary': 'QubitUnitary'}

        # depth
        self._circuit_count.append(circuit.depth())

        # gates
        for gate in circuit.data:
            self._circuit_gates[gates_name[gate[0].name]] += 1

    def get_circuit_info(self, coefficients, ansatz, hf_reference_fock):

        ansatz_qubit = ansatz.transform_to_scaled_qubit(coefficients)

        state_preparation_gates = self.get_preparation_gates(ansatz_qubit, hf_reference_fock)

        # Initialize circuit.
        n_qubits = len(hf_reference_fock)

        circuit = qiskit.QuantumCircuit(n_qubits)
        for gate in state_preparation_gates:
            circuit.append(gate)

        return {'depth': circuit.depth()}


if __name__ == '__main__':

    circuit = qiskit.QuantumCircuit(2)

    circuit.h(0)
    cx = Operator([[1, 0, 0, 0],
                   [0, 0, 0, 1],
                   [0, 0, 1, 0],
                   [0, 1, 0, 0]
                   ])


    #return self.append(HGate(), [qubit], [])

    list_gates = [CircuitInstruction(HGate(), [1]), CircuitInstruction(cx, [0, 1])]
    circuit2 = qiskit.QuantumCircuit(2)
    for gate in list_gates:
        circuit2.append(gate)

    print(circuit2)
    exit()

    for gate in circuit.data:
        print(gate)

    print('----------')
    print(circuit.data)

    #circuit = qiskit.QuantumCircuit()

    print(circuit)
