from vqemulti.simulators import SimulatorBase
from vqemulti.preferences import Configuration
from vqemulti.utils import convert_hamiltonian, group_hamiltonian
from vqemulti.simulators.tools import get_cnot_inversion_mat
from vqemulti.simulators.ibm_hardware import RHESampler, RHEstimator
from openfermion.utils import count_qubits
from qiskit.quantum_info import SparsePauliOp, Operator
from qiskit.circuit import CircuitInstruction
from qiskit.circuit.library import HGate, RXGate, RZGate, XGate, MCXGate
from qiskit import transpile
from qiskit_aer import AerSimulator, StatevectorSimulator
from qiskit_aer.primitives import Estimator, Sampler
import qiskit
import numpy as np


def trotter_step_standard(qubit_operator, time, n_qubits):
    """
    Creates the circuit for applying e^(-j*operator*time), simulating the time
    evolution of a state under the Hamiltonian 'operator'.
    The implementation is done using standard staircase algorithm

    :param qubit_operator: qubit operator
    :param time: the evolution time
    :return: trotter_gates
    """

    # Initialize list of gates
    trotter_gates = []

    # Order the terms the same way as done by OpenFermion's
    # trotter_operator_grouping function (sorted keys) for consistency.
    ordered_terms = sorted(list(qubit_operator.terms.keys()))
    top_wire = n_qubits - 1

    # Add to trotter_gates the gates necessary to simulate each Pauli string,
    # going through them by the defined order
    for pauliString in ordered_terms:
        # print('-->', pauliString)

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
            trotter_gates.append(CircuitInstruction(MCXGate(1), [control, target]))

            # Apply e^(-i*Z*coefficient) = Rz(coefficient*2) to the last involved qubit
        last_qubit = max(involved_qubits) if len(involved_qubits) != 0 else 0

        trotter_gates.append(CircuitInstruction(RZGate((2 * coefficient)), [last_qubit]))

        # Uncompute parity
        for i in range(len(involved_qubits) - 2, -1, -1):
            control = involved_qubits[i]
            target = involved_qubits[i + 1]
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
                trotter_gates.append(CircuitInstruction(RXGate(-np.pi / 2), [qubit_index]))

    return trotter_gates


def trotter_step_inverse(qubit_operator, time, n_qubits):
    """
    Creates the circuit for applying e^(-j*operator*time), simulating the time
    evolution of a state under the Hamiltonian 'operator'.
    The implementation is done using inverse staircase algorithm

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

            if pauli_operator == "Z":
                # Rotate to Z basis
                trotter_gates.append(CircuitInstruction(HGate(), [qubit_index]))

            if pauli_operator == "Y":
                # Rotate to Y Basis
                #trotter_gates.append(CircuitInstruction(RXGate(np.pi / 2), [qubit_index]))
                trotter_gates.append(CircuitInstruction(RZGate(-np.pi / 2), [qubit_index]))
                #trotter_gates.append(CircuitInstruction(HGate(), [qubit_index]))

        # Compute parity and store the result on the last involved qubit
        for i in range(len(involved_qubits) - 1):
            control = involved_qubits[i]
            target = involved_qubits[i + 1]
            trotter_gates.append(CircuitInstruction(MCXGate(1), [target, control]))

            # Apply e^(-i*Z*coefficient) = Rz(coefficient*2) to the last involved qubit
        last_qubit = max(involved_qubits) if len(involved_qubits) != 0 else 0

        trotter_gates.append(CircuitInstruction(RXGate((2 * coefficient)), [last_qubit]))

        # Uncompute parity
        for i in range(len(involved_qubits) - 2, -1, -1):
            control = involved_qubits[i]
            target = involved_qubits[i + 1]
            trotter_gates.append(CircuitInstruction(MCXGate(1), [target, control]))

        # Undo basis rotations
        for pauli in pauliString:

            # Get the index of the qubit this Pauli operator acts on
            qubit_index = pauli[0]

            # Get the Pauli operator identifier (X,Y or Z)
            pauli_operator = pauli[1]

            if pauli_operator == "Z":
                # Rotate to X basis from Z basis
                trotter_gates.append(CircuitInstruction(HGate(), [qubit_index]))

            if pauli_operator == "Y":
                # Rotate to X basis from Y Basis
                #trotter_gates.append(CircuitInstruction(HGate(), [qubit_index]))
                trotter_gates.append(CircuitInstruction(RZGate(np.pi / 2), [qubit_index]))

    return trotter_gates


def last_gate(gates_list, qubit_indices, ignore_z=False):
    """
    return the last gate connected to a qubit (or list of qubits) in a list of gates

    :param gates_list: list of gates to be implemented in a circuit
    :param qubit_indices: list of qubit indices
    :param ignore_z: ignores rz in the first index in qubit_indices (to ignore Rz applied to CNOT control qubit)
    :return:
    """
    tot_len = len(gates_list)
    for index, gate in enumerate(gates_list[::-1]):
        if gate.operation.name in ['rz', 'z'] and ignore_z and gate.qubits[0] == qubit_indices[0]:
            continue
        if np.any([q in gate.qubits for q in qubit_indices]):
            return gate, tot_len - index - 1

    return CircuitInstruction(type('MyClass', (object,), {'name': ''})()), None


def trotter_step(qubit_operator, time, n_qubits):
    """
    Creates the circuit for applying e^(-j*operator*time), simulating the time
    evolution of a state under the Hamiltonian 'operator'.
    The implementation is done using combined original/inverse staircase algorithm

    :param qubit_operator: qubit operator
    :param time: the evolution time
    :return: trotter_gates
    """

    # Initialize list of gates
    trotter_gates = []

    # Order the terms the same way as done by OpenFermion's
    # trotter_operator_grouping function (sorted keys) for consistency.
    ordered_terms = sorted(list(qubit_operator.terms.keys()))

    # define order of algorithm
    cnot_inversion_matrix = get_cnot_inversion_mat(ordered_terms, n_qubits)

    # Add to trotter_gates the gates necessary to simulate each Pauli string,
    # going through them by the defined order
    for pauliString, cnot_inversion_list in zip(ordered_terms, cnot_inversion_matrix):
        # print(pauliString)
        if len(pauliString) == 0:
            continue

        # handle first qubit in pauli string
        first_qubit_index = pauliString[0][0]
        if first_qubit_index > 0:
            cnot_inversion_list[first_qubit_index-1] = cnot_inversion_list[first_qubit_index]

        # Get real part of the coefficient (the immaginary one can't be simulated,
        # as the exponent would be real and the operation would not be unitary).
        # Multiply by time to get the full multiplier of the Pauli string.
        coefficient = float(np.real(qubit_operator.terms[pauliString])) * time

        # Keep track of the qubit indices involved in this particular Pauli string.
        # It's necessary so as to know which are included in the sequence of CNOTs
        # that compute the parity
        involved_qubits = []
        for _, pauli in enumerate(pauliString):

            # Get the index of the qubit this Pauli operator acts on
            qubit_index = pauli[0]
            involved_qubits.append(qubit_index)

            if qubit_index > 0:
                inverted_staircase = cnot_inversion_list[qubit_index-1]
            else:
                inverted_staircase = cnot_inversion_list[0]

            # Get the Pauli operator identifier (X,Y or Z)
            pauli_operator = pauli[1]

            if not inverted_staircase:
                if pauli_operator == "X":
                    previous_gate, index = last_gate(trotter_gates, (qubit_index,))
                    if previous_gate.operation.name == 'h':
                        del trotter_gates[index]
                    else:
                        # Rotate to Z basis from X basis
                        trotter_gates.append(CircuitInstruction(HGate(), [qubit_index]))
                        pass

                elif pauli_operator == "Y":
                    previous_gate, index = last_gate(trotter_gates, (qubit_index,))
                    if previous_gate.operation.name == 'rx':
                        del trotter_gates[index]
                    else:
                        # Rotate to Y Basis
                        trotter_gates.append(CircuitInstruction(RXGate(np.pi / 2), [qubit_index]))
            else:
                if pauli_operator == "Z":
                    previous_gate, index = last_gate(trotter_gates, (qubit_index,))
                    if previous_gate.operation.name == 'h':
                        del trotter_gates[index]
                    else:
                        # Rotate to Z basis from X basis
                        trotter_gates.append(CircuitInstruction(HGate(), [qubit_index]))

                if pauli_operator == "Y":
                    # Rotate to Y Basis
                    previous_gate, index = last_gate(trotter_gates, (qubit_index,))
                    if previous_gate.operation.name == 'rz':
                        del trotter_gates[index]
                    else:
                        trotter_gates.append(CircuitInstruction(RZGate(-np.pi / 2), [qubit_index]))

        last_qubit = max(involved_qubits) if len(involved_qubits) != 0 else 0

        # Compute parity and store the result on the last involved qubit
        for i in range(len(involved_qubits) - 1):

            inverted_staircase = cnot_inversion_list[involved_qubits[i]]

            target = involved_qubits[i] if inverted_staircase else involved_qubits[i + 1]
            control = involved_qubits[i + 1] if inverted_staircase else involved_qubits[i]

            if control < n_qubits-1 and target < n_qubits-1:
                transition_point = cnot_inversion_list[control] != cnot_inversion_list[target]
            else:
                transition_point = False

            previous_gate, index = last_gate(trotter_gates, (control, target), ignore_z=True)
            if previous_gate.operation.name == 'cx' and np.all(previous_gate.qubits == (control, target)):
                del trotter_gates[index]
            else:
                trotter_gates.append(CircuitInstruction(MCXGate(1), [control, target]))

            qubit_index_sup = involved_qubits[i + 1]
            #if i < len(cnot_inversion_list)-1 and cnot_inversion_list[i] != cnot_inversion_list[i+1]:
            if i < len(cnot_inversion_list) - 1 and transition_point:
                if not last_qubit == qubit_index_sup:
                    previous_gate, index = last_gate(trotter_gates, (qubit_index_sup,))
                    if previous_gate.operation.name == 'h':
                        del trotter_gates[index]
                    else:
                        trotter_gates.append(CircuitInstruction(HGate(), [qubit_index_sup]))

        # Apply e^(-i*Z*coefficient) = Rz(coefficient*2) to the last involved qubit
        if not cnot_inversion_list[-1]:
            trotter_gates.append(CircuitInstruction(RZGate((2 * coefficient)), [last_qubit]))
        else:
            trotter_gates.append(CircuitInstruction(RXGate((2 * coefficient)), [last_qubit]))

        # Uncompute parity
        for i in range(len(involved_qubits) - 2, -1, -1):
            inverted_staircase = cnot_inversion_list[involved_qubits[i]]

            qubit_index_sup = involved_qubits[i + 1]

            target = involved_qubits[i] if inverted_staircase else involved_qubits[i + 1]
            control = involved_qubits[i + 1] if inverted_staircase else involved_qubits[i]
            if control < n_qubits-1 and target < n_qubits-1:
                transition_point = cnot_inversion_list[control] != cnot_inversion_list[target]
            else:
                transition_point = False

            if i < len(cnot_inversion_list) - 1 and transition_point:

                if not last_qubit == qubit_index_sup:
                    trotter_gates.append(CircuitInstruction(HGate(), [qubit_index_sup]))

            trotter_gates.append(CircuitInstruction(MCXGate(1), [control, target]))

        # Undo basis rotations
        for i, pauli in enumerate(pauliString):

            # Get the index of the qubit this Pauli operator acts on
            qubit_index = pauli[0]
            involved_qubits.append(qubit_index)

            if qubit_index > 0:
                inverted_staircase = cnot_inversion_list[qubit_index - 1]
            else:
                inverted_staircase = cnot_inversion_list[0]

            # Get the Pauli operator identifier (X,Y or Z)
            pauli_operator = pauli[1]

            if not inverted_staircase:
                if pauli_operator == "X":
                    # Rotate to Z basis from X basis
                    trotter_gates.append(CircuitInstruction(HGate(), [qubit_index]))

                elif pauli_operator == "Y":
                    # Rotate to Z basis from Y Basis
                    trotter_gates.append(CircuitInstruction(RXGate(-np.pi / 2), [qubit_index]))
            else:
                if pauli_operator == "Z":
                    # Rotate to Z basis from X basis
                    trotter_gates.append(CircuitInstruction(HGate(), [qubit_index]))

                elif pauli_operator == "Y":
                    # Rotate to X basis from Y Basis
                    trotter_gates.append(CircuitInstruction(RZGate(np.pi / 2), [qubit_index]))

    return trotter_gates


class QiskitSimulator(SimulatorBase):

    def __init__(self,
                 trotter=False,
                 trotter_steps=1,
                 test_only=False,
                 hamiltonian_grouping=True,
                 separate_matrix_operators=True,
                 shots=1000,
                 qiskit_optimizer=False,
                 backend=AerSimulator(),
                 use_estimator=False,
                 session=None,
                 use_ibm_runtime=False
                 ):

        """
        :param trotter: Trotterize ansatz operators
        :param trotter_steps: number of trotter steps (only used if trotter=True)
        :param test_only: If true resolve QC circuit analytically instead of simulation (for testing circuit)
        :param separate_matrix_operators: separate adaptVQE matrix operators (only with test_only = True)
        :param shots: number of samples to perform in the simulation
        :param qiskit_optimizer: use qiskit transpiler optimization machinery to extra optimize the quantum circuits
        :param backend: qiskit backend to run the calculations
        :param use_estimator: use qiskit estimator instead of VQEmulti implementation
        :param session: IBM runtime session to run jobs on IBM computers (estimator)
        :param use_ibm_runtime: use ibm_runtime version of Estimator and Sampler
        """
        # backend.set_options(device='GPU')
        self._backend = backend
        self._session = session
        self._use_estimator = use_estimator
        self._qiskit_optimizer = qiskit_optimizer
        self._use_ibm_runtime = use_ibm_runtime

        if session is not None:
            self._use_ibm_runtime = True

            from qiskit_ibm_runtime import QiskitRuntimeService
            from qiskit.providers.exceptions import QiskitBackendNotFoundError
            try:
                # service = QiskitRuntimeService()
                #self._backend = service.backend(session.backend())
                self._backend = session._backend
            except QiskitBackendNotFoundError:
                raise Exception('Backend not found')

        super().__init__(trotter, trotter_steps, test_only, hamiltonian_grouping, separate_matrix_operators, shots)

    def _get_state_vector(self, state_preparation_gates, n_qubits):

        # Initialize circuit.
        circuit = qiskit.QuantumCircuit(n_qubits)
        for gate in state_preparation_gates:
            circuit.append(gate)

        self._get_circuit_stat_data(circuit)

        # set the same qubit order as other simulators
        circuit = circuit.reverse_bits()
        result = StatevectorSimulator().run(circuit).result()

        return np.array(result.get_statevector())

    def _get_matrix_operator_gates(self, hf_reference_fock, matrix_list):

        # Initialize qubits
        n_qubits = len(hf_reference_fock)

        # Add gates for HF reference
        state_preparation_gates = self._build_reference_gates(hf_reference_fock)

        # Append the ansatz directly as a matrix
        for matrix in matrix_list:
            matrix_gate = Operator(np.real(matrix.toarray()))
            state_preparation_gates.append(CircuitInstruction(matrix_gate, list(range(n_qubits-1, -1, -1))))

        return state_preparation_gates

    def _get_sampled_state_evaluation(self, qubit_hamiltonian, state_preparation_gates):
        """
        Obtains the expectation value in a state by sampling (using a simulator)

        :param qubit_hamiltonian: hamiltonian in qubits
        :param state_preparation_gates: list of gates in simulation library format that represents the state
        :param shots: number of samples
        :return: the expectation value of the energy
        """

        if self._use_estimator is False:
            return super()._get_sampled_state_evaluation(qubit_hamiltonian, state_preparation_gates)

        # get the number of qubits
        n_qubits = count_qubits(qubit_hamiltonian)

        # Format and group the Hamiltonian, so as to save measurements by using
        # the same data for Pauli strings that only differ by identities
        formatted_hamiltonian = convert_hamiltonian(qubit_hamiltonian)

        expectation_value, variance = self._measure_expectation_estimator(formatted_hamiltonian,
                                                                          state_preparation_gates,
                                                                          n_qubits,
                                                                          self._session)

        print('VARIANCE ESTIMATOR: ', variance)

        return expectation_value, variance

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

        if not self._use_ibm_runtime:
            result = self._backend.run(circuit, shots=self._shots, memory=True).result()
            # memory = result.get_memory()
        else:
            estimator = RHESampler(self._backend, n_qubits, self._session)
            result = estimator.run(circuit, shots=self._shots, memory=True).result()

        counts_total = result.get_counts()

        # draw circuit
        def str_to_bit(string):
            return 1 if string == '0' else -1

        # Get function return from measurements in Z according to sub_hamiltonian
        total_expectation_value = 0
        total_variance = 0
        for sub_string, coefficient in sub_hamiltonian.items():
            expectation_value = 0
            for measure_string, counts in counts_total.items():

                prod_function = 1
                for i, measure_z in enumerate([str_to_bit(k) for k in measure_string[::-1]]):
                    if main_string[i] != "I":
                        prod_function *= measure_z ** int(sub_string[i])

                expectation_value += prod_function * coefficient * counts/self._shots

            total_variance += coefficient ** 2 - expectation_value**2
            total_expectation_value += expectation_value

        return total_expectation_value, total_variance

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

        #if Configuration().verbose:
        #    print('circuit:')
        #    print(circuit)

        list_strings = []
        for pauli_string, coefficient in formatted_hamiltonian.items():
            # print(pauli_string, coefficient)
            list_strings.append((pauli_string[::-1], coefficient))

        measure_op = SparsePauliOp.from_list(list_strings)

        #if Configuration().verbose:
        #    print('measure operator:')
        #    print(measure_op)

        # circuit stats
        for _ in list_strings:
            self._get_circuit_stat_data(circuit)

        if not self._use_ibm_runtime:
            estimator = Estimator(abelian_grouping=self._hamiltonian_grouping)
            job = estimator.run(circuits=[circuit], observables=[measure_op], shots=self._shots)

        else:
            estimator = RHEstimator(self._backend, n_qubits, session=self._session)
            job = estimator.run(circuit, measure_op, shots=self._shots)

        expectation_value = sum(job.result().values)

        # get variance
        variance = sum([meta['variance'] for meta in job.result().metadata])
        std = np.sqrt(variance / self._shots)

        if Configuration().verbose > 1:
            print('Expectation value: ', expectation_value)
            print('variance: ', variance)
            print('std:', std)

        return expectation_value, variance

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
                reference_gates.append(CircuitInstruction(XGate(), [i]))

        return reference_gates

    def _trotterize_operator(self, qubit_operator, n_qubits):
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
        for step in range(1, self._trotter_steps + 1):
            # trotter_gates += trotter_step_standard(1j * qubit_operator, 1 / self._trotter_steps, n_qubits)
            # trotter_gates += trotter_step_inverse(1j * qubit_operator, 1 / self._trotter_steps, n_qubits)
            trotter_gates += trotter_step(1j * qubit_operator, 1 / self._trotter_steps, n_qubits)

        return trotter_gates

    def _get_circuit_stat_data(self, circuit):

        gates_name = {'x': 'PauliX', 'y': 'PauliY', 'z': 'PauliZ',
                      'rx': 'RX', 'ry': 'RY', 'rz': 'RZ',
                      'i': 'Identity', 'h': 'Hadamard', 'cx': 'CNOT',
                      'unitary': 'QubitUnitary'}

        # circuit optimization using qiskit
        if self._qiskit_optimizer:
            circuit = transpile(circuit, optimization_level=3, basis_gates=['x', 'cx', 'rx', 'ry', 'rz', 'h'])

        # circuit drawing
        self._circuit_draw.append(str(circuit.draw(fold=-1, reverse_bits=True)))
        if Configuration().verbose > 2:
            print(self._circuit_draw[-1])

        # depth
        self._circuit_count.append(circuit.depth())
        self._shot_count.append(self._shots)

        # gates
        for gate in circuit.data:
            try:
                self._circuit_gates[gates_name[gate[0].name]] += 1
            except KeyError:
                gates_name.update({gate[0].name: gate[0].name})
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

    def simulator_info(self):
        return 'qiskit ' + str(qiskit.__version__)
