from utils import convert_hamiltonian, group_hamiltonian, string_to_matrix
from openfermion.utils import count_qubits
from openfermion import get_sparse_operator
from openfermion import QubitOperator
import scipy
import numpy as np


class SimulatorBase:
    def __init__(self,
                 trotter=False,
                 trotter_steps=1,
                 test_only=False,
                 shots=1000):
        """
        :param trotter: Trotterize ansatz operators
        :param trotter_steps: number of trotter steps (only used if trotter=True)
        :param test_only: If true resolve QC circuit analytically instead of simulation (for testing circuit)
        :param shots: number of samples to perform in the simulation
        """

        self._trotter = trotter
        self._trotter_steps = trotter_steps
        self._test_only = test_only
        self._shots = shots

    def get_state_evaluation(self, qubit_hamiltonian, state_preparation_gates):

        if self._test_only:
            return self._get_exact_state_evaluation(qubit_hamiltonian, state_preparation_gates)
        else:
            return self._get_sampled_state_evaluation(qubit_hamiltonian, state_preparation_gates)

    def get_preparation_gates(self, ansatz, hf_reference_fock):

        if self._trotter:
            return self._get_preparation_gates_trotter(ansatz, hf_reference_fock)
        else:
            return self._get_preparation_gates_matrix(ansatz, hf_reference_fock)

    def _get_preparation_gates_matrix(self, ansatz, hf_reference_fock):
        """
        generate operation gates for a given ansantz in simulation library format (Cirq, pennylane, etc..)

        :param ansatz: operators list in qubit
        :param hf_reference_fock: reference HF in fock vspace vector
        :param n_qubits: number of qubits
        :return: gates list in simulation library format
        """

        # generate matrix operator that corresponds to ansatz
        identity = scipy.sparse.identity(2, format='csc', dtype=complex)
        matrix = identity
        n_qubits = len(hf_reference_fock)
        for _ in range(n_qubits - 1):
            matrix = scipy.sparse.kron(identity, matrix, 'csc')

        for operator in ansatz:
            # Get corresponding the operator matrix (exponent)
            operator_matrix = get_sparse_operator(operator, n_qubits)

            # Add unitary operator to matrix as exp(operator_matrix)
            matrix = scipy.sparse.linalg.expm(operator_matrix) * matrix

        # Get gates in simulation library format
        state_preparation_gates = self._get_matrix_operator_gates(hf_reference_fock, matrix)

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

        # Obtain the expectation value for each Pauli string by
        # calling the measure_expectation function, and perform the necessary weighed
        # sum to obtain the energy expectation value
        energy = 0
        for main_string, sub_hamiltonian in grouped_hamiltonian.items():
            expectation_value = self._measure_expectation(main_string,
                                                          sub_hamiltonian,
                                                          self._shots,
                                                          state_preparation_gates,
                                                          n_qubits)
            energy += expectation_value

        assert energy.imag < 1e-5

        return energy.real

    def _get_exact_state_evaluation(self, qubit_hamiltonian, state_preparation_gates):
        """
        Calculates the exact evaluation of a state with a given hamiltonian using matrix algebra.
        This function is basically used to test that the Pennylane circuit is correct

        :param qubit_hamiltonian: hamiltonian in qubits
        :param state_preparation_gates: list of gates in simulation library format that represents the state
        :return: the expectation value of the state given the hamiltonian
        """

        n_qubits = count_qubits(qubit_hamiltonian)
        state_vector = self._get_state_vector(state_preparation_gates, n_qubits)

        formatted_hamiltonian = convert_hamiltonian(qubit_hamiltonian)

        # Obtain the theoretical expectation value for each Pauli string in the
        # Hamiltonian by matrix multiplication, and perform the necessary weighed
        # sum to obtain the energy expectation value.
        energy = 0
        for pauli_string, coefficient in formatted_hamiltonian.items():
            ket = np.array(state_vector, dtype=complex)
            bra = np.conj(ket)

            pauli_ket = np.matmul(string_to_matrix(pauli_string), ket)
            expectation_value = np.real(np.dot(bra, pauli_ket))

            energy += coefficient * expectation_value

        return energy

    def _get_preparation_gates_trotter(self, ansatz_qubit, hf_reference_fock):
        """
        Trotterize the ansatz

        :param ansatz: operators list in qubit
        :param trotter_steps: number of trotter steps
        :param hf_reference_fock: reference HF in Fock vspace vector
        :return: trotterized gates list
        """

        coefficients = None
        # Initialize the ansatz gate list
        trotter_ansatz = []
        # Go through the operators in the ansatz
        for operator in ansatz_qubit:

            for op, time in operator.terms.items():
                operator_trotter_circuit = self._trotterize_operator(-QubitOperator(op),
                                                                     time.imag,
                                                                     self._trotter_steps)

                # Add the gates corresponding to this operator to the ansatz gate list
                trotter_ansatz += operator_trotter_circuit

        # Initialize the state preparation gates with the reference state preparation gates
        state_preparation_gates = self._build_reference_gates(hf_reference_fock)

        # return total trotterized ansatz
        return state_preparation_gates + trotter_ansatz

    # mock methods (to be implemented in subclasses)
    def _measure_expectation(self, *args):
        raise NotImplementedError()

    def _get_state_vector(self, *args):
        raise NotImplementedError()

    def _get_matrix_operator_gates(self, *args):
        raise NotImplementedError()

    def _build_reference_gates(self, *args):
        raise NotImplementedError()

    def _trotterize_operator(self, *args):
        raise NotImplementedError()
