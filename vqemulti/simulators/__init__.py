from h5py.h5p import PropDCID
from vqemulti.utils import convert_hamiltonian, group_hamiltonian, string_to_matrix, ansatz_to_matrix, ansatz_to_matrix_list
from openfermion.utils import count_qubits
from collections import defaultdict
import numpy as np
import warnings


class SimulatorBase:
    def __init__(self,
                 trotter=False,
                 trotter_steps=1,
                 test_only=False,
                 hamiltonian_grouping=True,
                 separate_matrix_operators=True,
                 shots=1000):
        """
        :param trotter: Trotterize ansatz operators
        :param trotter_steps: number of trotter steps (only used if trotter=True)
        :param test_only: If true resolve QC circuit analytically instead of simulation (for testing circuit)
        :param hamiltonian_grouping: organize Hamiltonian into Abelian commutative groups (reduce evaluations)
        :param separate_matrix_operators: separate adaptVQE matrix operators (only with test_only = True)
        :param shots: number of samples to perform in the simulation
        """

        self._trotter = trotter
        self._trotter_steps = trotter_steps
        self._test_only = test_only
        self._shots = shots
        self._circuit_count = []
        self._shot_count = []
        self._circuit_gates = defaultdict(int)
        self._circuit_draw = []
        self._shots_model = None
        self._hamiltonian_grouping = hamiltonian_grouping
        self._separate_matrix_operators = separate_matrix_operators
        self._n_hamiltonian_terms = None
        self.cnot_max = 0

    def set_shots_model(self, shots_model):
        self._shots_model = shots_model

    def get_state_evaluation(self, qubit_hamiltonian, state_preparation_gates):

        if self._test_only:
            evaluation, std_error = self._get_exact_state_evaluation(qubit_hamiltonian, state_preparation_gates)
            std_error = 0
        else:
            evaluation, std_error = self._get_sampled_state_evaluation(qubit_hamiltonian, state_preparation_gates)

        # make sure that main simulator class returns consistent float value
        return float(evaluation), float(std_error)

    def _get_sampled_state_evaluation(self, qubit_hamiltonian, state_preparation_gates):
        """
        Obtains the expectation value in a state by sampling (using a simulator)

        :param qubit_hamiltonian: hamiltonian in qubits
        :param state_preparation_gates: list of gates in simulation library format that represents the state
        :param shots: number of samples
        :return: the expectation value of the energy
        """

        n_qubits = count_qubits(qubit_hamiltonian)

        # Format and the Hamiltonian in pauli strings and coefficients
        formatted_hamiltonian = convert_hamiltonian(qubit_hamiltonian)

        if self._hamiltonian_grouping:
            # use hamiltonian grouping
            grouped_hamiltonian = group_hamiltonian(formatted_hamiltonian)
        else:
            # skip hamiltonian grouping
            grouped_hamiltonian = {}
            for pauli_string, coefficient in formatted_hamiltonian.items():
                grouped_hamiltonian[pauli_string] = {'1' * len(pauli_string): coefficient}

        # Obtain the expectation value for each Pauli string
        expectation_value = 0
        expectation_variance = 0

        for main_string, sub_hamiltonian in grouped_hamiltonian.items():
            energy, variance = self._measure_expectation(main_string,
                                                         sub_hamiltonian,
                                                         state_preparation_gates,
                                                         n_qubits)
            expectation_value += energy
            expectation_variance += variance

        std_error = np.sqrt(expectation_variance.real/self._shots)

        assert expectation_value.imag < 1e-5
        return expectation_value.real, std_error

    def get_state_evaluation_variance(self, qubit_hamiltonian, state_preparation_gates):

        if self._test_only:
            energy, std_error = self._get_exact_state_evaluation(qubit_hamiltonian, state_preparation_gates)
        else:
            energy, std_error = self._get_sampled_state_evaluation(qubit_hamiltonian, state_preparation_gates)

        variance = std_error**2*self._shots

        # make sure that main simulator class returns consistent float value
        return float(variance)

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

        variance_list = []
        energy_list = []

        for pauli_string, coefficient in formatted_hamiltonian.items():
            ket = np.array(state_vector, dtype=complex)
            bra = np.conj(ket)

            pauli_ket = np.matmul(string_to_matrix(pauli_string), ket)
            energy = coefficient * np.dot(bra, pauli_ket).real

            pauli_ket_square = np.matmul(string_to_matrix(pauli_string), pauli_ket)
            energy_2 = coefficient**2 * np.dot(bra, pauli_ket_square).real

            energy_list.append(energy)
            variance_list.append(energy_2 - energy**2)

            # print('variance: {:.5f} {}'.format(energy_2 - energy**2, n_shots))

        variance = np.sum(variance_list).real
        std_error = np.sqrt(variance/self._shots)

        return np.sum(energy_list).real, std_error

    def get_preparation_gates(self, ansatz, hf_reference_fock):
        """
        generate operation gates for a given ansantz in simulation library format (Cirq, pennylane, etc..)

        :param ansatz: operators list in qubit
        :param hf_reference_fock: reference HF in fock vspace vector
        :param n_qubits: number of qubits
        :return: gates list in simulation library format
        """
        n_qubits = len(hf_reference_fock)

        if self._trotter:
            # Use discrete (trotterized) operator gates

            trotter_ansatz = []
            # Go through the operators in the ansatz
            import time
            start = time.time()
            for operator in ansatz:
                # Add the gates corresponding to this operator to the ansatz gate list
                trotter_ansatz += self._trotterize_operator(operator, n_qubits)
            end = time.time()
            print('***********************')
            print('EXTRA TIME TROTTER', end - start)
            print('***********************')
            # Initialize the state preparation gates with the reference state preparation gates
            state_preparation_gates = self._build_reference_gates(hf_reference_fock)

            # return total trotterized ansatz
            return state_preparation_gates + trotter_ansatz

        else:
            # Use matrix gate
            if self._separate_matrix_operators:
                matrix_list = ansatz_to_matrix_list(ansatz, n_qubits)
            else:
                matrix_list = [ansatz_to_matrix(ansatz, n_qubits)]

            # Get gates in simulation library format
            return self._get_matrix_operator_gates(hf_reference_fock, matrix_list)

    def print_statistics(self):
        if len(self._circuit_count) <= 0:
            warnings.warn('No simulation statistics to show')
            return

        print('\n')
        if self._test_only:
            print('Exact evaluation (one evaluation per circuit)')
        else:
            print('Sampled evaluation (one evaluation per hamiltonian term)')
        print('------------------------------------')

        print('version: {}'.format(self.simulator_info()))
        if not self._test_only:
            if self._shots_model is None:
                print('Shots per evaluation: {}'.format(self._shots))
            print('Total shots: {}'.format(np.sum(self._shot_count)))

        if self._n_hamiltonian_terms is not None:
            print('Hamiltonian terms: {}'.format(self._n_hamiltonian_terms))

        print('Circuit evaluations: {}'.format(len(self._circuit_count)))
        print('Total circuit depth: {}'.format(sum(self._circuit_count)))
        print('Maximum circuit depth: {}'.format(np.max(self._circuit_count)))
        print('Maximum number of CNOTS: {}'.format(self.cnot_max))
        print('Average circuit depth: {:.2f}'.format(np.average(self._circuit_count)))
        print('Gate counts:')
        for k, v in self._circuit_gates.items():
            print(' {:14} : {}'.format(k, v))
        print('------------------------------------\n')

    def print_circuits(self):
        print('Total circuits: {}'.format(len(self._circuit_draw)))
        # self._circuit_draw stores all the circuits but ready to be drawed, this is done
        # inside each simulator in function _get_circuit_stat_data
        for i, c in enumerate(self._circuit_draw):
            print('\n Circuit {}\n'.format(i+1) + c)

    def get_circuits(self):
        return self._circuit_draw

    def update_model(self, **param_dict):
        if self._shots_model is not None:
            tolerance, self._shots = self._shots_model(param_dict)
            print('shots_update: ', self._shots)

            return tolerance

        return param_dict['precision']

    # mock methods (to be implemented in subclasses)
    def _measure_expectation(self, *args):
        raise NotImplementedError()

    def _measure_expectation_variance(self, *args):
        raise NotImplementedError()

    def _get_state_vector(self, *args):
        raise NotImplementedError()

    def _get_matrix_operator_gates(self, *args):
        raise NotImplementedError()

    def _build_reference_gates(self, *args, **kwargs):
        raise NotImplementedError()

    def _trotterize_operator(self, *args):
        raise NotImplementedError()

    def simulator_info(self, *args):
        raise NotImplementedError()
