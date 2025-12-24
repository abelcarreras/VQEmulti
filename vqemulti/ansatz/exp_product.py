from vqemulti.utils import get_sparse_ket_from_fock, get_sparse_operator
from openfermion.utils import count_qubits
from vqemulti.pool.tools import OperatorList
from vqemulti.utils import fermion_to_qubit
from vqemulti.ansatz import GenericAnsatz
from vqemulti.simulators.qiskit_simulator import QiskitSimulator
from openfermion import QubitOperator, is_hermitian
import numpy as np
import scipy as sp


class ProductExponentialAnsatz(GenericAnsatz):
    """
    ansatz type: Sum(e^O_i)
    """
    def __init__(self, parameters, operator_list: OperatorList | list, reference_fock : list):
        """
        :param parameters: list of parameters
        :param operator_list: list of operators
        :param reference_fock: non-entangled initial state as Fock space vector
        """
        super().__init__()
        self._operators = OperatorList(operator_list)
        self._parameters = list(parameters)
        self._reference_fock = reference_fock

        assert len(parameters) == len(operator_list)

        # check antihermiticity for each operator
        for op in operator_list:
            if not is_hermitian(1j * op):
                raise Exception('Non antihermitian operator')

    @property
    def n_qubits(self):
        return len(self._reference_fock)

    @property
    def operators(self):
        return self._operators

    def add_operator(self, operator, parameter):
        self._operators.append(operator)
        self._parameters.append(parameter)

    def _exact_energy(self, hamiltonian, return_std=False):
        """
        Calculates the energy of the state prepared by applying an ansatz (of the
        type of the VQE protocol) to a reference state.

        :param hamiltonian: Hamiltonian in FermionOperator/InteractionOperator
        :return: exact energy
        """

        # get sparse hamiltonian
        sparse_hamiltonian = get_sparse_operator(hamiltonian, self.n_qubits)

        # Transform reference vector into a Compressed Sparse Column matrix
        ket = get_sparse_ket_from_fock(self._reference_fock)

        # use trotterized operators (for adaptVQE)
        for coefficient, operator in zip(self._parameters, self._operators):
            # Get the operator matrix representation of the operator
            sparse_operator = coefficient * get_sparse_operator(operator, self.n_qubits)

            # Apply e ** (coefficient * operator) to the state (ket) for each operator in
            ket = sp.sparse.linalg.expm_multiply(sparse_operator, ket)

        # Get the corresponding bra and calculate the energy: |<bra| H |ket>|
        bra = ket.transpose().conj()
        energy = np.sum(bra * sparse_hamiltonian * ket).real

        if return_std:
            return energy, 0.0

        return energy

    def _simulate_energy(self, hamiltonian, simulator, return_std=False):
        """
        Obtain the hamiltonian expectation value for a given VQE state (reference + ansatz) and a hamiltonian

        :param hamiltonian: hamiltonian in FermionOperator/InteractionOperator
        :param simulator: simulation object
        :param return_std: return std also
        :return: the expectation value of the Hamiltonian in the current state (HF ref + ansatz)
        """

        # transform to qubit hamiltonian
        qubit_hamiltonian = fermion_to_qubit(hamiltonian)

        # get gates to prepare the state
        state_preparation_gates = self.get_preparation_gates(simulator)

        # evaluate hamiltonian
        energy, std_error = simulator.get_state_evaluation(qubit_hamiltonian, state_preparation_gates)

        if return_std:
            return energy, std_error

        return energy

    def _exact_gradient(self, hamiltonian):
        """
        Calculates the gradient vector of the energy with respect to the coefficients for adapt VQE Wave function.

        :param hamiltonian: Hamiltonian in FermionOperator/InteractionOperator
        :return: gradient vector
        """

        # Transform Hamiltonian to matrix representation
        sparse_hamiltonian = get_sparse_operator(hamiltonian, self.n_qubits)

        # Transform reference vector into a Compressed Sparse Column matrix
        ket = get_sparse_ket_from_fock(self._reference_fock)

        # Apply e ** (coefficient * operator) to the state (ket) for each operator in
        # the ansatz, following the order of the list
        for j, (coefficient, operator) in enumerate(zip(self._parameters, self._operators)):
            # Get the operator matrix representation of the operator
            sparse_operator = coefficient * get_sparse_operator(operator, self.n_qubits)

            # Exponentiate the operator and update ket t
            ket = sp.sparse.linalg.expm_multiply(sparse_operator, ket)

        bra = ket.transpose().conj()
        hbra = bra.dot(sparse_hamiltonian)

        gradient_vector = []

        def recurse(hbra, ket, term):

            if term > 0:
                operator = self._parameters[-term] * get_sparse_operator(self._operators[-term], self.n_qubits)
                hbra = (sp.sparse.linalg.expm_multiply(-operator, hbra.transpose().conj())).transpose().conj()
                ket = sp.sparse.linalg.expm_multiply(-operator, ket)

            operator = get_sparse_operator(self._operators[-(term + 1)], self.n_qubits)
            gradient_vector.insert(0, 2 * hbra.dot(operator).dot(ket)[0, 0].real)

            if term < len(self._parameters) - 1:
                recurse(hbra, ket, term + 1)

        recurse(hbra, ket, 0)

        return gradient_vector

    def _simulate_gradient(self, hamiltonian, simulator):
        """
        Calculates the gradient of the energy with respect to the coefficients for VQE/adaptVQE Wave function using simulator.
        To be used as a gradient function in the energy minimization

        :param coefficients: the list of coefficients of the ansatz operators
        :param ansatz: ansatz expressed in qubit/fermion operators
        :param hamiltonian: Hamiltonian in FermionOperator/InteractionOperator
        :param simulator: simulation object
        :return: gradient vector
        """

        # transform to qubit hamiltonian
        qubit_hamiltonian = fermion_to_qubit(hamiltonian)

        operators_list = OperatorList(self._operators)
        ansatz_qubit = operators_list.transform_to_scaled_qubit(self._parameters)

        state_preparation_gates = simulator.get_reference_gates(self._reference_fock)
        state_preparation_gates += simulator.get_exponential_gates(ansatz_qubit, self.n_qubits)

        # Calculate and print gradients
        gradient_vector = []
        for i, operator in enumerate(self._operators):

            # convert to qubit if necessary
            if not isinstance(operator, QubitOperator):
                operator = fermion_to_qubit(operator)

            # get gradient as dexp(c * A) / dc = < psi | [H, A] | psi >.
            commutator_hamiltonian = qubit_hamiltonian * operator - operator * qubit_hamiltonian

            sampled_gradient, variance = simulator.get_state_evaluation(commutator_hamiltonian, state_preparation_gates)

            gradient_vector.append(sampled_gradient)

        return gradient_vector

    def get_state_vector(self):
        """
        prepare state vector from coefficients ansatz and reference

        :return state vector
        """

        # Initialize the state vector with the reference state.
        # |state> = |state_old> · exp(i * coef · |ansatz>
        # imaginary i already included in |ansatz>
        state = get_sparse_ket_from_fock(self._reference_fock)

        # Apply the ansatz operators one by one to obtain the state as optimized by the last iteration
        for coefficient, operator in zip(self._parameters, self._operators):
            sparse_operator = coefficient * get_sparse_operator(operator, self.n_qubits)
            state = sp.sparse.linalg.expm_multiply(sparse_operator, state)
        return state

    def get_preparation_gates(self, simulator):

        # transform ansatz to qubit for adaptVQE (coefficients are included in qubits objects)
        operators_list = OperatorList(self._operators)
        ansatz_qubit = operators_list.transform_to_scaled_qubit(self._parameters)

        # get the gates to prepare the state
        state_preparation_gates = simulator.get_reference_gates(self._reference_fock)
        state_preparation_gates += simulator.get_exponential_gates(ansatz_qubit, self.n_qubits)

        return state_preparation_gates

    def pool_gradient_vector(self, hamiltonian, pool, simulator):

        if simulator is None:
            return self._exact_pool_gradient_vector(hamiltonian, pool)
        else:
            return self._simulate_pool_gradient_vector(hamiltonian, pool, simulator)

    def _exact_pool_gradient_vector(self, hamiltonian, pool):
        """
        computes the gradient vector of the energy with respect to a pool of operators

        :param hf_reference_fock: reference HF state in Fock space vector
        :param hamiltonian: Hamiltonian in FermionOperator/InteractionOperator
        :param ansatz: VQE ansatz in qubit operators
        :param coefficients: list of VQE coefficients
        :param pool: pool of qubit operators
        :return: the gradient vector
        """

        # transform hamiltonian to sparse
        sparse_hamiltonian = get_sparse_operator(hamiltonian, self.n_qubits)

        # Prepare the current state from ansatz (& coefficient) and HF reference
        sparse_state = self.get_state_vector()

        # Calculate and print gradients
        # print('pool size: ', len(pool))
        print("\nNon-Zero Gradients (exact)")
        gradient_vector = []
        for i, operator in enumerate(pool):
            sparse_operator = get_sparse_operator(operator, self.n_qubits)

            # gradient = 2 * <state | H · Op | state >  (non-explicit)
            bra = sparse_state.transpose().conj()
            ket = sparse_operator.dot(sparse_state)
            gradient = 2 * np.abs(bra * sparse_hamiltonian * ket)[0, 0].real

            if gradient > 1e-5:
                print("Operator {}: {:.6f}".format(i, gradient))

            gradient_vector.append(gradient)

        return gradient_vector

    def _simulate_pool_gradient_vector(self, hamiltonian, pool, simulator):
        """
        Calculates the gradient of the energy with respect to adding new operator from a pool.
        To be used in adaptVQE poll gradients calculation  using simulator

        :param hf_reference_fock: reference HF state in Fock space vector
        :param hamiltonian: hamiltonian in qubit operators
        :param ansatz: VQE ansatz in qubit/Fermion operators
        :param coefficients: list of VQE coefficients
        :param pool: pool of qubit operators
        :param simulator: simulation object
        :return: the gradient_vector
        """

        # transform to qubit hamiltonian
        qubit_hamiltonian = fermion_to_qubit(hamiltonian)

        # get gates to prepare the state
        state_preparation_gates = self.get_preparation_gates(simulator)

        if simulator._test_only:
            print('Non-Zero Gradients (Exact circuit evaluation)')
        else:
            print('Non-Zero Gradients (Simulated with {} shots)'.format(simulator._shots))

        # Calculate and print gradients
        gradient_vector = []
        for i, operator in enumerate(pool):

            # convert to qubit if necessary
            if not isinstance(operator, QubitOperator):
                operator = fermion_to_qubit(operator)

            # get gradient as dexp(c * A) / dc = < psi | [H, A] | psi >.
            commutator_hamiltonian = qubit_hamiltonian * operator - operator * qubit_hamiltonian

            sampled_gradient, variance = simulator.get_state_evaluation(commutator_hamiltonian, state_preparation_gates)

            # set absolute value for gradient (sign is not important, only magnitude)
            sampled_gradient = np.abs(sampled_gradient)
            gradient_vector.append(sampled_gradient)

            if sampled_gradient > 1e-6:
                print("Operator {}: {:.6f}".format(i, sampled_gradient))
                # just for testing
                # print_comparison_gradient_analysis(qubit_hamiltonian, hf_reference_fock, ansatz_qubit, operator, sampled_gradient)

        return gradient_vector

    def get_sampling(self, simulator):

        state_preparation_gates = self.get_preparation_gates(simulator)
        sampling = simulator.get_state_sampling(state_preparation_gates, self.n_qubits)

        return sampling



if __name__ == '__main__':

    from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
    from openfermionpyscf import run_pyscf
    from openfermion import MolecularData
    from vqemulti.utils import get_hf_reference_in_fock_space
    from vqemulti.operators import n_particles_operator, spin_z_operator, spin_square_operator
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    from qiskit_aer import AerSimulator

    from vqemulti.preferences import Configuration
    #config = Configuration()
    #config.verbose = 2

    simulator = Simulator(trotter=False,
                          trotter_steps=1,
                          test_only=True,
                          hamiltonian_grouping=True,
                          use_estimator=True, shots=10000,
                          # backend=FakeTorino(),
                          # use_ibm_runtime=True
                          )

    simulator_sqd = simulator.copy()
    simulator_sqd._backend = AerSimulator()
    simulator_sqd._use_ibm_runtime = True


    hydrogen = MolecularData(geometry=[('H', [0.0, 0.0, 0.0]),
                                       ('H', [2.0, 0.0, 0.0]),
                                       ('H', [4.0, 0.0, 0.0]),
                                       ('H', [6.0, 0.0, 0.0])],
                             basis='sto-3g',
                             multiplicity=1,
                             charge=0,
                             description='molecule')

    # run classical calculation
    n_frozen_orb = 0 # nothing
    n_total_orb = 4 # total orbitals
    molecule = run_pyscf(hydrogen, run_fci=False, nat_orb=False, guess_mix=False, verbose=True,
                         frozen_core=n_frozen_orb, n_orbitals=n_total_orb, run_ccsd=True)

    tol_ampl = 0.01

    from pyscf.fci import cistring
    mc = molecule._pyscf_data['casci']

    ncas = mc.ncas
    nelec = mc.nelecas

    # determinants α i β (representats com enters amb bits d’ocupació)
    na, nb = nelec
    alpha_det = cistring.make_strings(range(ncas), na)
    beta_det = cistring.make_strings(range(ncas), nb)


    def interleave_bits(a, b, ncas):
        """Return interleaved occupation string (αβ αβ ...)"""
        a_bits = [(a >> i) & 1 for i in range(ncas)]
        b_bits = [(b >> i) & 1 for i in range(ncas)]
        inter = []
        for i in reversed(range(ncas)):
            inter.append(str(a_bits[i]))
            inter.append(str(b_bits[i]))
        return ''.join(inter)[::-1]

    print('\namplitudes CASCI')
    for i, a in enumerate(alpha_det):
        for j, b in enumerate(beta_det):
            amp = mc.ci[i, j]
            if amp**2 > tol_ampl:
                # print(f"α={format(a, f'0{ncas}b')}  β={format(b, f'0{ncas}b')}  coef={amp:+.6f}")
                cfg = interleave_bits(a, b, ncas)
                print(f"{cfg}   {amp:+.6f}  ({amp**2:.6f}) ")

    hamiltonian = molecule.get_molecular_hamiltonian()

    n_electrons = molecule.n_electrons - n_frozen_orb * 2
    n_orbitals = n_total_orb - n_frozen_orb  # molecule.n_orbitals
    n_qubits = n_orbitals * 2
    print('n_qubits: ', n_qubits)

    hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, n_qubits)

    print('\nUCC ansatz\n==========')
    from vqemulti.ansatz.generators import get_ucc_generator
    generator = get_ucc_generator(None, molecule.ccsd_double_amps, use_qubit=False)


    from vqemulti.pool import get_pool_qubit_sd, get_pool_singlet_sd
    coefficients, generator = [1.2, 1.6, 1.8], get_pool_singlet_sd(n_electrons, n_orbitals).get_quibits_list(normalize=True)[-3:]
    #coefficients, generator = [1.2, 1.6, 1.8], get_pool_qubit_sd(n_electrons, n_orbitals)[:3]
    #coefficients, generator = [1.0], get_pool_qubit_sd(n_electrons, n_orbitals)[:1]
    print(generator)

    #pool = get_pool_singlet_sd(n_electrons, n_orbitals).get_quibits_list(normalize=True)
    ansatz = ProductExponentialAnsatz([], [], hf_reference_fock)

    ansatz.add_operator(generator[0], 1.2)
    print('energy 1: ', ansatz.get_energy(ansatz.parameters, hamiltonian, None))

    ansatz.add_operator(generator[1], 1.6)
    print('energy 2: ', ansatz.get_energy(ansatz.parameters, hamiltonian, None))

    ansatz.add_operator(generator[2], 1.8)
    print('energy 3: ', ansatz.get_energy(ansatz.parameters, hamiltonian, None))

    #coefficients = generator.get_quibits_list().operators_prefactors()
    #generator = generator.get_quibits_list(normalize=True)
    print(len(coefficients), len(generator))

    # ansatz = ProductExponentialAnsatz(coefficients, generator, hf_reference_fock)
    print('energy: ', ansatz.get_energy(ansatz.parameters, hamiltonian, None))

    print('simulator')
    simulator = QiskitSimulator(trotter=False, trotter_steps=1000, test_only=True, use_estimator=True)

    ansatz = ProductExponentialAnsatz(coefficients, generator, hf_reference_fock)

    print('energy SIM: ', ansatz.get_energy(ansatz.parameters, hamiltonian, simulator))
    print('energy Exact: ', ansatz.get_energy(ansatz.parameters, hamiltonian, None))

    print('energy gradients SIM: ', ansatz.get_gradients(ansatz.parameters, hamiltonian, simulator))
    print('energy gradients Exact: ', ansatz.get_gradients(ansatz.parameters, hamiltonian, None))

    #print(simulator.get_circuits()[0])
    from vqemulti.vqe import vqe

    print(ansatz.parameters)
    result = vqe(hamiltonian, ansatz, energy_simulator=simulator)
    print(ansatz.parameters)
    print(result)

    ansatz_opt = ProductExponentialAnsatz(result['coefficients'], generator, hf_reference_fock)

    print('energy SIM: ', ansatz.get_energy(ansatz.parameters, hamiltonian, simulator))
    print('energy Exact: ', ansatz.get_energy(ansatz.parameters, hamiltonian, None))

    print('gradient exact: ', ansatz_opt.get_gradients(ansatz.parameters, hamiltonian, None))
    print('gradient Simulation: ', ansatz_opt.get_gradients(ansatz.parameters, hamiltonian, simulator))

    sampling = ansatz_opt.get_sampling(simulator_sqd)
    print('sampling: ', sampling)
