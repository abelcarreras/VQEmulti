from vqemulti.utils import get_sparse_ket_from_fock, get_sparse_operator
from openfermion.utils import count_qubits
from vqemulti.pool.tools import OperatorList
from vqemulti.utils import fermion_to_qubit
from vqemulti.ansatz import GenericAnsatz
from vqemulti.simulators.qiskit_simulator import QiskitSimulator
from openfermion import QubitOperator, is_hermitian
import numpy as np
import scipy as sp


class ExponentialAnsatz(GenericAnsatz):
    """
    ansatz type: e^Sum(A_i)
    """
    def __init__(self, parameters, operator_list: OperatorList | list, reference_fock: list):
        """
        :param parameters: list of parameters
        :param operator_list: list of operators
        :param reference_fock: non-entangled initial state as Fock space vector
        """
        super().__init__()
        self._operators = operator_list
        self._parameters = parameters
        self._reference_fock = reference_fock

        assert len(parameters) == len(operator_list)

        if len(operator_list) > 0 and not is_hermitian(1j * sum(operator_list)):
            raise Exception('Non antihermitian operator')

    @property
    def n_qubits(self):
        return len(self._reference_fock)

    @property
    def operators(self):
        return self._operators

    def _exact_energy(self, hamiltonian, return_std=False):
        """
        Calculates the energy of the state prepared by applying an ansatz (of the
        type of the VQE protocol) to a reference state.

        :param hf_reference_fock: HF reference in Fock space vector
        :param hamiltonian: Hamiltonian in FermionOperator/InteractionOperator
        :return: exact energy
        """

        # get sparse hamiltonian
        sparse_hamiltonian = get_sparse_operator(hamiltonian, self.n_qubits)

        # get state vector
        ket = self.get_state_vector()

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
        :param trotterize: use trotterized wave function (adaptVQE style)
        :param return_std: return std also
        :return: the expectation value of the Hamiltonian in the current state (HF ref + ansatz)
        """

        # transform to qubit hamiltonian
        qubit_hamiltonian = fermion_to_qubit(hamiltonian)

        # get gates to prepare the state
        state_preparation_gates = self.get_preparation_gates(simulator)

        energy, std_error = simulator.get_state_evaluation(qubit_hamiltonian, state_preparation_gates)

        if return_std:
            return energy, std_error

        return energy

    def _exact_gradient(self, hamiltonian):
        """
        Calculates the EXACT gradient vector of the energy using the augmented state
        method for the derivative of the matrix exponential.

        :param hf_reference_fock: HF reference in Fock space vector
        :param hamiltonian: Hamiltonian in FermionOperator/InteractionOperator
        :return: gradient vector (list of floats)
        """

        hf_reference_fock = self._reference_fock
        n_qubit = self.n_qubits  # Assumed stored in the class
        dim = 2 ** n_qubit
        sparse_hamiltonian = get_sparse_operator(hamiltonian, n_qubit)
        ket_hf = get_sparse_ket_from_fock(hf_reference_fock)
        # 1. Construir el generador G total (complex!)
        g_generator = sp.sparse.csr_array((dim, dim), dtype=complex)
        for operator, coefficient in zip(self._operators, self._parameters):
            g_generator += coefficient * get_sparse_operator(operator, n_qubit)

        # 2. Calcular l'estat actual |Psi> = exp(G)|HF>
        #ket_hf = get_sparse_ket_from_fock(hf_reference_fock)
        ket_psi = sp.sparse.linalg.expm_multiply(g_generator, ket_hf)
        # Bra-side per a l'energia: <Psi|H
        bra_psi_h = ket_psi.transpose().conj() @ sparse_hamiltonian

        gradient = []

        # 3. Iterar per cada paràmetre c_k
        for a_term in self._operators:
            a_k = get_sparse_operator(a_term, n_qubit)

            # --- CONSTRUCCIÓ DE LA MATRIU AUGMENTADA M ---
            # M = [[G, Ak],
            #      [0, G ]]
            # Use bmat to create the sparse block matrix efficiently
            m_matrix = sp.sparse.bmat([
                [g_generator, a_k],
                [None, g_generator]
            ], format='csr', dtype=complex)

            # --- PREPARACIÓ DEL VECTOR AUGMENTAT ---
            # Vec = [ 0, |HF> ]
            augmented_vec = np.zeros(2 * dim, dtype=complex)
            # Fix: Ensure we flatten or convert to array properly
            augmented_vec[dim:] = ket_hf.toarray().ravel()

            # 4. expm_multiply amb la matriu esparsa M (això ja no donarà errors)
            # També podem passar la traça per evitar el warning: Trace(M) = 2 * Trace(G)
            trace_m = 2 * g_generator.diagonal().sum()

            res_augmented = sp.sparse.linalg.expm_multiply(m_matrix, augmented_vec, traceA=trace_m)

            # El bloc superior del resultat és (d_exp_G / dc_k) |HF>
            d_ket_psi = res_augmented[:dim]

            # dE/dc_k = 2 * Re(<Psi| H |d_Psi>)
            term = bra_psi_h @ d_ket_psi
            gradient.append(2 * term[0].real)

        return gradient

    def get_state_vector(self):
        """
        prepare state vector from coefficients ansatz and reference

        :return state vector
        """

        ref_state = get_sparse_ket_from_fock(self._reference_fock)

        exponent = sp.sparse.csr_array((2 ** self.n_qubits, 2 ** self.n_qubits), dtype=float)
        for coefficient, operator in zip(self._parameters, self._operators):
            exponent += coefficient * get_sparse_operator(operator, self.n_qubits)

        state = sp.sparse.linalg.expm_multiply(exponent, ref_state)

        return state

    def get_preparation_gates(self, simulator):
        """
        get the gates to prepare the state

        :param simulator: simulator for which the gates will be preparated
        :return: gates list
        """

        operators_list = OperatorList(self._operators)
        ansatz_qubit = operators_list.transform_to_scaled_qubit(self._parameters, join=True)
        state_preparation_gates = simulator.get_reference_gates(self._reference_fock)
        state_preparation_gates += simulator.get_exponential_gates(ansatz_qubit, self.n_qubits)

        return state_preparation_gates

    def get_sampling(self, simulator):

        state_preparation_gates = self.get_preparation_gates(simulator)
        sampling = simulator.get_state_sampling(state_preparation_gates, self.n_qubits)

        return sampling


if __name__ == '__main__':

    from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
    from openfermionpyscf import run_pyscf
    from openfermion import MolecularData
    from vqemulti.utils import get_hf_reference_in_fock_space
    from vqemulti.energy import get_vqe_energy, get_adapt_vqe_energy
    from vqemulti.operators import n_particles_operator, spin_z_operator, spin_square_operator
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    from qiskit_aer import AerSimulator

    from vqemulti.preferences import Configuration
    # config = Configuration()
    # config.verbose = 2

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
    coefficients, generator = get_ucc_generator(None, molecule.ccsd_double_amps, use_qubit=False)

    from vqemulti.pool import get_pool_qubit_sd, get_pool_singlet_sd
    coefficients, generator = [1.2, 1.6, 1.8], get_pool_singlet_sd(n_electrons, n_orbitals).get_quibits_list(normalize=True)[-3:]
    # coefficients, generator = [1.2, 1.6, 1.8], get_pool_qubit_sd(n_electrons, n_orbitals)[:3]
    # coefficients, generator = [1.0], get_pool_qubit_sd(n_electrons, n_orbitals)[:1]

    #coefficients = generator.get_quibits_list().operators_prefactors()
    #generator = generator.get_quibits_list(normalize=True)

    ansatz = ExponentialAnsatz(coefficients, generator, hf_reference_fock)

    hf_energy = ansatz.get_energy(ansatz.parameters, hamiltonian, None)
    print('energy HF: ', hf_energy)

    hf_energy = get_vqe_energy(coefficients, generator, hf_reference_fock, hamiltonian, None)

    print('energy HF: ', hf_energy)

    print('simulator')
    simulator = QiskitSimulator(trotter=False, trotter_steps=1000, test_only=True, use_estimator=True)
    #from vqemulti.gradient import simulate_vqe_energy_gradient

    #gradient = simulate_vqe_energy_gradient(coefficients, generator, hf_reference_fock, hamiltonian, simulator)
    #print('gradient OG: ', gradient)

    ansatz = ExponentialAnsatz(coefficients, generator, hf_reference_fock)

    print('energy SIM: ', ansatz.get_energy(ansatz.parameters, hamiltonian, simulator))
    print('energy Exact: ', ansatz.get_energy(ansatz.parameters, hamiltonian, None))

    simulator.print_circuits()

    print('energy gradients SIM: ', ansatz.get_gradients(ansatz.parameters, hamiltonian, simulator))
    print('energy gradients Exact: ', ansatz.get_gradients(ansatz.parameters, hamiltonian, None))

    # print(simulator.get_circuits()[0])
    # exit()
    from vqemulti.vqe import vqe

    print(ansatz.parameters)
    result = vqe(hamiltonian, ansatz, energy_simulator=simulator)
    print(ansatz.parameters)
    print(result)

    # hf_energy = get_vqe_energy(coefficients, generator, hf_reference_fock, hamiltonian, simulator)
    # print('energy HF: ', hf_energy)

    ansatz_opt = ExponentialAnsatz(result['coefficients'], generator, hf_reference_fock)

    print('energy SIM: ', ansatz.get_energy(ansatz.parameters, hamiltonian, simulator))
    print('energy Exact: ', ansatz.get_energy(ansatz.parameters, hamiltonian, None))

    print('gradient exact: ', ansatz_opt.get_gradients(ansatz.parameters, hamiltonian, None))
    print('gradient Simulation: ', ansatz_opt.get_gradients(ansatz.parameters, hamiltonian, simulator))

    #print('gradient Sim OG', simulate_vqe_energy_gradient(result['coefficients'], generator, hf_reference_fock, hamiltonian, simulator))

    sampling = ansatz_opt.get_sampling(simulator)
    print('sampling: ', sampling)
