from vqemulti.ansatz.generators import get_ucc_generator
from vqemulti.utils import get_sparse_operator
from copy import deepcopy
import numpy as np


def crop_local_amplitudes(amplitudes, n_neighbors=1):
    """
    remove interactions in the amplitudes between orbitals at larger distance than the set distance

    :param amplitudes: amplitudes matrix (occ x virt) / (occ x occ x virt x virt )
    :param n_neighbors: distance between orbitals
    :return: local amplitudes
    """

    if len(np.shape(amplitudes)) == 2:
        # a_j a_i^
        nocc, nvrt = amplitudes.shape
        local_amplitudes = np.array(amplitudes).copy()  # a_j a_i^
        for i in range(nocc):
            local_amplitudes[i, max(0, n_neighbors - nocc + i+1):] = 0

    elif len(np.shape(amplitudes)) == 4:
        # a_j a_l a_i^ a_k^
        # double electron amplitudes
        nocc, nvrt = amplitudes.shape[1:3]

        local_amplitudes = np.array(amplitudes).copy()  # a_j a_l a_i^ a_k^

        for i in range(nocc):
            # strict criteria, all pairs (i, j) (k, l) within distance
            local_amplitudes[i, :, max(0, n_neighbors - nocc + i + 1):, :] = 0
            local_amplitudes[:, i, :, max(0, n_neighbors - nocc + i + 1):] = 0

    else:
        raise Exception('Amplitudes format not accepted')

    return local_amplitudes


class GenericAnsatz:

    def __init__(self):
        self._operators = []
        self._parameters = []

    def __len__(self):
        return len(self._parameters)

    @property
    def n_qubits(self):
        return 0

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        self._parameters = list(parameters)

    def copy(self):
        return deepcopy(self)

    def get_energy(self, parameters, hamiltonian, energy_simulator, return_std=False):
        self._parameters = parameters
        if energy_simulator is None:
            return self._exact_energy(hamiltonian, return_std)
        else:
            return self._simulate_energy(hamiltonian, energy_simulator, return_std)

    def _exact_energy(self, hamiltonian, return_std=False):
        """
        Calculates the energy of the state prepared by applying an ansatz (of the
        type of the VQE protocol) to a reference state.

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

    def _simulate_energy(self, hamiltonian, energy_simulator, return_std):
        raise NotImplemented()

    def get_state_vector(self):
        raise NotImplemented()

    def get_gradients(self, parameters, hamiltonian, simulator):
        self._parameters = parameters
        if simulator is None:
            return self._exact_gradient(hamiltonian)
        else:
            return self._simulate_gradient(hamiltonian, simulator)

    def get_matrix_representation(self):
        raise NotImplemented()

    def _exact_gradient(self, hamiltonian):
        return self._simulate_gradient(hamiltonian, None)

    def _simulate_gradient(self, hamiltonian, simulator):
        return self._numerical_gradient(hamiltonian, simulator)

    def _numerical_gradient(self, hamiltonian, simulator):
        """
            Calculates the gradient numerically using central finite differences.

            :param coefficients: Current variational parameters
            :param epsilon: Step size for the numerical derivative
            :return: Gradient vector (numpy array)
        """

        epsilon = 1e-6

        def get_energy(parameters):
            cache_param = self._parameters
            energy = self.get_energy(parameters, hamiltonian, simulator)
            self._parameters = cache_param
            return energy

        # We use a copy to avoid modifying the original parameters during the loop
        n_params = len(self._parameters)
        gradient = np.zeros(n_params)
        params_copy = np.array(self._parameters, dtype=float)

        for i in range(n_params):
            # Save original value
            original_val = params_copy[i]

            # Calculate f(theta + epsilon)
            params_copy[i] = original_val + epsilon
            energy_plus = get_energy(params_copy)

            # Calculate f(theta - epsilon)
            params_copy[i] = original_val - epsilon
            energy_minus = get_energy(params_copy)

            # Central difference formula: [f(x+h) - f(x-h)] / (2h)
            gradient[i] = (energy_plus - energy_minus) / (2 * epsilon)

            # Restore original value for the next parameter
            params_copy[i] = original_val

        return gradient


if __name__ == '__main__':

    from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
    from openfermionpyscf import run_pyscf
    from openfermion import MolecularData
    from vqemulti.utils import get_hf_reference_in_fock_space
    # from vqemulti.energy import get_vqe_energy, get_adapt_vqe_energy
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

    simulator_jastrow = simulator.copy()
    #simulator_jastrow._backend = FakeTorino()
    #simulator_jastrow._use_ibm_runtime = True
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
    from openfermion import QubitOperator

    # hamiltonian = QubitOperator('Z0 Z1 Z4 Z5 Z7')
    # hamiltonian = QubitOperator('Z0 Z2 Z3 Z6 Z7')

    # print(hamiltonian)

    n_electrons = molecule.n_electrons - n_frozen_orb * 2
    n_orbitals = n_total_orb - n_frozen_orb  # molecule.n_orbitals
    n_qubits = n_orbitals * 2
    print('n_qubits: ', n_qubits)

    hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, n_qubits)

    print('\nUCC ansatz\n==========')
    uccsd_generator = get_ucc_generator(molecule.ccsd_single_amps, molecule.ccsd_double_amps, use_qubit=True)
    coefficients = np.ones_like(uccsd_generator)

    from vqemulti.ansatz.exponential import ExponentialAnsatz
    uccsd_ansatz = ExponentialAnsatz(coefficients, uccsd_generator, hf_reference_fock)

    energy = uccsd_ansatz.get_energy(uccsd_ansatz.parameters, hamiltonian, simulator)

    simulator.print_statistics()
    print('energy: ', energy)
    print(simulator.get_circuits()[-1])

    n_particle = uccsd_ansatz.get_energy(uccsd_ansatz.parameters, n_particles_operator(uccsd_ansatz.n_qubits//2), None)
    spin_z = uccsd_ansatz.get_energy(uccsd_ansatz.parameters, spin_z_operator(uccsd_ansatz.n_qubits//2), None)
    spin_square = uccsd_ansatz.get_energy(uccsd_ansatz.parameters, spin_square_operator(uccsd_ansatz.n_qubits//2), None)

    print('n_particles: {:8.4f}'.format(n_particle))
    print('Sz: {:8.4f}'.format(spin_z))
    print('S2: {:8.4f}'.format(spin_square))

    print('\nJASTROW ansatz\n==============')
    from vqemulti.ansatz.unitary_jastrow import UnitaryCoupledJastrowAnsatz

    ccsd = molecule._pyscf_data.get('ccsd', None)
    t2 = crop_local_amplitudes(ccsd.t2, n_neighbors=3)
    t1 = ccsd.t1

    ucja = UnitaryCoupledJastrowAnsatz(None, t2, n_terms=1, full_trotter=True)

    energy = ucja.get_energy(ucja.parameters, hamiltonian, simulator)

    simulator.print_statistics()
    print(simulator.get_circuits()[-1])

    print('Jastrow energy: ', energy)

    from vqemulti.vqe import vqe
    print(vqe(hamiltonian, ucja, energy_simulator=None))

    energy = ucja.get_energy(ucja.parameters, hamiltonian, simulator)
    print('Optimized Jastrow energy: ', energy)

    n_particle = ucja.get_energy(ucja.parameters, n_particles_operator(ucja.n_qubits//2), None)
    spin_z = ucja.get_energy(ucja.parameters, spin_z_operator(ucja.n_qubits//2), None)
    spin_square = ucja.get_energy(ucja.parameters, spin_square_operator(ucja.n_qubits//2), None)

    print('n_particles: {:8.4f}'.format(n_particle))
    print('Sz: {:8.4f}'.format(spin_z))
    print('S2: {:8.4f}'.format(spin_square))


    def get_projections(n_qubits, n_electrons, tolerance=0.01):

        from vqemulti.operators import configuration_projector_operator
        from vqemulti.utils import configuration_generator

        print('\nAmplitudes square')
        amplitudes_square = []
        for configuration in configuration_generator(n_qubits, n_electrons):
            amplitude2 = ucja.get_energy(ucja.parameters, configuration_projector_operator(configuration), None)

            amplitudes_square.append(amplitude2)

            if abs(amplitude2) > tolerance:
                print('{} {:8.5f} '.format(''.join([str(s) for s in configuration]), amplitude2.real))

        print('sum amplitudes square: ', np.sum(amplitudes_square))
        return amplitudes_square

    get_projections(ucja.n_qubits, n_electrons, tolerance=tol_ampl)

    # SQD
    from vqemulti.sqd import simulate_energy_sqd
    energy, samples = simulate_energy_sqd(ucja,
                                          hamiltonian,
                                          simulator_sqd,
                                          n_electrons)

    print(samples)
    print('SQD energy', energy)
