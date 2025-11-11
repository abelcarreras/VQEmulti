import matplotlib.pyplot as plt

from vqemulti.utils import get_hf_reference_in_fock_space, generate_reduced_hamiltonian
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.adapt_vqe import adaptVQE
from vqemulti.energy import get_adapt_vqe_energy
from vqemulti.energy.simulation import simulate_adapt_vqe_variance, simulate_energy_sqd
from vqemulti.preferences import Configuration
from vqemulti.symmetry import get_symmetry_reduced_pool, symmetrize_molecular_orbitals
from openfermionpyscf import run_pyscf
from vqemulti.errors import NotConvergedError
from vqemulti.basis_projection import get_basis_overlap_matrix, project_basis, prepare_ansatz_for_restart
from vqemulti.density import get_density_matrix, density_fidelity
from vqemulti.optimizers import OptimizerParams
from vqemulti.optimizers import adam, rmsprop, sgd, cobyla_mod
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import Session
from openfermion import MolecularData, get_sparse_operator, get_fermion_operator
from vqemulti.gradient import compute_gradient_vector, simulate_gradient
from noise_model import get_noise_model_thermal, get_noise_model_bitflip
import numpy as np
import warnings


def get_n_particles_operator(n_orbitals, to_pauli=False):
    from openfermion import FermionOperator, jordan_wigner

    sz = FermionOperator()
    for i in range(0, n_orbitals * 2):
        a = FermionOperator((i, 1))
        a_dag = FermionOperator((i, 0))
        sz += a * a_dag

    if to_pauli:
        sz = jordan_wigner(sz)
    return sz


def expand_interaction_operator(hamiltonian, new_n_qubits):
    from openfermion import InteractionOperator

    old_n_qubits = hamiltonian.n_qubits

    if new_n_qubits < old_n_qubits:
        raise ValueError("New number of qubits must be greater than or equal to the current one.")

    # Inicialitzem un InteractionOperator buit amb més qubits
    expanded = InteractionOperator(
        constant=hamiltonian.constant,
        one_body_tensor=np.ones((new_n_qubits, new_n_qubits))*1e-8,
        two_body_tensor=np.zeros((new_n_qubits, new_n_qubits, new_n_qubits, new_n_qubits)),
    )

    # Copiem les components existents
    expanded.one_body_tensor[:old_n_qubits, :old_n_qubits] = hamiltonian.one_body_tensor
    expanded.two_body_tensor[:old_n_qubits, :old_n_qubits, :old_n_qubits, :old_n_qubits] = hamiltonian.two_body_tensor

    return expanded


def simulate_configurations_sqd(coefficients, ansatz, hf_reference_fock, hamiltonian, simulator, n_electrons,
                                multiplicity=0,
                                generate_random=False,
                                backend='dice',
                                adapt=False):
    """
    Obtain the hamiltonian expectation value with SQD using a given adaptVQE state as reference.
    Only compatible with JW mapping!!

    :param coefficients: adaptVQE coefficients
    :param ansatz: ansatz expressed in qubit/fermion operators
    :param hf_reference_fock: reference HF in fock vspace vector
    :param hamiltonian: hamiltonian in InteractionOperator
    :param simulator: simulation object
    :param n_electrons: number of electrons
    :param multiplicity: multiplicity
    :param generate_random: generate random configuration distribution instead of simulation
    :param backend: backend to use for selected-CI  (dice, qiskit)
    :param adapt: True if adaptVQE False if VQE
    :return: the expectation value of the Hamiltonian in the current state (HF ref + ansatz)
    """

    from vqemulti.preferences import Configuration
    from vqemulti.utils import get_fock_space_vector, get_selected_ci_energy_dice, get_selected_ci_energy_qiskit
    from vqemulti.utils import get_dmrg_energy

    alpha_electrons = (multiplicity + n_electrons)//2
    beta_electrons = (n_electrons - multiplicity)//2

    # print('electrons: ', alpha_electrons, beta_electrons)

    from qiskit_addon_sqd.counts import generate_counts_uniform
    import numpy as np

    from openfermion.ops.representations import InteractionOperator
    if not isinstance(hamiltonian, InteractionOperator):
        # transform operator
        # from openfermion.transforms import get_interaction_operator
        # hamiltonian = get_interaction_operator(hamiltonian)
        raise Exception('Hamiltonian must be a InteractionOperator')

    # transform ansatz to qubit for VQE/adaptVQE
    ansatz_qubit = ansatz.transform_to_scaled_qubit(coefficients, join=not adapt)

    if generate_random:
        samples = generate_counts_uniform(simulator._shots, len(hf_reference_fock))
    else:
        samples = simulator.get_sampling(ansatz_qubit, hf_reference_fock)

    print(samples)

    configurations = []
    error_conf = 0
    error_detected = 0

    for bitstring in samples.keys():
        fock_vector = get_fock_space_vector([1 if b == '1' else 0 for b in bitstring[::-1]])

        if np.sum(fock_vector[-2:]) > 0:
            error_detected += 1
            fock_vector[-2] = [0, 0]
            # print('detected!')
            continue

        if np.sum(fock_vector[::2]) == alpha_electrons and np.sum(fock_vector[1::2]) == beta_electrons:
            # print(fock_vector)
            configurations.append(fock_vector)
        else:
            # print('error!')
            error_conf += 1

    return configurations, error_conf, error_detected

warnings.filterwarnings("ignore")

config = Configuration()
config.verbose = 2
config.mapping = 'pc'


from vqemulti.utils import load_wave_function, load_hamiltonian
coeff, ansatz = load_wave_function(filename='wf_qubit_trotter_4.yml', qubit_op=True)
# coeff, ansatz = load_wave_function(filename='wf_fermion.yml', qubit_op=False)


print(coeff)
print(ansatz)


# set HF
#coeff = coeff[:0]
#ansatz = ansatz[:0]

hamiltonian = load_hamiltonian(file='hamiltonian.npz')
#hamiltonian = get_fermion_operator(hamiltonian)
#hamiltonian.compress(1e-2)

matrix = get_sparse_operator(hamiltonian).toarray()
print('lowest eigenvalues:', np.real(np.sort(np.linalg.eigvals(matrix))[:4]))


n_electrons = 4
n_orbitals = 5


# hamiltonian = get_n_particles_operator(n_orbitals)

hamiltonian = expand_interaction_operator(hamiltonian, n_orbitals*2)
print(hamiltonian.one_body_tensor.shape)

from vqemulti.utils import get_sparse_ket_from_fock, get_sparse_operator
from openfermion import count_qubits

print('count: ', count_qubits(hamiltonian))

print('N electrons', n_electrons)
print('N Orbitals', n_orbitals)
# Choose specific pool of operators for adapt-VQE

# Get Hartree Fock reference in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, n_orbitals*2)


from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
# from vqemulti.simulators.penny_simulator import PennylaneSimulator as Simulator

from qiskit_ibm_runtime.fake_provider import FakeTorino, FakeAlmadenV2, FakeKyiv, FakeProviderForBackendV2
from qiskit_aer import AerSimulator

# backend = FakeTorino()
#backend = FakeAlmadenV2()
#backend = FakeVigoV2()
backend = AerSimulator()

#print(backend.coupling_map)

#service = QiskitRuntimeService()
#backend = service.least_busy(simulator=False, operational=True)
#backend = service.backend('ibm_torino')

print('backend: ', backend.name)

# Start session
print('Initialize Single point')

x = []
y = []
for n_shots in np.linspace(100, 500, 10, endpoint=True):

    energy_list = []
    n_conf_list = []

    for _ in range(20):

        # coeff_ = coeff[:i]
        # ansatz_ = ansatz[:i]

        simulator = Simulator(trotter=True,
                              trotter_steps=1,
                              test_only=False,
                              hamiltonian_grouping=True,
                              use_estimator=True,
                              #noise_model=get_noise_model_thermal(n_qubits=n_orbitals * 2, multiple=1.0),
                              noise_model=get_noise_model_bitflip(),
                              shots=n_shots)

        # run SP energy
        # simulator = None
        hf_energy = get_adapt_vqe_energy([], [], hf_reference_fock, hamiltonian, None)
        print('energy HF: ', hf_energy)

        energy = get_adapt_vqe_energy(coeff,
                                      ansatz,
                                      hf_reference_fock,
                                      hamiltonian,
                                      simulator
                                      )

        # print(simulator.get_circuits()[-1])
        # simulator.print_statistics()
        print('energy circuit: ', energy)

        configurations, n_error, n_detect = simulate_configurations_sqd(coeff,
                                                                        ansatz,
                                                                        hf_reference_fock,
                                                                        hamiltonian,
                                                                        simulator,
                                                                        n_electrons,
                                                                        generate_random=False,
                                                                        adapt=True)

        print('N error conf: ', n_error)
        print('N error dete: ', n_detect)

        print(configurations)

        from vqemulti.utils import get_selected_ci_energy_dice
        energy = get_selected_ci_energy_dice(configurations, hamiltonian)
        print('energy SQD: ', energy)

        energy_list.append(energy)
        n_conf_list.append(len(configurations))

    print(energy_list)
    print(n_conf_list)

    #plt.plot(n_conf_list, energy_list, 'o')
    #plt.show()

    x.append(np.average(energy_list))
    y.append(np.average(n_conf_list))

for xi, yi in zip(x, y):
    print('{:8} {:8}'.format(xi, yi))

plt.plot(x, y)
plt.show()