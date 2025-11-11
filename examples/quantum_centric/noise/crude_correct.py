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
from vqemulti.utils import get_selected_ci_energy_dice
import numpy as np
import warnings


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


def simulate_noise_configurations(hf_reference_fock, n_electrons,
                                  multiplicity=0,
                                  prob_error=0.1,
                                  n_samples=100,
                                  n_check_qubits=2):

    from vqemulti.utils import get_fock_space_vector
    from copy import deepcopy

    alpha_electrons = (multiplicity + n_electrons)//2
    beta_electrons = (n_electrons - multiplicity)//2

    configurations = []
    error_introduced = []


    for k in range(n_samples):
        error_mark = False
        conf = deepcopy(hf_reference_fock)
        for i, r in enumerate(np.random.rand(len(hf_reference_fock))):
            if r < prob_error:
                conf[i] = 1 if conf[i] == 0 else 1
                error_mark = True

        configurations.append(conf)
        error_introduced.append(error_mark)

    error_detected = []
    configurations_corrected = [get_fock_space_vector(hf_reference_fock)]
    error_conf = 0
    for conf in configurations:
        fock_vector = get_fock_space_vector(conf)

        if np.sum(fock_vector[-n_check_qubits:]) > 0 and n_check_qubits > 0:
            # error_detected += 1
            fock_vector[-n_check_qubits:] = [0] * n_check_qubits
            # print('detected! and corrected')
            error_detected.append(True)

        else:
            error_detected.append(False)

        # print(fock_vector)
        if np.sum(fock_vector[::2]) == alpha_electrons and np.sum(fock_vector[1::2]) == beta_electrons:
            # print(fock_vector)
            if fock_vector not in configurations_corrected:
                configurations_corrected.append(fock_vector)

            error_conf += 1

    return error_introduced, error_detected, configurations_corrected


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

n_check_orbitals = 1

n_electrons = 4
n_orbitals = 4 + n_check_orbitals


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

x = []
y = []
for prob_error in np.linspace(0, 0.5, 10, endpoint=True):

    print('prob_error: ', prob_error)
    energy_list = []
    n_conf_list = []

    n_shots = 100000
    for _ in range(1):

        error_introduced, error_detected, configurations_corrected = simulate_noise_configurations(hf_reference_fock,
                                                                                                   n_electrons,
                                                                                                   prob_error=prob_error,
                                                                                                   n_samples=int(n_shots),
                                                                                                   n_check_qubits=n_check_orbitals*2)
        total_samples = len(error_introduced)
        total_errors = np.sum([int(b) for b in error_introduced])
        total_detected = np.sum([int(b) for b in error_detected])
        print('N samples', total_samples)
        print('N error introduced: ', total_errors)
        print('N error detected: ', total_detected)

        correct_evaluated = 0
        false_positive = 0
        false_negative = 0
        for i, d in zip(error_introduced, error_detected):
            if i == d:
                correct_evaluated += 1
            else:
                if not d:
                    false_positive += 1
                if d:
                    false_negative += 1

        ratio_e = np.sum([int(b) for b in error_detected])/total_errors
        print('ratio errors found respect to total errors: {:4.2f} %'.format(ratio_e*100))

        ratio_s = correct_evaluated/total_samples
        print('ratio successfully evaluated : {:4.2f} %'.format(ratio_s*100))

        ratio_fp = false_positive/total_errors
        ratio_fn = false_negative/total_errors

        print('False positive: {:4.2f} %'.format(ratio_fp*100))
        print('False negative: {:4.2f} %'.format(ratio_fn*100))

        energy = get_selected_ci_energy_dice(configurations_corrected, hamiltonian)
        print('energy SQD: ', energy)

        print(ratio_e*100, '\t', ratio_s*100, '\t',ratio_fp*100, '\t', energy)

        energy_list.append(energy)
        n_conf_list.append(len(configurations_corrected))

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