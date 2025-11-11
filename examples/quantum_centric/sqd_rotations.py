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
from vqemulti.utils import store_hamiltonian, store_wave_function, load_wave_function, load_hamiltonian
from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
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

warnings.filterwarnings("ignore")

config = Configuration()
config.verbose = True
config.mapping = 'jw'

warnings.filterwarnings("ignore")


simulator = Simulator(trotter=True,
                      trotter_steps=1,
                      test_only=True,
                      hamiltonian_grouping=True,
                      use_estimator=True,
                      shots=2000)


hamiltonian = load_hamiltonian('hamiltonian.npz')
coefficients, ansatz = load_wave_function(filename='wf_fermion_1.yml', qubit_op=False)
# hf_reference_fock = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

from vqemulti.pool.tools import OperatorList, hermitian_conjugated
from openfermion import FermionOperator

op1 = FermionOperator('2^ 4')
op2 = FermionOperator('3^ 5')
coefficients = [np.pi/2,
                np.pi/2]
ansatz = OperatorList([op1 - hermitian_conjugated(op1),
                       op2 - hermitian_conjugated(op2)])

n_electrons = 4
n_orbitals = 5
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, n_orbitals*2)
print(hf_reference_fock)

#print(hf_reference_fock)
#hf_reference_fock = [1, 0, 0, 1, 1, 0, 0, 1, 0, 0]
hf_energy = get_adapt_vqe_energy([], [], hf_reference_fock, hamiltonian, None)
print('energy HF: ', hf_energy)
# exit()

#coefficients = coefficients[:0]
#ansatz = ansatz[:0]

energy = get_adapt_vqe_energy(coefficients,
                              ansatz,
                              hf_reference_fock,
                              hamiltonian,
                              simulator
                              )

print('energy VQE: ', energy)

energy = simulate_energy_sqd(coefficients,
                             ansatz,
                             hf_reference_fock,
                             hamiltonian,
                             simulator,
                             n_electrons,
                             generate_random=False,
                             adapt=True)

print('final SDQ energy: ', energy)
