from vqemulti.utils import get_hf_reference_in_fock_space, generate_reduced_hamiltonian
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.adapt_vqe import adaptVQE
from vqemulti.energy import get_adapt_vqe_energy
from vqemulti.energy.simulation import simulate_adapt_vqe_variance
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
import numpy as np

import warnings
warnings.filterwarnings("ignore")

config = Configuration()
config.verbose = 2
config.mapping = 'jw'


from vqemulti.utils import load_wave_function, load_hamiltonian
coeff, ansatz = load_wave_function(filename='wf_qubit.yml', qubit_op=True)
# coeff, ansatz = load_wave_function(filename='wf_fermion.yml', qubit_op=False)

# set HF
coeff = coeff[:4]
ansatz = ansatz[:4]


hamiltonian = load_hamiltonian(file='hamiltonian.npz')
hamiltonian = get_fermion_operator(hamiltonian)
hamiltonian.compress(1e-2)

n_electrons = 4
n_orbitals = 4

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

service = QiskitRuntimeService()
#backend = service.least_busy(simulator=False, operational=True)
# backend = service.backend('ibm_torino')

print('backend: ', backend.name)

# Start session
print('Initialize Single point')

with Session(backend=backend) as session:

    energies = []
    variances = []

    for _ in range(20):

        simulator = Simulator(trotter=False,
                              trotter_steps=1,
                              test_only=True,
                              hamiltonian_grouping=True,
                              session=session,
                              use_estimator=True,
                              shots=9000)

        # run SP energy
        # simulator = None
        energy = get_adapt_vqe_energy(coeff,
                                      ansatz,
                                      hf_reference_fock,
                                      hamiltonian,
                                      simulator)

        #for circuit in simulator.get_circuits():
        #    print(circuit)


        # print("SP energy:", energy)
        simulator_var = Simulator(trotter=True,
                                  trotter_steps=1,
                                  test_only=True,
                                  hamiltonian_grouping=True,
                                  session=session,
                                  use_estimator=True,
                                  shots=9000)

        #print('=====')
        variance = simulate_adapt_vqe_variance(coeff,
                                               ansatz,
                                               hf_reference_fock,
                                               hamiltonian,
                                               simulator_var)

        #print('------')
        # print('variance:', variance)
        print(energy, variance)
        energies.append(energy)
        variances.append(variance)


print('ave: ', np.average(energies))
print('std: ', np.std(energies))
print('var: ', np.var(energies))


