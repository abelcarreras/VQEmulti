# example of the use of IBM runtime and sessions (requires qiskit-ibm-runtime module)
import matplotlib.pyplot as plt
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from vqemulti.vqe import vqe
from vqemulti.utils import generate_reduced_hamiltonian, get_hf_reference_in_fock_space
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.simulators.qiskit_simulator import QiskitSimulator
#from vqemulti.simulators.penny_simulator import PennylaneSimulator as QiskitSimulator
#from vqemulti.simulators.cirq_simulator import CirqSimulator as QiskitSimulator

from qiskit_ibm_runtime import Session
from vqemulti.energy import get_vqe_energy
from vqemulti.energy.simulation import simulate_vqe_variance
from vqemulti.preferences import Configuration
import numpy as np

config = Configuration()
#config.verbose = 2



he_molecule = MolecularData(geometry=[['He', [0, 0, 0]],
                                      ['He', [0, 0, 1.0]]],
                            basis='3-21g',
                            # basis='sto-3g',
                            multiplicity=1,
                            charge=-2,
                            description='He2')

# run classical calculation
molecule = run_pyscf(he_molecule, run_fci=True, run_ccsd=True)

# Use a restricted MO space (2 frozen, 2 active)
n_electrons = molecule.n_electrons
n_orbitals = 4  # molecule.n_orbitals

print('n_electrons: ', n_electrons)
print('n_orbitals: ', n_orbitals)

hamiltonian = molecule.get_molecular_hamiltonian()
hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals, frozen_core=2)

print('n_qubits:', hamiltonian.n_qubits)

# Get UCCSD ansatz
uccsd_ansatz = get_pool_singlet_sd(n_electrons, n_orbitals, frozen_core=2)

# Get reference Hartree Fock state
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits, frozen_core=2)
print('hf reference', hf_reference_fock)

# fake backend
from qiskit_ibm_runtime.fake_provider import FakeBrisbane, FakeAlmadenV2, FakeTorino, FakeProvider
from vqemulti.simulators.fake_sim import FakeIdeal
from qiskit.providers.fake_provider import Fake20QV1
#backend = FakeBrisbane()
#backend = FakeTorino()
backend = FakeIdeal()
#backend = FakeProvider()
#backend = Fake20QV1()
#backend_new.num_qubits = backend.num_qubits
#backend_new.coupling_map = backend.coupling_map
#backend_new.target = backend.target
#backend_new.layout = [0, 1, 2]

#backend = backend_new
# real hardware backend
#backend = 'ibm_torino'

# least real backend least_busy
from qiskit_ibm_runtime import QiskitRuntimeService
#service = QiskitRuntimeService()
#backend = service.least_busy(simulator=False, operational=True)
print('backend name: ', backend.name)

# check layout
from vqemulti.simulators.backend_opt import get_backend_opt_layout
#print('layout: ', get_backend_opt_layout(backend, hamiltonian.n_qubits, plot_data=True))


# Start session
print('Initialize VQE')
with Session(backend=backend) as session:

    # Qiskit Simulator
    simulator = QiskitSimulator(trotter=True,
                                trotter_steps=1,
                                shots=1024,
                                test_only=False,
                                hamiltonian_grouping=False,
                                # session=session,
                                backend=backend,
                                use_estimator=True,
                                )

    simulator_exact = QiskitSimulator(trotter=True,
                                      trotter_steps=1,
                                      test_only=True,
                                      hamiltonian_grouping=False,
                                      )

    if True:
        variance_exact = simulate_vqe_variance([0.0, 0.8061306828994627], uccsd_ansatz, hf_reference_fock, hamiltonian, simulator_exact)
        energy_exact = get_vqe_energy([0.0, 0.8061306828994627], uccsd_ansatz, hf_reference_fock, hamiltonian, simulator_exact)
        print('Variance_exact: ', variance_exact)
        print('Energy_exact: ', energy_exact)

        energy_list = []
        for i in range(500):
            energy = get_vqe_energy([0.0, 0.8061306828994627], uccsd_ansatz, hf_reference_fock, hamiltonian, simulator)
            print('energy: ', energy, i)
            energy_list.append(energy)
            # variance = simulate_vqe_variance([0.0, 0.0], uccsd_ansatz, hf_reference_fock, hamiltonian, simulator)
            # print('Variance: ', variance)

            simulator.print_circuits()
            simulator.print_statistics()

            exit()

        plt.hist(energy_list)
        plt.show()

        print('FINAL')
        print('Variance: ', np.var(energy_list), variance_exact/simulator._shots)
        print('STD: ', np.std(energy_list), np.sqrt(variance_exact/simulator._shots))
        print('Energy error: ', np.mean(energy_list) - energy_exact)

    if False:
        energy = get_vqe_energy([0.0, 0.8061306828994627], uccsd_ansatz, hf_reference_fock, hamiltonian, simulator)
        print('energy: ', energy)
        print()

    if False:
        result = vqe(hamiltonian,
                     uccsd_ansatz,
                     hf_reference_fock,
                     energy_simulator=simulator)

        print('Energy HF: {:.8f}'.format(molecule.hf_energy))
        print('Energy VQE: {:.8f}'.format(result['energy']))
        print('Energy CCSD: {:.8f}'.format(molecule.ccsd_energy))
        print('Energy FullCI: {:.8f}'.format(molecule.fci_energy))

        print('Num operators: ', len(result['ansatz']))
        print('Ansatz:\n', result['ansatz'])
        print('Coefficients:\n', result['coefficients'])
