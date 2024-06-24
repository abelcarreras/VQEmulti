# example of the use of IBM runtime and sessions (requires qiskit-ibm-runtime module)
import numpy as np
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from vqemulti.vqe import vqe
from vqemulti.utils import generate_reduced_hamiltonian, get_hf_reference_in_fock_space
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.simulators.qiskit_simulator import QiskitSimulator
from qiskit_ibm_runtime import Session
from vqemulti.energy import get_vqe_energy
from vqemulti.energy.simulation import simulate_vqe_variance
from vqemulti.preferences import Configuration
from qiskit_ibm_runtime import QiskitRuntimeService

config = Configuration()
config.verbose = 1


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
from qiskit_ibm_runtime.fake_provider import FakeBrisbane, FakeAlmadenV2
backend = FakeBrisbane()
# backend = FakeAlmadenV2()

# real hardware backend
# backend = 'ibm_cusco'

service = QiskitRuntimeService()
# backend = service.least_busy(simulator=False, operational=True)
# print('backend: ', backend.name)

# Start session
print('Initialize VQE')

single_point_test = True

with Session(backend=backend) as session:

    # Qiskit Simulator
    simulator = QiskitSimulator(trotter=True,
                                trotter_steps=1,
                                shots=1024,
                                test_only=False,
                                hamiltonian_grouping=False,
                                session=session,
                                )

    if single_point_test:
        # single point energy
        variance = simulate_vqe_variance([0.0, 0.8061306828994627], uccsd_ansatz, hf_reference_fock, hamiltonian, simulator)
        print('Variance: ', variance)

        energy_list = []
        for i in range(20):
            energy = get_vqe_energy([0.0, 0.8061306828994627], uccsd_ansatz, hf_reference_fock, hamiltonian, simulator)
            print('energy: ', energy)
            energy_list.append(energy)
            # variance = simulate_vqe_variance([0.0, 0.0], uccsd_ansatz, hf_reference_fock, hamiltonian, simulator)
            # print('Variance: ', variance)

        print('Variance: ', np.var(energy_list), variance/simulator._shots)

    else:
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
