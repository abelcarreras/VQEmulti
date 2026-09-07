from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
from openfermionpyscf import run_pyscf
from openfermion import MolecularData
from vqemulti.utils import get_hf_reference_in_fock_space
from qiskit_ibm_runtime.fake_provider import FakeTorino
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator
from vqemulti.ansatz.exponential import ExponentialAnsatz
from vqemulti.preferences import Configuration
from collections import Counter
from vqemulti.utils import get_fock_space_vector, get_selected_ci_energy_dice
from vqemulti.utils import get_dmrg_energy, fermion_to_qubit
from vqemulti.sqd import get_subspace_configurations, configuration_recovery
from vqemulti.hamiltonian.tools import get_qdrift_hamiltonian, get_compressed_qubit_hamiltonian
import numpy as np

# config = Configuration()
# config.verbose = 2

#backend = FakeTorino()
# service = QiskitRuntimeService()
# backend = service.backend('ibm_basquecountry')
backend = AerSimulator()

simulator = Simulator(trotter=True,
                      trotter_steps=1,
                      test_only=True,
                      hamiltonian_grouping=True,
                      use_estimator=True,
                      shots=1000,
                      backend=backend,
                      use_ibm_runtime=True
                      )

d = 2.0
hydrogen = MolecularData(geometry=[('H', [0.0, 0.0, 0.0]),
                                   ('H', [1*d, 0.0, 0.0]),
                                   ('H', [2*d, 0.0, 0.0]),
                                   ('H', [3*d, 0.0, 0.0])],
                         basis='sto-3g',
                         multiplicity=1,
                         charge=0,
                         description='molecule')

# run classical CASCI calculation
from pyscf.fci import cistring
n_frozen_orb = 0  # nothing
n_total_orb = 4  # total orbitals
molecule = run_pyscf(hydrogen,
                     run_fci=True,
                     verbose=True,
                     run_casci=True,
                     frozen_core=n_frozen_orb,
                     n_orbitals=n_total_orb,
                     run_ccsd=True)

fci_energy = molecule.fci_energy

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
tol_ampl = 0.01
for i, a in enumerate(alpha_det):
    for j, b in enumerate(beta_det):
        amp = mc.ci[i, j]
        if amp ** 2 > tol_ampl:
            # print(f"α={format(a, f'0{ncas}b')}  β={format(b, f'0{ncas}b')}  coef={amp:+.6f}")
            cfg = interleave_bits(a, b, ncas)
            print(f"{cfg}   {amp:+.6f}  ({amp ** 2:.6f}) ")

hamiltonian = molecule.get_molecular_hamiltonian()
hamiltonian_te = fermion_to_qubit(hamiltonian)
print('H terms:', len(hamiltonian_te.terms))

multiplicity = 1
n_electrons = molecule.n_electrons
n_orbitals = molecule.n_orbitals
n_qubits = molecule.n_qubits
print('n_qubits:', n_qubits)
print('n_electrons:', n_electrons)

# reference
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, n_qubits)

# time step
dt = 2e-2

# get time-evolution approximated Hamiltonian
# hamiltonian_te = get_compressed_qubit_hamiltonian(hamiltonian, 2e-2)
hamiltonian_te = get_qdrift_hamiltonian(hamiltonian, 100)
print('H terms approx:', len(hamiltonian_te.terms))

energy_error_list = []
configuration_number = []
accumulated_samples = Counter({})

for time in np.arange(0.0, dt*30, dt):

    print('time:', time)
    generator = [-1j * hamiltonian_te]
    coefficients = [time]
    ansatz = ExponentialAnsatz(coefficients, generator, hf_reference_fock)

    energy_intial = ansatz.get_energy(ansatz.parameters, hamiltonian, simulator)
    print('energy guess: ', energy_intial)
    # print('energy SIM: ', ansatz.get_energy(ansatz.parameters, hamiltonian, simulator))

    samples = ansatz.get_sampling(simulator)
    accumulated_samples.update(samples)
    print('samples:', accumulated_samples)
    print('n_samples: ', len(accumulated_samples))

    rec_samples = configuration_recovery(accumulated_samples, hamiltonian, n_electrons,
                                         multiplicity=multiplicity,
                                         n_iter=4,
                                         n_max_diff=4,
                                         regularization_factor=0.7,
                                         max_configurations=1000)

    sampled_configurations = get_subspace_configurations(rec_samples,
                                                         max_configurations=1000,
                                                         add_hf_configuration=True)

    print('sampled configurations:', len(sampled_configurations))

    sci_energy = get_selected_ci_energy_dice(sampled_configurations, hamiltonian)
    print('SCI energy', sci_energy)
    print('Energy error', sci_energy - fci_energy)

    energy_error_list.append(sci_energy - fci_energy)
    configuration_number.append(len(sampled_configurations))

energy_dmrg = get_dmrg_energy(hamiltonian, n_electrons, max_bond_dimension=50)
print('energy DMRG: ', energy_dmrg)

import matplotlib.pyplot as plt

plt.title('SKQD')
plt.ylabel('Energy error [Ha]')
plt.plot(energy_error_list, label='SKQD')
plt.yscale('log')
plt.axhline(energy_dmrg - fci_energy, color='red', label='DMRG')
plt.legend()

plt.figure()
plt.title('Subspace size')
plt.ylabel('# of configurations')
plt.plot(configuration_number)

plt.show()

