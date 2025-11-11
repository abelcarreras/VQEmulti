# example to compute the variance of a sampled (using simulator) VQE calculation
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from vqemulti.utils import generate_reduced_hamiltonian, get_hf_reference_in_fock_space
from vqemulti.pool.singlet_sd import get_pool_singlet_sd
from vqemulti.vqe import vqe
from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
# from vqemulti.simulators.penny_simulator import PennylaneSimulator as Simulator
from vqemulti.simulators.cirq_simulator import CirqSimulator as Simulator

from vqemulti.energy.simulation import simulate_adapt_vqe_energy_square, simulate_adapt_vqe_energy
from vqemulti.energy import exact_adapt_vqe_energy
import matplotlib.pyplot as plt
import numpy as np
from vqemulti.preferences import Configuration

Configuration().mapping = 'bk'

# define molecule
h2_molecule = MolecularData(geometry=[['H', [0, 0, 0]],
                                      ['H', [0, 0, 1.0]],
                                       ['H', [0, 0, 2.0]],
                                       ['H', [0, 0, 3.0]]
                                      ],
                            basis='3-21g',
                            multiplicity=1,
                            charge=0,
                            description='H2'
                             )

# run classical calculation
molecule = run_pyscf(h2_molecule, run_fci=True, run_ccsd=True, nat_orb=False)

# get properties from classical SCF calculation
n_electrons = molecule.n_electrons
n_orbitals = 4  # molecule.n_orbitals

print('n_electrons: ', n_electrons)
print('n_orbitals: ', n_orbitals)

hamiltonian = molecule.get_molecular_hamiltonian()
hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals, frozen_core=0)

print('n_qubits:', hamiltonian.n_qubits)

# Get UCCSD ansatz
uccsd_ansatz = get_pool_singlet_sd(n_electrons, n_orbitals, frozen_core=0)

# print(n_electrons, hamiltonian.n_qubits)
# Get reference Hartree Fock state
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits, frozen_core=0)
print('hf reference', hf_reference_fock)

# Compute VQE to get a wave function (coefficients/ansatz)  [no simulation]
print('Initialize VQE')
simulator = Simulator(trotter=True, test_only=True, hamiltonian_grouping=True)

# uccsd_ansatz = []
# simulator = None
# print(uccsd_ansatz[:3])
result = vqe(hamiltonian, uccsd_ansatz[:1], hf_reference_fock, energy_simulator=simulator)

print(result['coefficients'])

print('Energy HF: {:.8f}'.format(molecule.hf_energy))
print('Energy VQE: {:.8f}'.format(result['energy']))
print('Energy FullCI: {:.8f}'.format(molecule.fci_energy))
print('------------------------')

from vqemulti.energy import simulate_vqe_energy
from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator

simulator = Simulator(trotter=False, test_only=True, shots=10000, hamiltonian_grouping=False)
exact = simulate_vqe_energy(result['coefficients'], result['ansatz'], hf_reference_fock, hamiltonian, simulator)

simulator.print_circuits()
print('Exact energy: ', exact)

exit()

#simulator.print_circuits()

# exit()
# Configuration().verbose = 2
# perform an energy evaluation for the calculated wave function (coefficients/ansatz)
exact = exact_adapt_vqe_energy(result['coefficients'], result['ansatz'], hf_reference_fock, hamiltonian)
print('Exact energy: ', exact)

# compute the variance as Var = E[X^2] - E[X]^2 with simulator
n_measures = 1000
simulator = Simulator(trotter=False, test_only=False, shots=10, hamiltonian_grouping=False)

print('computing e_list')
#e_list = [simulate_adapt_vqe_energy(result['coefficients'], result['ansatz'], hf_reference_fock, hamiltonian, simulator)
#          for _ in range(n_measures)]

#e2_list = [simulate_adapt_vqe_energy_square(result['coefficients'], result['ansatz'], hf_reference_fock, hamiltonian, simulator)
#           for _ in range(n_measures)]

print('computing e2_list')

energy_list = []
e_list = []
e2_list = []

simulator = Simulator(trotter=False, test_only=True, shots=1000, hamiltonian_grouping=False)
e = simulate_adapt_vqe_energy(result['coefficients'], result['ansatz'], hf_reference_fock, hamiltonian, simulator)
e2 = simulate_adapt_vqe_energy_square(result['coefficients'], result['ansatz'], hf_reference_fock, hamiltonian, simulator)
var_exact = e2 - e**2
print('Var exact: ', var_exact)

target_std = 2e-2
n_qubits = 2 * n_orbitals

n_shots = int(var_exact/(target_std * 2 / (n_orbitals-1))**2)
n_shots = max(1, n_shots)
print('N_shots: ', n_shots)
# n_shots = 300


for i in range(n_measures):
    print('i:', i)
    simulator = Simulator(trotter=False, test_only=False, shots=n_shots, hamiltonian_grouping=False)
    e = simulate_adapt_vqe_energy(result['coefficients'], result['ansatz'], hf_reference_fock, hamiltonian, simulator)
    e2 = simulate_adapt_vqe_energy_square(result['coefficients'], result['ansatz'], hf_reference_fock, hamiltonian, simulator)
    e_list.append(e**2)
    e2_list.append(e2)
    energy_list.append(e)

plt.title('Energy distribution')
plt.hist(e_list, histtype=u'step', density=True, label='E')
plt.xlabel('E (Hartree)')
plt.hist(e2_list, histtype=u'step', density=True, label='E2')
plt.xlabel('E (Hartree)')
plt.show()


print('E[X^2]: ', np.average(e2_list))
print('E[X]^2: ', np.average(e_list))
#print('E[X]: ', np.average(e_list))

variance = np.average(e2_list) - np.average(e_list)

# variance = e2 - e**2

print('Variance: ', variance)

std = np.sqrt(abs(variance))
print('standard error: ', std)
#print('Variance/STD energy: ', np.var(e_list), np.std(e_list))
print('Variance/STD energy: ', np.var(energy_list), np.std(energy_list))
print('Predicted var/std', var_exact/n_shots, np.sqrt(var_exact/n_shots),
      np.sqrt(var_exact/n_shots)/2)
target_std = np.std(energy_list)

print('Predicted N_shots: ',  var_exact/(target_std * 2)**2)

plt.title('Energy distribution')
plt.hist(e_list, histtype=u'step', density=True, label='E^2')
plt.xlabel('E (Hartree)')
plt.figure()
plt.title('Energy square distribution')
plt.hist(energy_list, histtype=u'step', density=True, label='E^2')
#plt.hist(np.square(e_list), histtype=u'step', density=True, label='E^2 (2)', bins=10)
plt.xlabel('E^2 (Hartree^2)')

plt.show()