# example to compute the variance of a sampled (using simulator) VQE calculation
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from vqemulti.utils import generate_reduced_hamiltonian, get_hf_reference_in_fock_space
from vqemulti.pool.singlet_sd import get_pool_singlet_sd
from vqemulti.vqe import vqe
from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
# from vqemulti.simulators.penny_simulator import PennylaneSimulator as Simulator
from vqemulti.simulators.cirq_simulator import CirqSimulator as Simulator

from vqemulti.energy.simulation import simulate_adapt_vqe_energy_square, simulate_adapt_vqe_energy, simulate_vqe_variance, simulate_adapt_vqe_variance
from vqemulti.energy import exact_adapt_vqe_energy
import matplotlib.pyplot as plt
import numpy as np
from vqemulti.preferences import Configuration


Configuration().mapping = 'jw'


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
n_orbitals = 3  # molecule.n_orbitals

print('n_electrons: ', n_electrons)
print('n_orbitals: ', n_orbitals)

hamiltonian = molecule.get_molecular_hamiltonian()
hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals, frozen_core=1)
print(hamiltonian)

print('n_qubits:', hamiltonian.n_qubits)

# Get UCCSD ansatz
uccsd_ansatz = get_pool_singlet_sd(n_electrons, n_orbitals, frozen_core=1)

# print(n_electrons, hamiltonian.n_qubits)
# Get reference Hartree Fock state
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits, frozen_core=1)
print('hf reference', hf_reference_fock)

# Compute VQE to get a wave function (coefficients/ansatz)  [no simulation]
print('Initialize VQE')
simulator = Simulator(trotter=True, test_only=True, hamiltonian_grouping=True)

# uccsd_ansatz = []
# simulator = None
# print(uccsd_ansatz[:3])
print('full ansatz size: ', len(uccsd_ansatz))
n_coeff = 3

result = vqe(hamiltonian, uccsd_ansatz[:n_coeff], hf_reference_fock, energy_simulator=simulator)

print('Coefficients: ', result['coefficients'])

print('Energy HF: {:.8f}'.format(molecule.hf_energy))
print('Energy VQE: {:.8f}'.format(result['energy']))
print('Energy FullCI: {:.8f}'.format(molecule.fci_energy))
print('------------------------')

from vqemulti.energy import simulate_vqe_energy
from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator

simulator = Simulator(trotter=False, test_only=True, shots=10000, hamiltonian_grouping=True)
exact = simulate_vqe_energy(result['coefficients'], result['ansatz'], hf_reference_fock, hamiltonian, simulator)
# simulator.print_circuits()
print('Exact VQE energy: ', exact)

#exact = exact_adapt_vqe_energy(result['coefficients'], result['ansatz'], hf_reference_fock, hamiltonian)
#print('Exact adaptVQE energy: ', exact)

# compute the variance as Var = E[X^2] - E[X]^2 with simulator
n_measures = 1

n_shots = 5900  # 201  # 370 # 185
n_shots = max(1, n_shots)
print('N_shots: ', n_shots)
# n_shots = 300

print('final coefficients: ', result['coefficients'])
energy_list = []
variance_list = []
use_opt = True

simulator = Simulator(trotter=False, test_only=True, shots=n_shots, hamiltonian_grouping=False)

for i in range(n_measures):
    print('i:', i)

    if not use_opt:
        e = simulate_vqe_energy(result['coefficients'], result['ansatz'], hf_reference_fock, hamiltonian, simulator)
    else:
        results = vqe(hamiltonian, result['ansatz'], hf_reference_fock, result['coefficients'],
                      energy_simulator=simulator, energy_threshold=1e-2)

        e = results['energy']
    energy_list.append(e)

    # print(energy_list)
    # variance = simulate_adapt_vqe_variance(result['coefficients'], result['ansatz'], hf_reference_fock, hamiltonian, simulator)

    variance = simulate_vqe_variance(result['coefficients'], result['ansatz'], hf_reference_fock, hamiltonian, simulator)
    variance_list.append(variance)
    # print(variance)
    # target_dev = 1e-2
    # print('NShots (1e-2): : ', int(variance / target_dev ** 2))


variance = np.average(variance_list)

target_dev = 1e-1
print('NShots (1e-1): ', int(variance/target_dev**2))
target_dev = 1e-2
print('NShots (1e-2): ', int(variance/target_dev**2))
target_dev = 1e-3
print('NShots (1e-3): ', int(variance/target_dev**2))
target_dev = 2e-3
print('NShots (2e-3): ', int(variance/target_dev**2))

# variance = e2 - e**2

print('Average VQE energy: ', np.average(energy_list))
print('Variance/STD energy: ', np.var(energy_list), np.std(energy_list))

print(energy_list)
plt.title('Energy distribution')
plt.hist(energy_list, histtype=u'step', density=True, label='E')
plt.xlabel('E (Hartree)')

simulator.print_statistics()

plt.show()