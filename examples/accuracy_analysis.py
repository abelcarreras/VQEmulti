from vqemulti.utils import get_hf_reference_in_fock_space, generate_reduced_hamiltonian
from vqemulti.pool import get_pool_singlet_sd
from openfermionpyscf import run_pyscf
import matplotlib.pyplot as plt
from vqemulti.vqe import vqe
from vqemulti.ansatz.exp_product import ProductExponentialAnsatz
import numpy as np


vqe_energies = []
energies_fullci = []
energies_hf = []


# molecule definition
from generate_mol import tetra_h4_mol, linear_h4_mol, square_h4_mol
# h4_molecule = tetra_h4_mol(distance=5.0, basis='sto-3g')
# h4_molecule = linear_h4_mol(distance=3.0, basis='3-21g')
h4_molecule = square_h4_mol(distance=3.0, basis='sto-3g')

# run classical calculation
molecule = run_pyscf(h4_molecule, run_fci=True, nat_orb=False, guess_mix=False)

print('FullCI energy result:', molecule.fci_energy)

# get properties from classical SCF calculation
n_electrons = molecule.n_electrons
n_orbitals = 4  # molecule.n_orbitals
hamiltonian = molecule.get_molecular_hamiltonian()
hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals)

# hamiltonian analysis
max_2body = np.max(np.abs(hamiltonian.two_body_tensor))
max_1body = np.max(np.abs(hamiltonian.one_body_tensor))
print('Max valued hamiltonian terms: ', max_1body, max_2body)

ave_2body = np.average(np.abs(hamiltonian.two_body_tensor))
ave_1body = np.average(np.abs(hamiltonian.one_body_tensor))

print('Average valued hamiltonian terms: ', ave_1body, ave_2body)

print('N electrons', n_electrons)
print('N Orbitals', n_orbitals)
# Choose specific pool of operators for adapt-VQE
pool = get_pool_singlet_sd(n_electrons=n_electrons, n_orbitals=n_orbitals)

#print('\nPOOL\n---------------------------')
#pool.print_compact_representation()

# Get Hartree Fock reference in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)
from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
#from vqemulti.simulators.penny_simulator import PennylaneSimulator as Simulator


# simulator = None
precision = 1e-1
print('precision: ', precision)


#coefficients = np.array([])
#ansatz = OperatorList([])

# coefficients = np.array([2.48875])
# ansatz = pool[9: 10]
ansatz = ProductExponentialAnsatz([], [], hf_reference_fock)

std_list = []
et_list = [0.5, 0.4, 0.3, 0.2, 0.1]

for energy_threshold in et_list:

    # energy_threshold = 1e-2

    simulator = Simulator(trotter=True,
                          trotter_steps=1,
                          test_only=False,
                          # use_estimator=True,
                          shots=100)

    energy_list = []
    for i in range(50):

        results = vqe(hamiltonian, ansatz, simulator, energy_threshold=energy_threshold)
        energy_list.append(results['energy'])

    print('energy threshold:', energy_threshold)
    print('VQE energy:', energy_list[-1])
    print('HF energy:', molecule.hf_energy)

    print('Average: ', np.average(energy_list))
    print('Variance: ', np.var(energy_list))
    print('STD: ', np.std(energy_list))

    std_list.append(np.std(energy_list))

    if False:
        plt.hist(energy_list)
        plt.axvline(x=molecule.hf_energy, color='red', label='HF', linewidth=3)
        plt.axvline(x=molecule.hf_energy+np.std(energy_list), color='green', linewidth=1)
        plt.axvline(x=molecule.hf_energy-np.std(energy_list), color='green', linewidth=1)

        plt.show()

plt.plot(et_list, std_list)
plt.show()