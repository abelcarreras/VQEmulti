from vqemulti.utils import get_hf_reference_in_fock_space, generate_reduced_hamiltonian
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.preferences import Configuration
from openfermionpyscf import run_pyscf
import matplotlib.pyplot as plt
from vqemulti.energy.simulation import simulate_adapt_vqe_energy_square, simulate_adapt_vqe_energy
from vqemulti.energy import exact_adapt_vqe_energy
from vqemulti.pool.tools import OperatorList
from scipy.optimize import curve_fit
import scipy
import numpy as np


vqe_energies = []
energies_fullci = []
energies_hf = []

Configuration().verbose = True

# molecule definition
from generate_mol import tetra_h4_mol, linear_h4_mol, square_h4_mol
# h4_molecule = tetra_h4_mol(distance=5.0, basis='sto-3g')
# h4_molecule = linear_h4_mol(distance=3.0, basis='3-21g')
h4_molecule = square_h4_mol(distance=3.0, basis='sto-3g')

# run classical calculation
molecule = run_pyscf(h4_molecule, run_fci=True, nat_orb=False, guess_mix=False)

print('FullCI energy result:', molecule.fci_energy)

# get properties from classical SCF calculation
n_electrons = 0  # molecule.n_electrons
n_orbitals = 3  # molecule.n_orbitals
hamiltonian = molecule.get_molecular_hamiltonian()
hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals)
print(hamiltonian)


# hamiltonian analysis
max_2body = np.max(np.abs(hamiltonian.two_body_tensor))
max_1body = np.max(np.abs(hamiltonian.one_body_tensor))
print('Max valued hamiltonian terms: ', max_1body, max_2body)


print('N electrons', n_electrons)
print('N Orbitals', n_orbitals)
# Choose specific pool of operators for adapt-VQE
pool = get_pool_singlet_sd(n_electrons=n_electrons, n_orbitals=n_orbitals)

#print('\nPOOL\n---------------------------')
#pool.print_compact_representation()

# Get Hartree Fock reference in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)
print(hf_reference_fock)

from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
#from vqemulti.simulators.cirq_simulator import CirqSimulator as Simulator
#from vqemulti.simulators.penny_simulator import PennylaneSimulator as Simulator


#coefficients = np.array([3.6506357220676118, -0.5917326170294643, 1.0159462461638593, 0.934297380915808, -0.12873809970649322])
#index_list = [9, 8, 11, 6, 9]
#ansatz = OperatorList([pool[i-1] for i in index_list])


coefficients = np.array([2.48875])
ansatz = pool[9: 10]


coefficients = np.array([])
ansatz = OperatorList([])

# print('ansatz:', ansatz)

std_list = []
et_list = [0.1]
shot_list = [10, 20, 40, 60, 80, 100]

theoretical = True
for shots in shot_list:
    #continue
    energy_threshold = 1e-2

    simulator = Simulator(trotter=False,
                          trotter_steps=1,
                          test_only=False,
                          hamiltonian_grouping=True,
                          # use_estimator=False,
                          shots=shots
                          )

    energy_list = []
    energy_2_list = []

    for i in range(5000):
        if False:
            results = scipy.optimize.minimize(simulate_vqe_energy,
                                              coefficients,
                                              (ansatz, hf_reference_fock, hamiltonian, simulator),
                                              method='COBYLA',  # SPSA for real hardware
                                              tol=energy_threshold,
                                              options={'maxiter':1, 'disp': False})  # 'rhobeg': 0.01)
            print(results)
            energy_list.append(results.fun)
        else:
            # e = exact_vqe_energy(coefficients, ansatz, hf_reference_fock, hamiltonian)
            e = simulate_adapt_vqe_energy(coefficients, ansatz, hf_reference_fock, hamiltonian, simulator)
            # e2 = simulate_vqe_energy_square(coefficients, ansatz, hf_reference_fock, hamiltonian, simulator)

            energy_list.append(e)
            # energy_2_list.append(e2)


        print('i', i, energy_list[-1], shots)

    #variance = np.average(energy_2_list) - np.average(energy_list) ** 2
    #print('Variance operator: ', variance)

    print('energy threshold:', energy_threshold)
    print('shots:', shots)

    # print('VQE energy:', energy_list[-1])
    print('HF energy:', molecule.hf_energy)

    print('Average: ', np.average(energy_list))
    print('Variance: ', np.var(energy_list))
    print('STD: ', np.std(energy_list))

    #simulator.print_circuits()
    # exit()

    std_list.append(np.std(energy_list))

    if False:
        plt.hist(energy_list)
        plt.axvline(x=molecule.hf_energy, color='red', label='HF', linewidth=3)
        plt.axvline(x=molecule.hf_energy+np.std(energy_list), color='green', linewidth=1)
        plt.axvline(x=molecule.hf_energy-np.std(energy_list), color='green', linewidth=1)

        plt.show()

print('std_list')
print(std_list)

#std_list = [0.074596311, 0.0574256,  0.0374256, 0.03008588, 0.02543836926, 0.02265]
#std_list = [0.0384268001523973, 0.043750467964261, 0.04259113886678, 0.059238783873518, 0.05637465521179, 0.098248652096104][::-1]

plt.plot(shot_list, std_list, 'o')


popt, _ = curve_fit(lambda x, coeff: coeff / np.sqrt(x), shot_list, std_list)
coeff = popt[0]
# coeff = 0.35 # 0.227

print('coeff: ', coeff, coeff**2)

# print theoretical fitted curve
shot_theo = np.linspace(10, 100, 100)
plt.plot(shot_theo, [coeff / np.sqrt(i) for i in shot_theo])

plt.show()
