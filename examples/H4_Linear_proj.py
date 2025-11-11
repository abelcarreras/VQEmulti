from vqemulti.utils import get_hf_reference_in_fock_space, generate_reduced_hamiltonian
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.adapt_vqe import adaptVQE
from vqemulti.preferences import Configuration
from openfermionpyscf import run_pyscf
from vqemulti.errors import NotConvergedError
from vqemulti.basis_projection import get_basis_overlap_matrix, project_basis, prepare_ansatz_for_restart
from vqemulti.density import get_density_matrix, density_fidelity
import matplotlib.pyplot as plt
import numpy as np
import time


vqe_energies = []
energies_fullci = []
energies_hf = []

Configuration().verbose = True

# molecule definition
from generate_mol import tetra_h4_mol, linear_h4_mol, square_h4_mol
# h4_molecule = tetra_h4_mol(distance=5.0, basis='sto-3g')
#h4_molecule = linear_h4_mol(distance=3.0, basis='3-21g')
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


print('N electrons', n_electrons)
print('N Orbitals', n_orbitals)
# Choose specific pool of operators for adapt-VQE
pool = get_pool_singlet_sd(n_electrons=n_electrons, n_orbitals=n_orbitals)

print('\nPOOL\n---------------------------')
pool.print_compact_representation()

# Get Hartree Fock reference in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)
from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
# from vqemulti.simulators.penny_simulator import PennylaneSimulator as Simulator


simulator = Simulator(trotter=True,
                      trotter_steps=1,
                      test_only=False)

simulator_grad = Simulator(trotter=True,
                           trotter_steps=1,
                           test_only=True)

# simulator_grad = simulator

# simulator = None
precision = 1e-2
print('precision', precision)
st = time.time()

# run adaptVQE
try:
    result = adaptVQE(hamiltonian,
                      pool,
                      hf_reference_fock,
                      # opt_qubits=True,
                      energy_threshold=precision,
                      # coeff_tolerance=precision,
                      # threshold=0.1,
                      max_iterations=5,
                      operator_update_number=1,
                      operator_update_max_grad=precision,
                      energy_simulator=simulator,
                      gradient_simulator=simulator_grad,
                      )

except NotConvergedError as e:
    result = e.results

print("HF energy:", molecule.hf_energy)
print("Final adaptVQE energy:", result["energy"])
print("FullCI energy:", molecule.fci_energy)

# Compare error vs FullCI calculation
error = result["energy"] - molecule.fci_energy
print("Error:", error)

# run results
#print("Ansatz:", result["ansatz"])
print("Indices:", result["indices"])
print("Coefficients:", result["coefficients"])
print("Num operators: {}".format(len(result["ansatz"])))

indices_1 = result["indices"]

energies_list = result['iterations']['energies']
error_list = np.array(energies_list) - molecule.fci_energy
norms_list = result['iterations']['norms']
n_evaluations = np.sum(np.multiply(result['iterations']['f_evaluations'],
                                   result['iterations']['ansatz_size']))

print('n_evaluations: ', n_evaluations)

# fidelity
density_matrix = get_density_matrix(result['coefficients'],
                                    result['ansatz'],
                                    hf_reference_fock,
                                    n_orbitals)

print('\nFidelity measure: {:5.2f}'.format(density_fidelity(molecule.fci_one_rdm, density_matrix)))
simulator.print_statistics()


if True:
    plt.figure(0)
    plt.plot(energies_list, label='AdaptVQE', color='red')
    plt.title('Energies')
    plt.ylabel('Energy [H]')

    plt.figure(1)
    plt.title('Error')
    plt.ylabel('Energy [H]')
    plt.yscale('log', base=10)
    plt.plot(error_list, 'o-', label='AdaptVQE', color='red')

    plt.figure(2)
    plt.title('Norm')
    plt.plot(norms_list, 'o-', label='AdaptVQE', color='red')


print('partial data')
for i, en in enumerate(energies_list):
    print('{:16.12f} {:16.12f} {:16.12f}'.format(en, error_list[i], norms_list[i]))


h4_molecule.basis = '3-21g'
molecule_2 = run_pyscf(h4_molecule, run_fci=True, run_ccsd=True, nat_orb=False, guess_mix=False)
# molecule_2 = molecule

# get properties from classical SCF calculation
n_orbitals = 4  # molecule_2.n_orbitals

hamiltonian = molecule_2.get_molecular_hamiltonian()
hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals)

# print data
print('n_electrons: ', n_electrons)
print('n_orbitals: ', n_orbitals)
print('n_qubits:', hamiltonian.n_qubits)

# Get a pool of operators for adapt-VQE
pool = get_pool_singlet_sd(n_electrons=n_electrons, n_orbitals=n_orbitals)

# Get Hartree Fock reference in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

if False:
    ansatz = sum(op * coeff for op, coeff in zip(result['ansatz'], result['coefficients']))
    basis_overlap_matrix = get_basis_overlap_matrix(molecule, molecule_2)
    projected_ansatz = project_basis(ansatz, basis_overlap_matrix, n_orb_2=n_orbitals)
    restart_coefficients, restart_ansatz = prepare_ansatz_for_restart(projected_ansatz, max_val=1e-2, ref_pool=pool)
    #restart_coefficients, restart_ansatz =  result['coefficients'], result['ansatz']

    #restart_coefficients, restart_ansatz = ansatz_projection_into_pool(pool, projected_ansatz, max_val=1e-2)

    print('projected ansatz')
    print('coef: ', restart_coefficients)
    #print('ansatz: ', restart_ansatz)
    restart_ansatz.print_compact_representation()

    print('Projection ratio: ', len(restart_coefficients)/len(result['coefficients']))

else:
    # simple projection
    restart_ansatz, restart_coefficients = result['ansatz'], result['coefficients']


try:
    result = adaptVQE(hamiltonian,
                      pool,
                      hf_reference_fock,
                      # opt_qubits=True,
                      energy_threshold=precision,
                      # coeff_tolerance=precision,
                      # threshold=precision,
                      max_iterations=20,
                      operator_update_number=1,
                      operator_update_max_grad=precision,
                      coefficients=restart_coefficients,
                      ansatz=restart_ansatz,
                      energy_simulator=simulator,
                      gradient_simulator=simulator_grad,
                      )

except NotConvergedError as e:
    result = e.results

et = time.time()

# Compare error vs FullCI calculation
print('Energy HF: {:.8f}'.format(molecule.hf_energy))
print('Energy HF (2): {:.8f}'.format(molecule_2.hf_energy))
print('Energy adaptVQE: ', result['energy'])
print('Energy FullCI: ', molecule_2.fci_energy)
error = result['energy'] - molecule_2.fci_energy
print('Error:', error)

# run results
print("Ansatz:", result["ansatz"])
print("Indices:", result["indices"])
print("Coefficients:", result["coefficients"])
print("Num operators: {}".format(len(result["ansatz"])))

indices_2 = result["indices"]


# fidelity
density_matrix_2 = get_density_matrix(result['coefficients'],
                                      result['ansatz'],
                                      hf_reference_fock,
                                      n_orbitals)

print('\nFidelity measure (previous): {:5.2f}'.format(density_fidelity(molecule_2.fci_one_rdm, density_matrix)))
print('\nFidelity measure: {:5.2f}'.format(density_fidelity(molecule_2.fci_one_rdm, density_matrix_2)))


n_evaluations += np.sum(np.multiply(result['iterations']['f_evaluations'],
                                    result['iterations']['ansatz_size']))

print("Total num evaluations: {}".format(n_evaluations))
simulator.print_statistics()

error_list = np.array(energies_list) - molecule_2.fci_energy
energies_list_2 = result['iterations']['energies']
error_list_2 = np.array(energies_list_2) - molecule_2.fci_energy
norms_list_2 = result['iterations']['norms']


for i, en in enumerate(energies_list):
    print('{:16.12f} {:16.12f} {:16.12f}'.format(en, error_list[i], norms_list[i]))


for i, en in enumerate(energies_list_2):
    print('{:16.12f} {:16.12f} {:16.12f}'.format(en, error_list_2[i], norms_list_2[i]))


plt.figure(0)
plt.plot(energies_list_2, 'o-', label='AdaptVQE proj', color='blue')
plt.legend()

plt.figure(1)
plt.plot(error_list_2, 'o-', label='AdaptVQE proj', color='blue')
plt.legend()

plt.figure(2)
plt.plot(norms_list_2, 'o-', label='AdaptVQE proj', color='blue')
plt.legend()

plt.show()

from vqemulti.energy import get_adapt_vqe_energy

restart_ansatz, restart_coefficients = result['ansatz'], result['coefficients']
energy = get_adapt_vqe_energy(result['coefficients'], result['ansatz'], hf_reference_fock, hamiltonian, None)
print('energy exact from coeff: ', energy)
print('total running time (s): ', et - st, 's')









exit()

restart_ansatz, restart_coefficients = result['ansatz'], result['coefficients']

#join_index = indices_1 + indices_2


join_index = [9, 13, 4, 8, 11, 9, 11]
restart_coefficients = [-1.1721770404591654, 0.9389031124265976, 0.9381137163650409, -1.3455918146706862, -1.8270170671190429, 0.7252233424070096, 1.9507590692382433]

restart_ansatz = [pool[i] for i in join_index]

from vqemulti.pool.tools import OperatorList

print('In use')
print(restart_coefficients)
print('Indices', join_index)

energy = get_adapt_vqe_energy(restart_coefficients, OperatorList(restart_ansatz), hf_reference_fock, hamiltonian, None)
print('energy: ', energy)
exit()

result = adaptVQE(hamiltonian,
                  pool,
                  hf_reference_fock,
                  # opt_qubits=True,
                  energy_threshold=precision,
                  coeff_tolerance=precision,
                  # threshold=precision,
                  max_iterations=20,
                  operator_update_number=1,
                  operator_update_max_grad=1e-1,
                  coefficients=restart_coefficients,
                  ansatz=restart_ansatz,
                  # energy_simulator=simulator,
                  # gradient_simulator=simulator,
                  )

print('energy: ', result['energy'])