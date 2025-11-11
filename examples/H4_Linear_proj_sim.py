from vqemulti.utils import get_hf_reference_in_fock_space, generate_reduced_hamiltonian
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.adapt_vqe import adaptVQE
from vqemulti.preferences import Configuration
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
import matplotlib.pyplot as plt
import numpy as np
from vqemulti.errors import NotConvergedError
from vqemulti.basis_projection import get_basis_overlap_matrix, project_basis, prepare_ansatz_for_restart


vqe_energies = []
energies_fullci = []
energies_hf = []

Configuration().verbose = True
#Configuration().mapping = 'bk'

# molecule definition
from generate_mol import tetra_h4_mol, linear_h4_mol, square_h4_mol
#h4_molecule = tetra_h4_mol(distance=5.0, basis='sto-3g')
h4_molecule = linear_h4_mol(distance=3.0, basis='sto-3g')
#h4_molecule = square_h4_mol(distance=3.0, basis='sto-3g')

# run classical calculation
molecule = run_pyscf(h4_molecule, run_fci=True, nat_orb=False, guess_mix=False)

print('FullCI energy result:', molecule.fci_energy)

# get properties from classical SCF calculation
n_electrons = molecule.n_electrons
n_orbitals = 4  # molecule.n_orbitals
hamiltonian = molecule.get_molecular_hamiltonian()
hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals)
print('N electrons', n_electrons)
print('N Orbitals', n_orbitals)

# Choose specific pool of operators for adapt-VQE
pool = get_pool_singlet_sd(n_electrons=n_electrons, n_orbitals=n_orbitals)
pool.print_compact_representation()
print(pool)
exit()

# Get Hartree Fock reference in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)
from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
#from vqemulti.simulators.penny_simulator import PennylaneSimulator as Simulator
#from vqemulti.simulators.cirq_simulator import CirqSimulator as Simulator

simulator = Simulator(trotter=True,
                      trotter_steps=1,
                      test_only=True,
                      shots=1000)

simulator_2 = Simulator(trotter=True,
                        trotter_steps=1,
                        test_only=True,
                        shots=1000)

# run adaptVQE
try:
    result = adaptVQE(hamiltonian,
                      pool,
                      hf_reference_fock,
                      #opt_qubits=True,
                      # threshold=1e-4,
                      max_iterations=6,
                      operator_update_number=4,
                      operator_update_max_grad=1e-1,
                      energy_simulator=simulator,
                      gradient_simulator=simulator,
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
print("Ansatz:", result["ansatz"])
print("Indices:", result["indices"])
print("Coefficients:", result["coefficients"])
print("Num operators: {}".format(len(result["ansatz"])))

energies_list = result['iterations']['energies']
error_list = np.array(energies_list) - molecule.fci_energy
norms_list = result['iterations']['norms']

print(error_list)

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

# define max ratio
max_op_ratio = 1.4
max_op = int(max_op_ratio * len(result['coefficients']))

ansatz = sum(op * coeff for op, coeff in zip(result['ansatz'], result['coefficients']))
basis_overlap_matrix = get_basis_overlap_matrix(molecule, molecule_2)
projected_ansatz = project_basis(ansatz, basis_overlap_matrix, n_orb_2=n_orbitals)
restart_coefficients, restart_ansatz = prepare_ansatz_for_restart(projected_ansatz, max_val=1e-1)

print('projected ansatz')
print('coef: ', restart_coefficients)
print('ansatz: ', restart_ansatz)
print('Projection ratio: ', len(restart_coefficients)/len(result['coefficients']))


try:
    result = adaptVQE(hamiltonian,
                      pool,
                      hf_reference_fock,
                      #opt_qubits=True,
                      # threshold=1e-7,
                      max_iterations=14,
                      coefficients=restart_coefficients,
                      ansatz=restart_ansatz,
                      energy_simulator=simulator_2,
                      gradient_simulator=simulator_2,
                      operator_update_number=4,
                      operator_update_max_grad=1e-1,
                      )

except NotConvergedError as e:
    result = e.results


# Compare error vs FullCI calculation
error = result["energy"] - molecule_2.fci_energy
print("Error:", error)

# run results
print("Ansatz:", result["ansatz"])
print("Indices:", result["indices"])
print("Coefficients:", result["coefficients"])
print("Num operators: {}".format(len(result["ansatz"])))

error_list = np.array(energies_list) - molecule_2.fci_energy
energies_list_2 = result['iterations']['energies']
error_list_2 = np.array(energies_list_2) - molecule_2.fci_energy
norms_list_2 = result['iterations']['norms']


for i, en in enumerate(energies_list):
    print('{:16.12f} {:16.12f} {:16.12f}'.format(en, error_list[i], norms_list[i]))


for i, en in enumerate(energies_list_2):
    print('{:16.12f} {:16.12f} {:16.12f}'.format(en, error_list_2[i], norms_list_2[i]))


simulator.print_statistics()
simulator_2.print_statistics()


plt.figure(0)
plt.plot(energies_list_2, label='AdaptVQE proj', color='blue')
plt.legend()

plt.figure(1)
plt.plot(error_list_2, 'o-', label='AdaptVQE proj', color='blue')
plt.legend()

plt.figure(2)
plt.plot(norms_list_2, 'o-', label='AdaptVQE proj', color='blue')
plt.legend()

plt.show()