# example of oxygen molecule using frozen core of 7 orbitals (14 electrons)
# 1 occupied orbital (8) and 1 virtual orbital (9).
# calculation done in 4 qubits (8a, 8b, 9a, 9b).
# https://doi.org/10.48550/arXiv.2009.01872
import numpy as np
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.utils import generate_reduced_hamiltonian, get_hf_reference_in_fock_space
from vqemulti.density import get_density_matrix, density_fidelity
from vqemulti import adaptVQE
import matplotlib.pyplot as plt
from vqemulti import NotConvergedError


molecule_1 = MolecularData(geometry=[['O', [0.0000000000,  0.0000000000, 0.0000000000]],
                                     ['H', [0.0000000000,  0.7906895374, 0.6122172800]],
                                     ['H', [0.0000000000, -0.7906895374, 0.6122172800]]],
                            basis='3-21g',
                            # basis='sto-3g',
                            multiplicity=1,
                            charge=0,
                            description='H2O')

molecule_3 = MolecularData(geometry=[['O', [ 0.0000000000,  0.0000000000, 0.0000000000]],
                                     ['H', [ 0.0000000000,  2.3720687212, 1.8366518401]],
                                     ['H', [ 0.0000000000, -2.3720687212, 1.8366518401]]],
                            basis='3-21g',
                            # basis='sto-3g',
                            multiplicity=1,
                            charge=0,
                            description='H2O')

o2_molecule = molecule_3
# run classical calculation
molecule = run_pyscf(o2_molecule, run_fci=True, run_ccsd=True)

# get properties from classical SCF calculation
n_electrons = molecule.n_electrons
n_orbitals = 11  # molecule.n_orbitals
n_frozen = 1

print('n_electrons: ', n_electrons)
print('n_orbitals: ', n_orbitals)

# test initial CAS
from openfermionpyscf import prepare_pyscf_molecule
mol = prepare_pyscf_molecule(o2_molecule)
myhf = mol.RHF().run()
mycas = myhf.CASCI(n_orbitals - n_frozen, n_electrons - n_frozen*2).run()
print()

hamiltonian = molecule.get_molecular_hamiltonian()
hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals, frozen_core=n_frozen)
# print(hamiltonian)

print('n_qubits:', hamiltonian.n_qubits)

# Get UCCSD ansatz
pool = get_pool_singlet_sd(n_electrons, n_orbitals, frozen_core=n_frozen)

# print(uccsd_ansatz)
# exit()
# Get reference Hartree Fock state
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits, frozen_core=n_frozen)
print('hf reference', hf_reference_fock)


print('Initialize adaptVQE')
try:
    result = adaptVQE(hamiltonian,
                      pool,
                      hf_reference_fock,
                      # opt_qubits=True,
                      energy_threshold=1e-5,
                      max_iterations=40,
                      # energy_simulator=simulator,
                      # gradient_simulator=simulator,
                      )

except NotConvergedError as e:
    result = e.results


print('Energy HF: {:.8f}'.format(molecule.hf_energy))
print('Energy VQE: {:.8f}'.format(result['energy']))
print('Energy CCSD: {:.8f}'.format(molecule.ccsd_energy))
print('Energy FullCI: {:.8f}'.format(molecule.fci_energy))

print('Error CCSD: {:.8f}'.format(result['energy'] - molecule.ccsd_energy))
print('Error FullCI: {:.8f}'.format(result['energy'] - molecule.fci_energy))

print('Num operators: ', len(result['ansatz']))
print('Ansatz (compact representation):')
# result['ansatz'].print_compact_representation()
# print('Coefficients:\n', result['coefficients'])


density_matrix = get_density_matrix(result['coefficients'], result['ansatz'],
                                    hf_reference_fock, n_orbitals, frozen_core=n_frozen)

#print('fidelity: {:.5f}'.format(density_fidelity(molecule.fci_one_rdm, density_matrix)))

# print(result['iterations'])

# CAS-CI reference calculation
from openfermionpyscf import prepare_pyscf_molecule
mol = prepare_pyscf_molecule(o2_molecule)
myhf = mol.RHF().run()
mycas = myhf.CASCI(n_orbitals - n_frozen, n_electrons - n_frozen*2).run()
#mycas.verbose = 4
#mycas.analyze()
print('Error CASCI: {:.8f}'.format(result['energy'] - mycas.e_tot))


plt.figure()
plt.title('adaptVQE energy')
plt.plot(result['iterations']['energies'])

plt.figure()
plt.title('Error with respect CCSD')
plt.plot(np.array(result['iterations']['energies']) - molecule.ccsd_energy)
plt.yscale('log')

plt.figure()
plt.title('Error with respect FullCI')
plt.plot(np.array(result['iterations']['energies']) - molecule.fci_energy)
plt.yscale('log')


plt.figure()
# CAS-CI reference calculation
from openfermionpyscf import prepare_pyscf_molecule
mol = prepare_pyscf_molecule(o2_molecule)
myhf = mol.RHF().run()
mycas = myhf.CASCI(10, 8).run()

plt.title('Error with respect CAS (10, 8)')
plt.plot(np.array(result['iterations']['energies']) - mycas.e_tot)
plt.yscale('log')

plt.show()
