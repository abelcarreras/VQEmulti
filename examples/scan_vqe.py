# example of hydrogen molecule using UCCSD ansatz with VQE method
from utils import get_hf_reference_in_fock_space
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from vqe import vqe
import matplotlib.pyplot as plt
import numpy as np
import openfermion
from utils import generate_reduced_hamiltonian


n_points = 20
vqe_energies = []
vqe_energies_Nonia = []
energies_fullci = []
energies_ccsd = []
for d in np.linspace(0.3, 3, n_points):

    # molecule definition
    h2_molecule = MolecularData(geometry=[['H', [0, 0, 0]],
                                          ['H', [0, 0, d]]],
                                basis='6-31g',
                                multiplicity=1,
                                charge=0,
                                description='H2')

    # run classical calculation
    molecule = run_pyscf(h2_molecule, run_fci=True, run_ccsd=True)

    # get properties from classical SCF calculation
    n_electrons = 2 # molecule.n_electrons
    n_orbitals = 2  # molecule.n_orbitals
    hamiltonian = molecule.get_molecular_hamiltonian()
    hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals)
    # Get Hartree Fock reference in Fock space
    hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

    # Prepare UCCSD ansatz
    packed_amplitudes = openfermion.uccsd_singlet_get_packed_amplitudes(molecule.ccsd_single_amps,
                                                                        molecule.ccsd_double_amps,
                                                                        n_orbitals * 2,
                                                                        n_electrons)
    packed_amplitudes = np.ones_like(packed_amplitudes)
    uccsd_ansatz = openfermion.uccsd_singlet_generator(packed_amplitudes,
                                                       n_orbitals * 2,
                                                       n_electrons)

    print('Initialize VQE')
    result = vqe(hamiltonian,  # fermionic hamiltonian
                 uccsd_ansatz,  # fermionic ansatz
                 hf_reference_fock,
                 exact_energy=False,
                 shots=1000,
                 test_only=True)

    print('Energy VQE: {:.8f}'.format(result['energy']))
    print('Energy VQE Nonia: {:.8f}'.format(result['energy_Nonia']))
    print('Energy FullCI: {:.8f}'.format(molecule.fci_energy))
    print('Energy CCSD: {:.8f}'.format(molecule.ccsd_energy))

    print('Coefficients:\n', result['coefficients'])
    print('Coefficients Nonia:\n', result['coefficients_Nonia'])

    vqe_energies.append(result["energy"])
    vqe_energies_Nonia.append(result['energy_Nonia'])
    energies_fullci.append(molecule.fci_energy)
    energies_ccsd.append(molecule.ccsd_energy)

plt.title('Absolute energies')
plt.xlabel('Interatomic distance [Angs]')
plt.ylabel('Energy [H]')

plt.plot(np.linspace(0.3, 3, n_points), vqe_energies, label='VQE')
plt.plot(np.linspace(0.3, 3, n_points), vqe_energies_Nonia, label='VQE_Nonia')
plt.plot(np.linspace(0.3, 3, n_points), energies_fullci, label='FullCI')
plt.plot(np.linspace(0.3, 3, n_points), energies_ccsd, label='CCSD')
plt.legend()

plt.figure()
plt.title('Difference between fullCI')
plt.xlabel('Interatomic distance [Angs]')
plt.ylabel('Energy [H]')
plt.yscale('log')

diff_fullci = np.subtract(vqe_energies, energies_fullci)
diff_ccsd = np.subtract(energies_ccsd, energies_fullci)
plt.plot(np.linspace(0.3, 3, n_points), diff_fullci, label='VQE')
plt.plot(np.linspace(0.3, 3, n_points), diff_ccsd, label='CCSD')
plt.legend()

plt.show()
