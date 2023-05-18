# example of hydrogen molecule using UCCSD ansatz with VQE method
from utils import get_hf_reference_in_fock_space
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from vqe import vqe
import matplotlib.pyplot as plt
import numpy as np
from utils import generate_reduced_hamiltonian, get_uccsd_operators

n_points = 20
vqe_energies = []
hf_energies = []
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

    # Get UCCSD ansatz
    uccsd_ansatz = get_uccsd_operators(n_electrons, n_orbitals)

    print('Initialize VQE')
    result = vqe(hamiltonian,  # fermionic hamiltonian
                 uccsd_ansatz,  # fermionic ansatz
                 hf_reference_fock,
                 exact_energy=True,
                 opt_qubits=False,
                 shots=1000,
                 test_only=True)

    print('Energy VQE: {:.8f}'.format(result['energy']))
    print('Energy HF: {:.8f}'.format(molecule.hf_energy))
    print('Energy FullCI: {:.8f}'.format(molecule.fci_energy))
    print('Energy CCSD: {:.8f}'.format(molecule.ccsd_energy))

    print('Coefficients:\n', result['coefficients'])

    vqe_energies.append(result["energy"])
    hf_energies.append(molecule.hf_energy)
    energies_fullci.append(molecule.fci_energy)
    energies_ccsd.append(molecule.ccsd_energy)

plt.title('Absolute energies')
plt.xlabel('Interatomic distance [Angs]')
plt.ylabel('Energy [H]')

plt.plot(np.linspace(0.3, 3, n_points), vqe_energies, label='VQE')
plt.plot(np.linspace(0.3, 3, n_points), hf_energies, label='HF')
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
diff_hf = np.subtract(hf_energies, energies_fullci)

plt.plot(np.linspace(0.3, 3, n_points), diff_fullci, label='VQE')
plt.plot(np.linspace(0.3, 3, n_points), diff_ccsd, label='CCSD')
plt.plot(np.linspace(0.3, 3, n_points), diff_hf, label='HF')

plt.legend()

plt.show()
