from vqemulti.utils import get_hf_reference_in_fock_space
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from vqemulti import vqe
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.simulators.qiskit_simulator import QiskitSimulator
from vqemulti.preferences import Configuration
from vqemulti.energy import get_vqe_energy


conf = Configuration()
# set mapping
conf.mapping = 'bk'  # jw: Jordan-wigner , bk: Bravyi-Kitaev, pc: parity transform
# set printing level
conf.verbose = 1  # show optimization function info

h2_molecule = MolecularData(geometry=[['H', [0, 0, 0]],
                                      ['H', [0, 0, 0.8]]],
                            basis='sto-3g',
                            multiplicity=1,
                            charge=0,
                            description='H2')


molecule = run_pyscf(h2_molecule, run_fci=True, run_ccsd=True)

# get properties from classical SCF calculation
n_electrons = molecule.n_electrons
n_orbitals = molecule.n_orbitals
hamiltonian = molecule.get_molecular_hamiltonian()


# Get Hartree Fock reference in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

# Get UCCSD ansatz
uccsd_ansatz = get_pool_singlet_sd(n_electrons, n_orbitals)

# define simulator
simulator = QiskitSimulator(trotter=True,
                            test_only=True,  # test only (evaluate circuit exactly)
                            )

print('Initialize VQE')
result = vqe(hamiltonian,  # fermionic hamiltonian
             uccsd_ansatz,  # fermionic ansatz
             hf_reference_fock,
             energy_simulator=simulator)

print('Energy HF: {:.8f}'.format(molecule.hf_energy))
print('Energy VQE: {:.8f}'.format(result['energy']))
print('Energy FullCI: {:.8f}'.format(molecule.fci_energy))
print('Energy CCSD: {:.8f}'.format(molecule.ccsd_energy))

print('Ansatz:\n', result['ansatz'])
print('Coefficients:\n', result['coefficients'])

simulator.print_statistics()

# perform a single call to energy evaluator an use new simulator object to store only one circuit
simulator_sp = QiskitSimulator(trotter=True, test_only=True)

energy = get_vqe_energy(result['coefficients'],
                        result['ansatz'],
                        hf_reference_fock,
                        hamiltonian,
                        energy_simulator=simulator_sp)

print('Energy VQE: {:.8f}'.format(energy))

# print circuits
simulator_sp.print_circuits()
