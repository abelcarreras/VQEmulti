import pennylane as qml
from pennylane import numpy as np


symbols = ["He", "H"]
geometry = np.array([[0.00000000, 0.00000000, -0.87818361],
                     [0.00000000, 0.00000000,  0.87818362]])

H, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry, charge=1, basis='sto-3g')



generators = qml.symmetry_generators(H)
paulixops = qml.paulix_ops(generators, qubits)


for idx, generator in enumerate(generators):
    print(f"generator {idx+1}: {generator}, paulix_op: {paulixops[idx]}")


n_electrons = 2
paulix_sector = qml.qchem.optimal_sector(H, generators, n_electrons)
print(paulix_sector)


H_tapered = qml.taper(H, generators, paulixops, paulix_sector)
print(H_tapered)


H_sparse = qml.SparseHamiltonian(H.sparse_matrix(), wires=H.wires)
H_tapered_sparse = qml.SparseHamiltonian(H_tapered.sparse_matrix(), wires=H_tapered.wires)

print("Eigenvalues of H:\n", qml.eigvals(H_sparse, k=16))
print("\nEigenvalues of H_tapered:\n", qml.eigvals(H_tapered_sparse, k=4))


state_tapered = qml.qchem.taper_hf(generators, paulixops, paulix_sector,
                                   num_electrons=n_electrons, num_wires=len(H.wires))
print(state_tapered)


dev = qml.device("default.qubit", wires=H.wires)
@qml.qnode(dev, interface="autograd")
def circuit():
    qml.BasisState(np.array([1, 1, 0, 0]), wires=H.wires)
    return qml.state()

qubit_state = circuit()
HF_energy = qubit_state.T @ H.sparse_matrix().toarray() @ qubit_state
print(f"HF energy: {np.real(HF_energy):.8f} Ha")

dev = qml.device("default.qubit", wires=H_tapered.wires)
@qml.qnode(dev, interface="autograd")
def circuit():
    qml.BasisState(np.array([1, 1]), wires=H_tapered.wires)
    return qml.state()

qubit_state = circuit()
HF_energy = qubit_state.T @ H_tapered.sparse_matrix().toarray() @ qubit_state
print(f"HF energy (tapered): {np.real(HF_energy):.8f} Ha")



from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from vqemulti.utils import generate_reduced_hamiltonian, get_hf_reference_in_fock_space
from vqemulti.pool import get_pool_singlet_sd
from vqemulti import vqe

h2_molecule = MolecularData(geometry=[['H', [0, 0, 0]],
                                      ['H', [0, 0, 1.0]]],
                            basis='3-21g',
                            multiplicity=1,
                            charge=0,
                            description='H2')

# run classical calculation
molecule = run_pyscf(h2_molecule, run_fci=True, run_ccsd=True)

# get properties from classical SCF calculation
n_electrons = molecule.n_electrons
n_orbitals = molecule.n_orbitals_active

print('n_electrons: ', n_electrons)
print('n_orbitals: ', n_orbitals)

# obtain hamiltonian in fermion operators
hamiltonian = molecule.get_molecular_hamiltonian()
hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals)

print('n_qubits:', hamiltonian.n_qubits)

# Get UCCSD ansatz in fermion operators
uccsd_ansatz = get_pool_singlet_sd(n_electrons, n_orbitals, frozen_core=7)

# Get reference Hartree Fock state in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)
print('hf reference', hf_reference_fock)

# Define Pennylane simulator
from vqemulti.simulators.penny_simulator import PennylaneSimulator as Simulator

simulator = Simulator(trotter=True,     # use Trotter transformation
                      trotter_steps=1,  # define Trotter transformation steps
                      shots=1000)       # define number of shots

print('Initialize VQE')
result = vqe(hamiltonian,                # hamiltonian in fermion operators
             uccsd_ansatz,               # ansatz in fermion operators
             hf_reference_fock,          # reference vector in Fock space
             energy_simulator=simulator) # use simulator

print('Energy HF: {:.8f}'.format(molecule.hf_energy))
print('Energy VQE: {:.8f}'.format(result['energy']))
print('Energy CCSD: {:.8f}'.format(molecule.ccsd_energy))
print('Energy FullCI: {:.8f}'.format(molecule.fci_energy))

print('Num operators: ', len(result['ansatz']))
print('Ansatz:\n', result['ansatz'])
print('Coefficients:\n', result['coefficients'])

