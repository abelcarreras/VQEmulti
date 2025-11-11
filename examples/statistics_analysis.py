# example of the use of IBM runtime and sessions
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from vqemulti import vqe
from vqemulti.utils import generate_reduced_hamiltonian, get_hf_reference_in_fock_space
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.simulators.qiskit_simulator import QiskitSimulator
# from simulators.cirq_simulator import CirqSimulator as QiskitSimulator
# from simulators.penny_simulator import PennylaneSimulator as QiskitSimulator

from qiskit_ibm_runtime import Session


he_molecule = MolecularData(geometry=[['He', [0, 0, 0]],
                                      ['He', [0, 0, 1.0]]],
                            basis='3-21g',
                            # basis='sto-3g',
                            multiplicity=1,
                            charge=-2,
                            description='He2')

# run classical calculation
molecule = run_pyscf(he_molecule, run_fci=True, run_ccsd=True)

# Use a restricted MO space (2 frozen, 2 active)
n_electrons = molecule.n_electrons
n_orbitals = 4  # molecule.n_orbitals

print('n_electrons: ', n_electrons)
print('n_orbitals: ', n_orbitals)

hamiltonian = molecule.get_molecular_hamiltonian()
hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals, frozen_core=2)

print('n_qubits:', hamiltonian.n_qubits)

# Get UCCSD ansatz
uccsd_ansatz = get_pool_singlet_sd(n_electrons, n_orbitals, frozen_core=2)

# Get reference Hartree Fock state
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits, frozen_core=2)
print('hf reference', hf_reference_fock)
print('Energy HF: {:.8f}'.format(molecule.hf_energy))

# Start session
print('Initialize VQE')

energies = []
with Session(backend="ibmq_qasm_simulator") as session:
    for i in range(1000):
        # Qiskit Simulator
        simulator = QiskitSimulator(trotter=True,
                                    trotter_steps=1,
                                    shots=1000,
                                    session=session,
                                    #test_only=True,
                                    )

        result = vqe(hamiltonian,
                     uccsd_ansatz,
                     hf_reference_fock,
                     energy_simulator=simulator,
                     opt_qubits=False)

        # print('Energy HF: {:.8f}'.format(molecule.hf_energy))
        print('Energy VQE: {:.8f} {}'.format(result['energy'], i))
        energies.append(result['energy'])

import matplotlib.pyplot as plt

plt.plot(energies, '.')
plt.figure()
plt.hist(energies)
plt.show()
