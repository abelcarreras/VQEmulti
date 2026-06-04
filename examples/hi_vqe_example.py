from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from vqemulti.utils import get_hf_reference_in_fock_space
from vqemulti.pool.singlet_sd import get_pool_singlet_sd
from vqemulti.ansatz.unitary_jastrow import UnitaryCoupledJastrowAnsatz
from vqemulti.ansatz.exponential import ExponentialAnsatz
from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
from vqemulti.preferences import Configuration
from vqemulti.optimizers import OptimizerParams
from vqemulti.hi_vqe import hi_vqe
import matplotlib.pyplot as plt
import numpy as np


Configuration().mapping = 'jw'
Configuration().verbose = True
Configuration().temp_dir = '/Users/abel/TEST/DICE'

# singlet reference with restricted orbitals
nitrogen = MolecularData(geometry=[('N', [0.0, 0.0, 0.0]),
                                   ('N', [2.0, 0.0, 0.0])],
                         basis='sto-3g',
                         multiplicity=1,
                         charge=0,
                         description='molecule')

# run reference calculation
molecule = run_pyscf(nitrogen, run_fci=False, nat_orb=False, guess_mix=False, verbose=True,
                     frozen_core=4, n_orbitals=10, run_ccsd=True, run_casci=True)

hamiltonian = molecule.get_molecular_hamiltonian()
# print(hamiltonian)

# modify number of electrons and multiplicity for actual calculation
n_electrons = molecule.n_electrons + 1
n_orbitals = molecule.n_orbitals
multiplicity = molecule.multiplicity + 1

print('n_electrons: ', n_electrons)
print('n_orbitals: ', n_orbitals)
print('multiplicity', multiplicity)
print('n_qubits:', hamiltonian.n_qubits)


# Get reference Hartree Fock state
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits, multiplicity)
print('hf reference', hf_reference_fock)

# Build UCC ansatz
uccsd_pool = get_pool_singlet_sd(n_electrons, n_orbitals)
initial_parameters = np.zeros_like(uccsd_pool)
uccsd_ansatz = ExponentialAnsatz(initial_parameters, uccsd_pool, hf_reference_fock)

# Build LUCJ ansatz
ccsd = molecule._pyscf_data.get('ccsd', None)
ucja_ansatz = UnitaryCoupledJastrowAnsatz(ccsd.t1, ccsd.t2, hf_reference_fock, n_terms=1, full_trotter=True)

# Simulator
simulator = Simulator(trotter=False,
                      trotter_steps=1,
                      test_only=True,
                      shots=10000)

# sampling
#print(ucja.get_sampling(simulator))

# SQD parameters for Hi-VQE
sqd_conf = {'recovery_type': 1}

# define optimizer
opt_cobyla = OptimizerParams(method='COBYLA', options={'rhobeg': 0.1})

# Run Hi-VQE
print('Initialize Hi-VQE')
result = hi_vqe(hamiltonian,
                ucja_ansatz,
                energy_simulator=simulator,
                optimizer_params=opt_cobyla,
                sqd_params=sqd_conf,
                )

# print results
print('Energy HF: {:.8f}'.format(molecule.hf_energy))
print('Energy VQE: {:.8f}'.format(result['energy']))
print('Energy CCSD: {:.8f}'.format(molecule.ccsd_energy))
print('Energy CASCI: {:.8f}'.format(molecule.casci_energy))

print('Num operators: ', len(result['ansatz']))
#print('Ansatz:\n', result['ansatz']._operators)
print('Coefficients:\n', result['coefficients'])
repr(result)

simulator.print_statistics()

