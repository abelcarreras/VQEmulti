from vqemulti.utils import get_hf_reference_in_fock_space, generate_reduced_hamiltonian
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.adapt_vqe import adaptVQE
from vqemulti.energy import get_vqe_energy, get_adapt_vqe_energy
from vqemulti.energy.simulation import simulate_adapt_vqe_variance, simulate_energy_sqd
from vqemulti.preferences import Configuration
from openfermionpyscf import run_pyscf
from openfermion import MolecularData, get_sparse_operator, get_fermion_operator
from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
from vqemulti.ansatz import get_ucc_ansatz
import numpy as np
import warnings


def get_n_particles_operator(n_orbitals, to_pauli=False):
    from openfermion import FermionOperator, jordan_wigner

    sz = FermionOperator()
    for i in range(0, n_orbitals * 2):
        a = FermionOperator((i, 1))
        a_dag = FermionOperator((i, 0))
        sz += a * a_dag

    if to_pauli:
        sz = jordan_wigner(sz)
    return sz


warnings.filterwarnings("ignore")

config = Configuration()
config.verbose = True
config.mapping = 'bk'

warnings.filterwarnings("ignore")

def get_molecule(filename):
    """
    read molecule from xyz file

    :param filename: XYZ file
    :return: pyscf molecule
    """

    with open(filename, 'r') as f:
        lines = f.readlines()[2:]

    symbols = []
    coordinates = []

    for line in lines:
        symbols.append(line.split()[0])
        coordinates.append(line.split()[1:])

    coordinates = np.array(coordinates, dtype=float)

    geometry = []
    for s, c in zip(symbols, coordinates):
        geometry.append([s, [c[0], c[1], c[2]]])

    mol = MolecularData(geometry=geometry,
                        basis='sto-3g',
                        multiplicity=1,
                        charge=0,
                        description='molecule')

    return mol


simulator = Simulator(trotter=True,
                      trotter_steps=1,
                      test_only=True,
                      hamiltonian_grouping=True,
                      use_estimator=True,
                      shots=2000)


hydrogen = get_molecule('H4.xyz')

# run classical calculation
molecule = run_pyscf(hydrogen, run_fci=True, nat_orb=False, guess_mix=False, verbose=True,
                     run_ccsd=True)

print('FCI energy: ', molecule.fci_energy)

hamiltonian = molecule.get_molecular_hamiltonian()

n_electrons = molecule.n_electrons
n_orbitals = molecule.n_orbitals
n_qubits = n_orbitals * 2

hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, n_qubits)


hf_energy = get_vqe_energy([], [], hf_reference_fock, hamiltonian, None)
print('HF energy: ', hf_energy)

coefficients, ansatz = get_ucc_ansatz(molecule.ccsd_single_amps, molecule.ccsd_double_amps)

energy = get_vqe_energy(coefficients,
                        ansatz,
                        hf_reference_fock,
                        hamiltonian,
                        simulator)

print('UCC energy: ', energy)
simulator.print_statistics()


energy = simulate_energy_sqd(coefficients,
                             ansatz,
                             hf_reference_fock,
                             hamiltonian,
                             simulator,
                             n_electrons,
                             generate_random=False,
                             adapt=False)
print('SDQ energy: ', energy)

