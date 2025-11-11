from vqemulti.utils import get_hf_reference_in_fock_space, generate_reduced_hamiltonian
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.adapt_vqe import adaptVQE
from vqemulti.energy import get_adapt_vqe_energy
from vqemulti.energy.simulation import simulate_adapt_vqe_variance, simulate_adapt_vqe_energy_sqd
from vqemulti.preferences import Configuration
from vqemulti.symmetry import get_symmetry_reduced_pool, symmetrize_molecular_orbitals
from openfermionpyscf import run_pyscf
from vqemulti.errors import NotConvergedError
from vqemulti.basis_projection import get_basis_overlap_matrix, project_basis, prepare_ansatz_for_restart
from vqemulti.density import get_density_matrix, density_fidelity
from vqemulti.optimizers import OptimizerParams
from vqemulti.optimizers import adam, rmsprop, sgd, cobyla_mod
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import Session
from openfermion import MolecularData, get_sparse_operator, get_fermion_operator
from vqemulti.gradient import compute_gradient_vector, simulate_gradient
from vqemulti.utils import store_hamiltonian, store_wave_function, load_wave_function, load_hamiltonian
from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
import numpy as np
import warnings
import pyscf


warnings.filterwarnings("ignore")

config = Configuration()
config.verbose = True
config.mapping = 'jw'

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
# simulator = None


hydrogen = get_molecule('H4.xyz')

# run classical calculation
frozen_orb = 0
n_total = 4
molecule = run_pyscf(hydrogen, run_fci=False, nat_orb=False, guess_mix=False, verbose=True,
                     frozen_core=frozen_orb,
                     n_orbitals=n_total)

# get coefficinets CCSD
n_frozen = 0
active_space = range(n_frozen, molecule.n_orbitals)
frozen = [i for i in range(molecule.n_orbitals) if i not in active_space]



from pyscf.cc import CCSD
scf = pyscf.scf.RHF(molecule._pyscf_data['mol']).run()
print('scf: ', len(scf.mo_coeff))
ccsd = CCSD(scf).run()


occupied_indices = np.arange(ccsd.nocc)
virtual_indices = np.arange(ccsd.nocc, ccsd.nmo)


t1 = ccsd.t1
t2 = ccsd.t2

print(t1.shape)
print(t2.shape)

from vqemulti.pool.tools import OperatorList, hermitian_conjugated, normal_ordered
from openfermion import FermionOperator

n_vir = ccsd.nmo - ccsd.nocc
n_occ = ccsd.nocc

print('n_orb: ',molecule.n_orbitals)
print('occupied: ', n_occ)
print('virtual: ', n_vir)

print('shape: t1', np.shape(t1))
print('shape: t2', np.shape(t2))

tol = 1e-8
operators = []
coefficients = []
for i in range(n_occ):
    for j in range(n_vir):
        print('i j', t1[i, j])
        if abs(t1[i, j]) > tol:
            for s in range(2): # spin
                op1 = FermionOperator('{}^ {}'.format(2*i+s, n_occ + 2*j+s))
                coefficients.append(t1[i, j])
                operators.append(op1 - hermitian_conjugated(op1))

                # check anti-hermitian
                assert normal_ordered(operators[-1] + hermitian_conjugated(operators[-1])).isclose(FermionOperator.zero(), tol)

print()
print(operators)


from numpy.linalg import svd
M = np.tensordot(t2, t2, axes=([2, 3], [2, 3]))  # dim (occ,vir)-(occ,vir)
# Descomposició SVD
U, S, Vh = svd(M, full_matrices=False)
print('U')
print(U)
print('S')
print(S)
print('Vh')
print(Vh)

print('singular values')
singular_values = [np.diag(S[k]) for k in range(S.shape[0])]
print(singular_values)

"""
stats = []
max_rank=5
tol=1e-6
for k in range(min(max_rank, len(S))):
    u = U[:, k]
    stat = u.reshape((n_occ, n_vir))
    stats.append((S[k], stat))

print('-------------')
print(stats)

"""

j_matrix = np.zeros((molecule.n_orbitals, molecule.n_orbitals))
for i in range(n_occ):
    for j in range(n_vir):
        j_matrix[i, j + n_occ] = np.sum(t2[i, j, :, :])
        j_matrix[j+ n_occ, i] = j_matrix[i, j + n_occ]

evals, evecs = np.linalg.eigh(j_matrix)

j_matrix = np.zeros((molecule.n_orbitals, molecule.n_orbitals))
for eval, evec in zip(evals, evecs):
    j_matrix += eval*np.outer(evec, evec)

for i in range(molecule.n_orbitals):
    for j in range(i+1, molecule.n_orbitals):
        if abs(j_matrix[i, j]) > tol:
            for s1 in range(2):
                for s2 in range(2):
                    # alpha
                    op1 = FermionOperator('{0:}^ {0:} {1:}^ {1:}'.format(2*i+s1, 2*j+s2))
                    coefficients.append(j_matrix[i, j])
                    operators.append(1j * op1)

                    # check anti-hermitian
                    assert normal_ordered(operators[-1] + hermitian_conjugated(operators[-1])).isclose(FermionOperator.zero(), tol)


ansatz = OperatorList(operators, normalize=False, antisymmetrize=False)
#for op, coef in zip(ansatz, coefficients):
#    print(coef, ':', op)
#    print()

#exit()

hamiltonian = molecule.get_molecular_hamiltonian()

#print(hf_reference_fock)
n_electrons = molecule.n_electrons - frozen_orb * 2
n_orbitals = n_total - frozen_orb  # molecule.n_orbitals

hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, n_orbitals*2)

hf_energy = get_adapt_vqe_energy([], [], hf_reference_fock, hamiltonian, None)
print('energy HF: ', hf_energy)
# exit()

print('ansatz')
print(ansatz)
print(coefficients)

energy = get_adapt_vqe_energy(coefficients,
                              ansatz,
                              hf_reference_fock,
                              hamiltonian,
                              simulator
                              )

print('energy ANSATZ: ', energy, '(', hf_energy - energy, ')')
energy = simulate_adapt_vqe_energy_sqd(coefficients,
                                       ansatz,
                                       hf_reference_fock,
                                       hamiltonian,
                                       simulator,
                                       n_electrons,
                                       generate_random=False)
print('final SDQ energy: ', energy)
