from vqemulti.utils import get_hf_reference_in_fock_space, generate_reduced_hamiltonian
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.adapt_vqe import adaptVQE
from vqemulti.preferences import Configuration
from vqemulti.symmetry import get_symmetry_reduced_pool, symmetrize_molecular_orbitals
from openfermionpyscf import run_pyscf
from vqemulti.errors import NotConvergedError
from vqemulti.optimizers import OptimizerParams
from vqemulti.optimizers import adam, rmsprop, sgd, cobyla_mod
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import Session
from openfermion import MolecularData, get_sparse_operator, get_fermion_operator
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore")

config = Configuration()
config.verbose = 1



def get_molecule(basis='sto-3g', distance=1.0):
    """
    read N2 molecule
    :return: pyscf molecule
    """

    from pyscf import gto

    mol = gto.Mole()
    mol.build(
        atom='''N 0 0 0; 
                N 0 0 {}'''.format(distance),
        basis=basis,
    )
    mol.multiplicity = 1
    mol.charge = 0
    mol.spin = 0  # 2j == nelec_alpha - nelec_beta

    return mol


def get_of_molecule(basis='sto-3g', distance=1.0):
    mol = MolecularData(geometry=[['N', [0, 0, 0]],
                                  ['N', [distance, 0, 0]]],
                        basis=basis,
                        multiplicity=1,
                        charge=0,
                        description='molecule')
    return mol


def hf_energy(mol):
    myhf = mol.HF()
    myhf.kernel()
    return myhf.e_tot


def cisd_energy(mol):
    mf = mol.HF().run()
    mycc = mf.CISD().run()
    return mycc.e_tot


def ccsd_energy(mol):
    from pyscf import scf, cc

    mf = scf.RHF(mol).run()
    mycc = cc.CCSD(mf)
    # mycc.diis_space = 50
    mycc.diis_start_cycle = 4
    #mycc.direct = True
    mycc.kernel()

    return mycc.e_tot

# molecule definition

# from vqemulti.utils import generate_reduced_hamiltonian, get_uccsd_operators
# print('len UCC: ', len(get_uccsd_operators(4, 4).terms))

energies_hf = []
energies_cisd = []
energies_ccsd = []
r_range = np.linspace(0.6, 3.5, 30)
for d in r_range:

    n2_mol = get_of_molecule(basis='6-31g', distance=d)


    # run classical calculation
    molecule = run_pyscf(n2_mol, run_fci=False, nat_orb=False, guess_mix=False, verbose=True)
    mol = molecule._pyscf_data['mol']
    print('HF energy: ', hf_energy(mol))

    # get properties from classical SCF calculation
    n_electrons = molecule.n_electrons
    n_orbitals = molecule.n_orbitals_active

    hamiltonian = molecule.get_molecular_hamiltonian()

    from vqemulti.utils import store_hamiltonian
    # store_hamiltonian(hamiltonian, file='hamiltonian.npz')


    hamiltonian = get_fermion_operator(hamiltonian)
    print('H terms (not compress)', len(hamiltonian.terms))
    hamiltonian.compress(1e-2)
    print('H terms (compress)', len(hamiltonian.terms))

    # print(hamiltonian)
    # from openfermion import jordan_wigner
    # print('pauli operators')
    # print(jordan_wigner(hamiltonian))
    # exit()

    # matrix = get_sparse_operator(hamiltonian).toarray()
    #print('lowest eigenvalues:', np.sort(np.linalg.eigvals(matrix))[:4])

    print('N electrons', n_electrons)
    print('N Orbitals', n_orbitals)

    print(molecule.hf_energy)
    energies_hf.append(molecule.hf_energy + 109.0)
    energies_cisd.append(cisd_energy(mol) + 109.0)
    energies_ccsd.append(ccsd_energy(mol) + 109.0)

print(energies_hf)
print(energies_cisd)
print(energies_ccsd)

import matplotlib.pyplot as plt

plt.plot(r_range, energies_hf, label='HF')
plt.plot(r_range, energies_cisd, label='CISD')
plt.plot(r_range, energies_ccsd, label='CCSD')

plt.legend()

plt.ylim(-0.1, 0.8)
plt.xlim(0.5, 3.5)
plt.show()

