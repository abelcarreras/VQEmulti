from vqemulti.utils import get_hf_reference_in_fock_space, generate_reduced_hamiltonian
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.adapt_vqe import adaptVQE
from vqemulti.preferences import Configuration
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
import matplotlib.pyplot as plt
import numpy as np
from vqemulti.errors import NotConvergedError
from vqemulti.basis_projection import get_basis_overlap_matrix, project_basis, prepare_ansatz_for_restart


vqe_energies = []
energies_fullci = []
energies_hf = []

Configuration().verbose = True

# molecule definition
from generate_mol import tetra_h4_mol, linear_h4_mol, square_h4_mol
#h4_molecule = tetra_h4_mol(distance=5.0, basis='sto-3g')
h4_molecule = linear_h4_mol(distance=3.0, basis='3-21g')
#h4_molecule = square_h4_mol(distance=3.0, basis='sto-3g')

# run classical calculation
molecule = run_pyscf(h4_molecule, run_fci=True, nat_orb=False, guess_mix=False)

print('FullCI energy result:', molecule.fci_energy)

# get properties from classical SCF calculation
n_electrons = molecule.n_electrons
n_orbitals = 4  # molecule.n_orbitals
hamiltonian = molecule.get_molecular_hamiltonian()
hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals)
print('N electrons', n_electrons)
print('N Orbitals', n_orbitals)
# Choose specific pool of operators for adapt-VQE
pool = get_pool_singlet_sd(n_electrons=n_electrons, n_orbitals=n_orbitals)
pool.print_compact_representation()
#print(pool)
#exit()


def operator_dot(operator_1, operator_2):
    """
    compute overlap between two operators

    :param operator_1: operator 1
    :param operator_2: operator 2
    :return:
    """
    sum = 0
    for term_op1, coeff_op1 in operator_1.terms.items():
        for term_op2, coeff_op2 in operator_2.terms.items():
            if term_op1 == term_op2:
                sum += coeff_op1 * coeff_op2
    return sum



from vqemulti.pool.tools import OperatorList

def ansatz_projection(ansatz_ref, ansatz, max_val=1e-2):
    """
    prohect ansatz (operator list)  into another ansatz (operator list)
    :param ansatz_ref: reference ansatz (result will be expressed in this basis)
    :param ansatz: target ansatz
    :param max_val: truncate the operators with coefficients lower than to this value

    :return: coefficients, projected ansatz (operators list)
    """

    coefficients = []
    operator_list = []
    for op in ansatz_ref:
        overlap = operator_dot(op, ansatz)
        if abs(overlap) > max_val:
            coefficients.append(overlap)
            operator_list.append(op)

    return coefficients, OperatorList(operator_list, antisymmetrize=False)


ansatz = sum([op for op in pool])

c, a = ansatz_projection(pool, ansatz, max_val=0)

print(c)