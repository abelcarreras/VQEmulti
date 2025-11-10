from openfermion import FermionOperator
from vqemulti.utils import fermion_to_qubit
from functools import reduce


def n_particles_operator(n_orbitals, to_pauli=False):

    sz = FermionOperator()
    for i in range(0, n_orbitals * 2):
        a = FermionOperator((i, 1))
        a_dag = FermionOperator((i, 0))
        sz += a * a_dag

    if to_pauli:
        sz = fermion_to_qubit(sz)
    return sz


def spin_square_operator(n_orbitals, to_pauli=False):

    sx = FermionOperator()
    sy = FermionOperator()
    sz = FermionOperator()

    for i in range(0, n_orbitals * 2, 2):
        a_alpha = FermionOperator((i, 1))
        a_alpha_dag = FermionOperator((i, 0))
        a_beta = FermionOperator((i + 1, 1))
        a_beta_dag = FermionOperator((i + 1, 0))

        sx += 0.5 * (a_alpha_dag * a_beta + a_beta_dag * a_alpha)
        sy += 0.5j * (a_alpha_dag * a_beta - a_beta_dag * a_alpha)
        sz += 0.5 * (a_alpha_dag * a_alpha - a_beta_dag * a_beta)

    if to_pauli:
        sz = fermion_to_qubit(sx**2 + sy**2 + sz**2)
    return sz

def spin_z_operator(n_orbitals, to_pauli=False):

    sz = FermionOperator()

    for i in range(0, n_orbitals*2, 2):

        a_alpha = FermionOperator((i, 1))
        a_alpha_dag = FermionOperator((i, 0))
        a_beta = FermionOperator((i + 1, 1))
        a_beta_dag = FermionOperator((i + 1, 0))

        sz += 0.5 * (a_alpha_dag * a_alpha - a_beta_dag * a_beta)

    if to_pauli:
        sz = fermion_to_qubit(sz)
    return sz


def configuration_projector_operator(configuration, to_pauli=False):

    ops = []
    for i, occ in enumerate(configuration):
        if occ == 1:
            ops.append(FermionOperator(f"{i}^ {i}"))
        else:
            ops.append(FermionOperator('') - FermionOperator(f"{i}^ {i}"))  # 1 - n_i

    proj = reduce(lambda a, b: a * b, ops)

    if to_pauli:
        proj = fermion_to_qubit(proj)
    return proj
