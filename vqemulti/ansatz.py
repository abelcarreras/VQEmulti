from openfermion import FermionOperator, hermitian_conjugated, normal_ordered
from vqemulti.pool.tools import OperatorList
import numpy as np


def get_ucc_ansatz(t1, t2, tolerance=1e-6):
    """
    Get unitary coupled cluster ansatz from 1-excitation and 2-excitation CC amplitudes

    :param t1: 1-excitation amplitudes in spin-orbital basis ( n x n )
    :param t2: 1-excitation amplitudes in spin-orbital basis ( n x n x n x n )
    :param tolerance: amplitude cutoff to include term in the ansatz
    :return: coefficients as list and ansatz as OperatorsList
    """

    n_spin_orbitals = len(t1)

    coefficients = []
    operators = []

    # 1-exciation terms
    for i in range(n_spin_orbitals):
        for j in range(n_spin_orbitals):
            if abs(molecule.ccsd_single_amps[i,j]) > tolerance:
                operator = FermionOperator('{}^ {}'.format(i, j))
                coefficients.append(molecule.ccsd_single_amps[i, j])
                operators.append(operator - hermitian_conjugated(operator))

    # 2-excitation terms
    for i in range(n_spin_orbitals):
        for j in range(n_spin_orbitals):
            for k in range(n_spin_orbitals):
                for l in range(j, n_spin_orbitals):  # avoid duplicates
                    if abs(t2[i, j, k, l]) > tolerance:
                        if np.mod(i, 2) + np.mod(k, 2) == np.mod(j, 2) + np.mod(l, 2):  # keep spin symmetry
                            operator = FermionOperator('{}^ {} {}^ {}'.format(i, j, k, l))
                            coefficients.append(t2[i, j, k, l])
                            operators.append(operator - hermitian_conjugated(operator))

                            # assert normal_ordered(operators[-1] + hermitian_conjugated(operators[-1])).isclose(FermionOperator.zero(), tol)

    assert normal_ordered(sum(operators) + hermitian_conjugated(sum(operators))).isclose(FermionOperator.zero(), tolerance)

    ansatz = OperatorList(operators, normalize=False, antisymmetrize=False)

    return coefficients, ansatz


if __name__ == '__main__':

    from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
    from openfermionpyscf import run_pyscf
    from openfermion import MolecularData
    from vqemulti.utils import get_hf_reference_in_fock_space
    from vqemulti.energy import get_vqe_energy, get_adapt_vqe_energy
    from vqemulti.operators import n_particles_operator, spin_z_operator, spin_square_operator


    simulator = Simulator(trotter=True,
                          trotter_steps=1,
                          test_only=True,
                          hamiltonian_grouping=True,
                          use_estimator=True)

    hydrogen = MolecularData(geometry=[('H', [0.0, 0.0, 0.0]),
                                       ('H', [2.0, 0.0, 0.0]),
                                       ('H', [4.0, 0.0, 0.0]),
                                       ('H', [6.0, 0.0, 0.0])],
                        basis='sto-3g',
                        multiplicity=1,
                        charge=0,
                        description='molecule')

    # run classical calculation
    n_frozen_orb = 0
    n_total_orb = 4
    molecule = run_pyscf(hydrogen, run_fci=False, nat_orb=False, guess_mix=False, verbose=True,
                         frozen_core=n_frozen_orb, n_orbitals=n_total_orb, run_ccsd=True)

    hamiltonian = molecule.get_molecular_hamiltonian()

    n_electrons = molecule.n_electrons - n_frozen_orb * 2
    n_orbitals = n_total_orb - n_frozen_orb  # molecule.n_orbitals
    n_qubits = n_orbitals * 2

    hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, n_qubits)

    hf_energy = get_vqe_energy([], [], hf_reference_fock, hamiltonian, None)
    print('energy HF: ', hf_energy)

    coefficients, ansatz = get_ucc_ansatz(molecule.ccsd_single_amps, molecule.ccsd_double_amps)

    energy = get_vqe_energy(coefficients,
                            ansatz,
                            hf_reference_fock,
                            hamiltonian,
                            simulator)

    print('UCC ANSATZ energy: ', energy)

    n_particle = get_vqe_energy(coefficients,
                                ansatz,
                                hf_reference_fock,
                                n_particles_operator(n_orbitals),
                                simulator)

    spin_z = get_vqe_energy(coefficients,
                                ansatz,
                                hf_reference_fock,
                                spin_z_operator(n_orbitals),
                                simulator)

    spin_square = get_vqe_energy(coefficients,
                                ansatz,
                                hf_reference_fock,
                                spin_square_operator(n_orbitals),
                                simulator)

    print('n_particles: ', n_particle)
    print('Sz: ', spin_z)
    print('S2: ', spin_square)
