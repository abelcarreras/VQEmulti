from vqemulti.energy import exact_vqe_energy, get_vqe_energy, simulate_vqe_energy
from vqemulti.gradient.simulation import simulate_vqe_energy_gradient
from vqemulti.gradient import exact_vqe_energy_gradient
from vqemulti.pool.tools import OperatorList
from vqemulti.optimizers import OptimizerParams
from vqemulti.preferences import Configuration
import numpy as np
import scipy


def vqe(hamiltonian,
        ansatz,
        hf_reference_fock,
        coefficients=None,
        energy_simulator=None,
        energy_threshold=1e-4,
        optimizer_params=None
        ):
    """
    Perform a VQE calculation

    :param hamiltonian: hamiltonian in fermionic operators
    :param ansatz: ansatz (fermionic operator list) to optimize
    :param coefficients: initial guess coefficients (leave None to initialize to zero)
    :param opt_qubits: choose basis of optimization (True: qubits operators, False: fermion operators)
    :param hf_reference_fock: HF reference in Fock space vector (occupations)
    :param energy_simulator: Simulator object used to obtain the energy, if None do not use simulator (exact)
    :param energy_threshold: energy convergence threshold for classical optimization (in Hartree)
    :return: results dictionary
    """

    # set default optimizer params
    if optimizer_params is None:
        optimizer_params = OptimizerParams()

    print('optimizer params: ', optimizer_params)

    # transform to qubit hamiltonian
    ansatz = OperatorList(ansatz)

    # initial guess
    n_terms = len(ansatz)
    if coefficients is None:
        coefficients = np.zeros(n_terms)

    assert len(coefficients) == len(ansatz)

    # check if no coefficients
    if n_terms == 0:
        energy = get_vqe_energy(coefficients, ansatz, hf_reference_fock, hamiltonian, energy_simulator)
        return {'energy': energy, 'coefficients': [], 'ansatz': ansatz, 'f_evaluations': 0}

    # Optimize the results from analytical calculation
    if energy_simulator is None:
        results = scipy.optimize.minimize(exact_vqe_energy,
                                          coefficients,
                                          (ansatz, hf_reference_fock, hamiltonian),
                                          jac=exact_vqe_energy_gradient,
                                          method=optimizer_params.method,
                                          options=optimizer_params.options,
                                          tol=energy_threshold,
                                          )

    else:
        results = scipy.optimize.minimize(simulate_vqe_energy,
                                          coefficients,
                                          (ansatz, hf_reference_fock, hamiltonian, energy_simulator),
                                          jac=simulate_vqe_energy_gradient,
                                          method=optimizer_params.method,
                                          options=optimizer_params.options,
                                          tol=energy_threshold,
                                          )

    return {'energy': results.fun,
            'coefficients': list(results.x),
            'ansatz': ansatz,
            'f_evaluations': results.nfev}



if __name__ == '__main__':

    from openfermion import MolecularData
    from openfermionpyscf import run_pyscf
    from utils import generate_reduced_hamiltonian, get_hf_reference_in_fock_space
    from pool.singlet_sd import get_pool_singlet_sd
    from vqemulti.preferences import Configuration

    # set Bravyi-Kitaev mapping
    Configuration().mapping = 'bk'
    Configuration().verbose = True

    h2_molecule = MolecularData(geometry=[['He', [0, 0, 0]],
                                          ['He', [0, 0, 1.0]]],
                                basis='3-21g',
                                # basis='sto-3g',
                                multiplicity=1,
                                charge=-2,
                                description='H2')

    # run classical calculation
    molecule = run_pyscf(h2_molecule, run_fci=True, run_ccsd=True)

    # get properties from classical SCF calculation
    n_electrons = molecule.n_electrons
    n_orbitals = 4  # molecule.n_orbitals

    print('n_electrons: ', n_electrons)
    print('n_orbitals: ', n_orbitals)

    hamiltonian = molecule.get_molecular_hamiltonian()
    hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals, frozen_core=2)
    # print(hamiltonian)

    print('n_qubits:', hamiltonian.n_qubits)

    # Get UCCSD ansatz
    uccsd_ansatz = get_pool_singlet_sd(n_electrons, n_orbitals, frozen_core=2)
    # uccsd_ansatz = []

    # Get reference Hartree Fock state
    hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits, frozen_core=2)
    print('hf reference', hf_reference_fock)

    # Simulator
    # from simulators.penny_simulator import PennylaneSimulator as Simulator
    from simulators.cirq_simulator import CirqSimulator as Simulator
    # from simulators.qiskit_simulator import QiskitSimulator as Simulator

    simulator = Simulator(trotter=False,
                          trotter_steps=1,
                          test_only=True,
                          shots=10000)

    print('Initialize VQE')
    result = vqe(hamiltonian,
                 uccsd_ansatz,
                 hf_reference_fock,
                 # energy_simulator=simulator,
                 opt_qubits=False)

    print('Energy HF: {:.8f}'.format(molecule.hf_energy))
    print('Energy VQE: {:.8f}'.format(result['energy']))
    print('Energy CCSD: {:.8f}'.format(molecule.ccsd_energy))
    print('Energy FullCI: {:.8f}'.format(molecule.fci_energy))

    print('Num operators: ', len(result['ansatz']))
    print('Ansatz:\n', result['ansatz'])
    print('Coefficients:\n', result['coefficients'])

    simulator.print_statistics()