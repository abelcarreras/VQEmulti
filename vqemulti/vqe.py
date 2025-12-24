from vqemulti.energy import get_vqe_energy
from vqemulti.gradient import get_vqe_energy_gradient
from vqemulti.pool.tools import OperatorList
from vqemulti.optimizers import OptimizerParams
from vqemulti.ansatz import GenericAnsatz
from vqemulti.utils import log_message
import numpy as np
import scipy


def vqe(hamiltonian,
        ansatz: GenericAnsatz,
        energy_simulator=None,
        energy_threshold=1e-4,
        optimizer_params=None
        ):
    """
    Perform a VQE calculation

    :param hamiltonian: hamiltonian in fermionic operators
    :param ansatz: Ansatz object tooptimize
    :param energy_simulator: Simulator object used to obtain the energy, if None do not use simulator (exact)
    :param energy_threshold: energy convergence threshold for classical optimization (in Hartree)
    :return: results dictionary
    """

    # set default optimizer params
    if optimizer_params is None:
        optimizer_params = OptimizerParams()

    log_message('optimizer params: ', optimizer_params, log_level=1)

    # initial guess
    coefficients = np.array(ansatz.parameters, dtype=float)

    # check if no coefficients
    if len(ansatz) == 0:
        energy = ansatz.get_energy(coefficients, hamiltonian, energy_simulator)
        return {'energy': energy, 'coefficients': [], 'ansatz': ansatz, 'f_evaluations': 0}

    # Optimize the results from analytical calculation
    results = scipy.optimize.minimize(ansatz.get_energy,
                                      coefficients,
                                      (hamiltonian, energy_simulator),
                                      jac=ansatz.get_gradients,
                                      method=optimizer_params.method,
                                      options=optimizer_params.options,
                                      tol=energy_threshold,
                                      )

    ansatz.parameters = results.x

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

    # Get UCCSD params
    uccsd_pool = get_pool_singlet_sd(n_electrons, n_orbitals, frozen_core=2)
    # uccsd_ansatz = []

    # Get reference Hartree Fock state
    hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits, frozen_core=2)
    print('hf reference', hf_reference_fock)

    # get ansatz
    from vqemulti.ansatz.exponential import ExponentialAnsatz

    initial_parameters = np.zeros_like(uccsd_pool)
    uccsd_ansatz = ExponentialAnsatz(initial_parameters, uccsd_pool, hf_reference_fock)


    # Simulator
    # from simulators.penny_simulator import PennylaneSimulator as Simulator
    # from simulators.cirq_simulator import CirqSimulator as Simulator
    from simulators.qiskit_simulator import QiskitSimulator as Simulator

    simulator = Simulator(trotter=False,
                          trotter_steps=1,
                          test_only=True,
                          shots=10000)

    print('Initialize VQE')
    result = vqe(hamiltonian,
                 uccsd_ansatz,
                 energy_simulator=simulator,
                 )

    print('Energy HF: {:.8f}'.format(molecule.hf_energy))
    print('Energy VQE: {:.8f}'.format(result['energy']))
    print('Energy CCSD: {:.8f}'.format(molecule.ccsd_energy))
    print('Energy FullCI: {:.8f}'.format(molecule.fci_energy))

    print('Num operators: ', len(result['ansatz']))
    print('Ansatz:\n', result['ansatz']._operators)
    print('Coefficients:\n', result['coefficients'])

    simulator.print_statistics()
