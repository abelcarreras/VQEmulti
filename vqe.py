from energy import exact_vqe_energy, simulate_vqe_energy
from utils import fermion_to_qubit
from pool.tools import OperatorList
import numpy as np
import scipy


def vqe(hamiltonian,
        ansatz,
        hf_reference_fock,
        coefficients=None,
        opt_qubits=False,
        energy_simulator=None):
    """
    Perform a VQE calculation

    :param hamiltonian: hamiltonian in fermionic operators
    :param ansatz: ansatz (fermionic operator list) to optimize
    :param coefficients: initial guess coefficients (leave None to initialize to zero)
    :param opt_qubits: choose basis of optimization (True: qubits operators, False: fermion operators)
    :param hf_reference_fock: HF reference in Fock space vector (occupations)
    :param energy_simulator: Simulator used to obtain the energy, if None do not use simulator (exact).
    :return: results dictionary
    """

    # transform to qubit hamiltonian
    qubit_hamiltonian = fermion_to_qubit(hamiltonian)

    ansatz = OperatorList(ansatz, antisymmetrize=True, normalize=True)

    if opt_qubits:
        # transform to qubit ansatz
        ansatz = ansatz.get_quibits_list(normalize=True)

    # initial guess
    n_terms = len(ansatz)
    if coefficients is None:
        coefficients = np.zeros(n_terms)

    # Optimize the results from analytical calculation
    if energy_simulator is None:
        results = scipy.optimize.minimize(exact_vqe_energy,
                                          coefficients,
                                          (ansatz, hf_reference_fock, qubit_hamiltonian),
                                          method="COBYLA",
                                          options={'rhobeg': 0.1, 'disp': True},
                                          )

    # Optimize the results from the CIRQ simulation
    else:
        results = scipy.optimize.minimize(simulate_vqe_energy,
                                          coefficients,
                                          (ansatz, hf_reference_fock, qubit_hamiltonian, energy_simulator),
                                          method="COBYLA",
                                          options={'rhobeg': 0.1, 'disp': True},
                                          )

    # testing consistency
    energy_exact = exact_vqe_energy(results.x, ansatz, hf_reference_fock, qubit_hamiltonian)

    if energy_simulator is not None:
        energy_sim_test = simulate_vqe_energy(results.x, ansatz, hf_reference_fock, qubit_hamiltonian,
                                              type(energy_simulator)(trotter=False, test_only=True))

        #assert abs(energy_exact - energy_sim_test) < 1e-6

    if energy_simulator is not None:
        circuit_info = energy_simulator.get_circuit_info(coefficients, ansatz, hf_reference_fock)
        print('circuit depth: ', circuit_info['depth'])

    return {'energy': results.fun,
            'coefficients': list(results.x),
            'ansatz': list(ansatz)}


if __name__ == '__main__':

    from openfermion import MolecularData
    from openfermionpyscf import run_pyscf
    from utils import generate_reduced_hamiltonian, get_hf_reference_in_fock_space
    from pool.singlet_sd import get_pool_singlet_sd

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
    #from simulators.penny_simulator import PennylaneSimulator as Simulator
    from simulators.cirq_simulator import CirqSimulator as Simulator
    # from simulators.quiskit_simulator import QiskitSimulator as Simulator

    simulator = Simulator(trotter=True,
                          trotter_steps=1,
                          test_only=False,
                          shots=100000)

    print('Initialize VQE')
    result = vqe(hamiltonian,
                 uccsd_ansatz,
                 hf_reference_fock,
                 energy_simulator=simulator,
                 opt_qubits=False)

    print('Energy HF: {:.8f}'.format(molecule.hf_energy))
    print('Energy VQE: {:.8f}'.format(result['energy']))
    print('Energy CCSD: {:.8f}'.format(molecule.ccsd_energy))
    print('Energy FullCI: {:.8f}'.format(molecule.fci_energy))

    print('Num operators: ', len(result['ansatz']))
    print('Ansatz:\n', result['ansatz'])
    print('Coefficients:\n', result['coefficients'])
