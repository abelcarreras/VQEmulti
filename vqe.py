from utils import get_hf_reference_in_fock_space
from energy import exact_vqe_energy, simulate_vqe_energy
from openfermion.transforms import jordan_wigner
from openfermion import QubitOperator, FermionOperator
from openfermion import reverse_jordan_wigner
import numpy as np
import scipy


def vqe(hamiltonian,
        ansatz,
        hf_reference_fock,
        exact_energy=False,
        trotter=False,
        trotter_steps=1,
        test_only=False,
        shots=1000):
    """
    Perform a VQE calculation

    :param hamiltonian: hamiltonian in fermionic operators
    :param ansatz: ansatz to optimize in fermionic operators)
    :param hf_reference_fock: HF reference in Fock space vector (occupations)
    :param exact_energy: Set True to compute energy analyticaly, set False to simulate
    :param trotter: Trotterize ansatz operators
    :param trotter_steps: number of trotter steps (only used if trotter=True)
    :param test_only: If true resolve QC circuit analytically instead of simulation (for testing circuit)
    :param shots: number of samples to perform in the simulation
    :return: results dictionary
    """

    # transform to qubit hamiltonian using JW transformation
    qubit_hamiltonian = jordan_wigner(hamiltonian)

    # transform to qubit ansatz using JW transformation
    qubit_ansatz = jordan_wigner(ansatz)

    # initial guess
    n_terms = len(qubit_ansatz.terms)
    coefficients = np.zeros(n_terms)

    # Optimize the results from analytical calculation
    if exact_energy:
        results = scipy.optimize.minimize(exact_vqe_energy,
                                          coefficients,
                                          (qubit_ansatz, hf_reference_fock, qubit_hamiltonian),
                                          method="COBYLA",
                                          options={'rhobeg': 0.1, 'disp': True},
                                          )

    # Optimize the results from the CIRQ simulation
    else:
        results = scipy.optimize.minimize(simulate_vqe_energy,
                                          coefficients,
                                          (qubit_ansatz, hf_reference_fock, qubit_hamiltonian,
                                           shots, trotter, trotter_steps, test_only),
                                          method="COBYLA",
                                          options={# 'rhobeg': 0.01,
                                                   'disp': True},
                                          )

    # generate operators list adn global coefficients
    operators_list = []
    global_coefficients = []
    for coefficient, op in zip(results.x, qubit_ansatz.terms):
        operators_list.append(1j*QubitOperator(op))
        global_coefficients.append(coefficient * np.real(qubit_ansatz.terms[op]/1j))

    # generated updated ansatz (in Qubits)
    optimized_ansatz_qubit = QubitOperator()
    for coefficient, op in zip(global_coefficients, operators_list):
        optimized_ansatz_qubit += coefficient * op

    return {'energy': results.fun,
            'coefficients': global_coefficients,
            'operators': operators_list}


if __name__ == '__main__':

    from openfermion import MolecularData
    from openfermionpyscf import run_pyscf
    from utils import generate_reduced_hamiltonian, get_uccsd_operators

    h2_molecule = MolecularData(geometry=[['H', [0, 0, 0]],
                                          ['H', [0, 0, 0.74]]],
                                basis='3-21g',
                                # basis='sto-3g',
                                multiplicity=1,
                                charge=0,
                                description='H2')

    # run classical calculation
    molecule = run_pyscf(h2_molecule, run_fci=True, run_ccsd=True)

    # get properties from classical SCF calculation
    n_electrons = molecule.n_electrons
    n_orbitals = 2 # molecule.n_orbitals

    hamiltonian = molecule.get_molecular_hamiltonian()
    hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals)

    print('n_electrons: ', n_electrons)
    print('n_orbitals: ', n_orbitals)
    print('n_qubits:', hamiltonian.n_qubits)

    # define UCCSD ansatz
    uccsd_ansatz = get_uccsd_operators(n_electrons, n_orbitals)

    # Get reference Hartree Fock state
    hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

    print('Initialize VQE')
    result = vqe(hamiltonian,
                 uccsd_ansatz,
                 hf_reference_fock,
                 exact_energy=True,
                 trotter=False,
                 trotter_steps=2,
                 shots=1000,
                 test_only=False)

    print('Energy HF: {:.8f}'.format(molecule.hf_energy))
    print('Energy VQE: {:.8f}'.format(result['energy']))
    print('Energy CCSD: {:.8f}'.format(molecule.ccsd_energy))
    print('Energy FullCI: {:.8f}'.format(molecule.fci_energy))

    print('Num operators: ', len(result['operators']))
    print('Operators:\n', result['operators'])
    print('Coefficients:\n', result['coefficients'])
