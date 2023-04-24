from utils import get_hf_reference_in_fock_space
from energy import exact_vqe_energy, simulate_vqe_energy
from openfermion.transforms import jordan_wigner
import numpy as np
import openfermion
import scipy


def vqe(hamiltonian,
        ansatz,
        hf_reference_fock,
        exact_energy=False,
        trotter=False,
        trotter_steps=1,
        sample=False,
        shots=1000):
    '''
    Runs the VQE algorithm to find the ground state of a molecule

    Arguments:
      amplitudes (list, np.array): the amplitudes that specify the starting
        parameters of the UCCSD circuit
      molecule (openfermion.MolecularData): the molecule in consideration
      shots (int): the number of circuit repetitions to be used in the
        expectation estimation
      optolerance (float,float): values of fatol and xatol that define the
        accepted tolerance for convergence in the optimization
      simulate (bool): if False, the circuit will not be trotterized
      sample (bool): if False, the full state vector will be simulated and
        the result will be free of sampling noise
    '''

    # transform to qubit hamiltonian using JW transformation
    qubit_hamiltonian = jordan_wigner(hamiltonian)

    # transform to qubit ansatz using JW transformation
    qubit_ansatz = jordan_wigner(ansatz)

    # initial guess
    n_terms = len(qubit_ansatz.terms)
    coefficients = np.ones(n_terms)

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
                                           shots, trotter, trotter_steps, sample),
                                          method="COBYLA",
                                          options={# 'rhobeg': 0.01,
                                                   'disp': True},
                                          )

    return {'energy': results.fun,
            'coefficients': results.x}


if __name__ == '__main__':

    from openfermion import MolecularData
    from openfermionpyscf import run_pyscf
    from utils import generate_reduced_hamiltonian

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

    # Prepare UCCSD ansatz
    packed_amplitudes = openfermion.uccsd_singlet_get_packed_amplitudes(molecule.ccsd_single_amps,
                                                                        molecule.ccsd_double_amps,
                                                                        n_orbitals * 2,
                                                                        n_electrons)

    uccsd_ansatz = openfermion.uccsd_singlet_generator(packed_amplitudes,
                                                       n_orbitals * 2,
                                                       n_electrons)

    # Get reference Hartree Fock state
    hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

    print('Initialize VQE')
    result = vqe(hamiltonian,  # fermionic hamiltonian
                 uccsd_ansatz,  # fermionic ansatz
                 hf_reference_fock,
                 exact_energy=False,
                 shots=1000,
                 sample=False)

    print('Energy VQE: {:.8f}'.format(result['energy']))
    print('Energy FullCI: {:.8f}'.format(molecule.fci_energy))
    print('Energy CCSD: {:.8f}'.format(molecule.ccsd_energy))

    print('Num operators: ', len(result['coefficients']))
    print('Coefficients:\n', result['coefficients'])
