from utils import get_hf_reference_in_fock_space
from energy import exact_vqe_energy, simulate_vqe_energy
from openfermion.transforms import jordan_wigner
import openfermion
import scipy


def vqe(hamiltonian,
        uccsd_operator,
        hf_reference_fock,
        exact_energy=True,
        trotter=True,
        trotter_steps=1,
        sample=True,
        shots=1000):
    '''
    Runs the VQE algorithm to find the ground state of a molecule

    Arguments:
      packed_amplitudes (list): the amplitudes that specify the starting
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

    # get JW hamiltonian
    qubit_hamiltonian = jordan_wigner(hamiltonian)

    # get JW UCCSD operator
    qubit_operator = jordan_wigner(uccsd_operator)


    # Optimize the results from analytical calculation
    if exact_energy:
        results = scipy.optimize.minimize(exact_vqe_energy,
                                          packed_amplitudes,
                                          (qubit_operator, hf_reference_fock, qubit_hamiltonian),
                                          method="COBYLA",
                                          options={'rhobeg': 0.1, 'disp': True},
                                          )

    # Optimize the results from the CIRQ simulation
    else:
        results = scipy.optimize.minimize(simulate_vqe_energy,
                                          packed_amplitudes,
                                          (qubit_operator, hf_reference_fock, qubit_hamiltonian,
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

    h2_molecule = MolecularData(geometry=[['H', [0, 0, 0]],
                                          ['H', [0, 0, 0.74]]],
                                # basis='3-21g',
                                basis='sto-3g',
                                multiplicity=1,
                                charge=0,
                                description='H2')

    # run classical calculation
    molecule = run_pyscf(h2_molecule, run_fci=True, run_ccsd=True)

    # get properties from classical SCF calculation
    hamiltonian = molecule.get_molecular_hamiltonian()
    n_electrons = molecule.n_electrons
    n_orbitals = molecule.n_orbitals

    print('n_electrons: ', n_electrons)
    print('n_orbitals: ', n_orbitals)
    print('n_qubits:', hamiltonian.n_qubits)

    # Prepare UCCSD operator
    packed_amplitudes = openfermion.uccsd_singlet_get_packed_amplitudes(molecule.ccsd_single_amps,
                                                                        molecule.ccsd_double_amps,
                                                                        n_orbitals*2,
                                                                        n_electrons)

    uccsd_operator = openfermion.uccsd_singlet_generator(packed_amplitudes,
                                                         n_orbitals*2,
                                                         n_electrons)

    print('UCCSD fermion operators:\n', uccsd_operator)

    # Get reference Hartree Fock state
    hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

    print('Initialize VQE')
    results = vqe(hamiltonian,  # fermionic hamiltonian
                  uccsd_operator,
                  hf_reference_fock,
                  exact_energy=False,
                  shots=10000,
                  sample=True)

    print('Energy VQE: {:.8f}'.format(results['energy']))
    print('Energy FullCI: {:.8f}'.format(molecule.fci_energy))
    print('Energy CCSD: {:.8f}'.format(molecule.ccsd_energy))

    print('Coefficients: ', results['coefficients'])