from vqemulti.utils import get_hf_reference_in_fock_space, generate_reduced_hamiltonian
from vqemulti.pool import get_pool_singlet_sd
from openfermionpyscf import run_pyscf
import numpy as np
from vqemulti.energy.simulation import simulate_adapt_vqe_energy_square, simulate_adapt_vqe_energy


def get_hamiltonian():
    # molecule definition
    from generate_mol import tetra_h4_mol, linear_h4_mol, square_h4_mol
    # h4_molecule = tetra_h4_mol(distance=5.0, basis='sto-3g')
    # h4_molecule = linear_h4_mol(distance=3.0, basis='3-21g')
    h4_molecule = square_h4_mol(distance=3.0, basis='sto-3g')

    # run classical calculation
    molecule = run_pyscf(h4_molecule, run_fci=True, nat_orb=False, guess_mix=False)

    print('FullCI energy result:', molecule.fci_energy)

    # get properties from classical SCF calculation
    n_electrons = molecule.n_electrons
    n_orbitals = 4  # molecule.n_orbitals
    hamiltonian = molecule.get_molecular_hamiltonian()
    hamiltonian = generate_reduced_hamiltonian(hamiltonian, n_orbitals)

    # hamiltonian analysis
    max_2body = np.max(np.abs(hamiltonian.two_body_tensor))
    max_1body = np.max(np.abs(hamiltonian.one_body_tensor))
    print('Max valued hamiltonian terms: ', max_1body, max_2body)

    print('N electrons', n_electrons)
    print('N Orbitals', n_orbitals)
    # Choose specific pool of operators for adapt-VQE
    pool = get_pool_singlet_sd(n_electrons=n_electrons, n_orbitals=n_orbitals)

    # Get Hartree Fock reference in Fock space
    hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

    return pool, hf_reference_fock, hamiltonian


def get_energy(pool, hf_reference_fock, hamiltonian):
    from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator

    simulator = Simulator(trotter=True, test_only=False, shots=100)

    coefficients = [0.0]
    ansatz = pool[0:1]

    energy = simulate_adapt_vqe_energy(coefficients, ansatz, hf_reference_fock, hamiltonian, simulator)
    return energy

pool, hf_reference_fock, hamiltonian = get_hamiltonian()

energy = get_energy(pool, hf_reference_fock, hamiltonian)
print('energy: ', energy)
