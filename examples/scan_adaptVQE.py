# example of hydrogen molecule dissociation using adaptVQE method
# and Pennylane simulator (1000 shots)
from vqemulti.utils import get_hf_reference_in_fock_space, generate_reduced_hamiltonian
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.adapt_vqe import adaptVQE
from vqemulti.analysis import get_info
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
import matplotlib.pyplot as plt
import numpy as np
import math

vqe_energies = []
energies_fullci = []
energies_hf = []
for d in np.linspace(0.3, 3, 20):
    def generate_tetrahedron_coordinates(axis_length):
        # Calculate the height of the tetrahedron
        height = math.sqrt(2 / 3) * axis_length

        # Calculate the coordinates of the vertices
        vertices = []
        vertices.append((0, 0, 0))
        vertices.append((axis_length, 0, 0))
        vertices.append((axis_length / 2, axis_length * math.sqrt(3) / 2, 0))
        vertices.append((axis_length / 2, axis_length * math.sqrt(3) / 6, height))

        return vertices


    def tetra_h4_mol(distance, basis='sto-3g'):
        coor = generate_tetrahedron_coordinates(distance)

        mol = MolecularData(geometry=[['H', coor[0]],
                                      ['H', coor[1]],
                                      ['H', coor[2]],
                                      ['H', coor[3]]],
                            basis=basis,
                            multiplicity=1,
                            charge=0,
                            description='H4')
        return mol


    def square_h4_mol(distance, basis='sto-3g'):
        mol = MolecularData(geometry=[['H', [0, 0, 0]],
                                      ['H', [distance, 0, 0]],
                                      ['H', [0, distance, 0]],
                                      ['H', [distance, distance, 0]]],
                            basis=basis,
                            multiplicity=1,
                            charge=0,
                            description='H4')
        return mol


    def linear_h4_mol(distance, basis='sto-3g'):
        mol = MolecularData(geometry=[['H', [0, 0, 0]],
                                      ['H', [0, 0, distance]],
                                      ['H', [0, 0, 2 * distance]],
                                      ['H', [0, 0, 3 * distance]]],
                            basis=basis,
                            multiplicity=1,
                            charge=0,
                            description='H4')
        return mol


    # run classical calculation
    molecule = run_pyscf(linear_h4_mol(d, '3-21g'), nat_orb= True, run_fci=True)

    # get additional info about electronic structure properties
    # get_info(molecule, check_HF_data=False)

    # get properties from classical SCF calculation
    n_electrons = 4  # molecule.n_electrons
    n_orbitals = 4  # molecule.n_orbitals
    hamiltonian = molecule.get_molecular_hamiltonian()

    # Choose specific pool of operators for adapt-VQE
    pool = get_pool_singlet_sd(n_electrons=n_electrons,
                               n_orbitals=n_orbitals)

    # Get Hartree Fock reference in Fock space
    hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

    from vqemulti.simulators.penny_simulator import PennylaneSimulator as Simulator

    # define simulator paramters
    simulator = Simulator(trotter=True,
                          trotter_steps=1,
                          shots=1000)

    # run adaptVQE
    result = adaptVQE(hamiltonian,
                      pool,
                      hf_reference_fock,
                      opt_qubits=False,
                      max_iterations=10,
                      coeff_tolerance=1e-3,
                      energy_threshold=1e-4,
                      threshold=1e-9
                      # energy_simulator=simulator,  # comment this line to not use sampler simulator
                      # gradient_simulator=simulator,  # comment this line to not use sampler simulator
                      )

    print("Final energy:", result["energy"])

    # Compare error vs FullCI calculation
    error = result["energy"] - molecule.fci_energy
    print("Error:", error)

    # run results
    print("Ansatz:", result["ansatz"])
    print("Indices:", result["indices"])
    print("Coefficients:", result["coefficients"])
    print("Num operators: {}".format(len(result["ansatz"])))

    energies_hf.append(molecule.hf_energy)
    vqe_energies.append(result["energy"])
    energies_fullci.append(molecule.fci_energy)

plt.title('Absolute energies')
plt.xlabel('Interatomic distance [Angs]')
plt.ylabel('Energy [H]')

plt.plot(np.linspace(0.3, 3, 20), energies_hf, label='HF')
plt.plot(np.linspace(0.3, 3, 20), vqe_energies, label='adaptVQE')
plt.plot(np.linspace(0.3, 3, 20), energies_fullci, label='FullCI')
plt.legend()

plt.figure()
plt.title('Difference between fullCI')
plt.xlabel('Interatomic distance [Angs]')
plt.ylabel('Energy [H]')
plt.yscale('log')

diff_fullci = np.subtract(vqe_energies, energies_fullci)
plt.plot(np.linspace(0.3, 3, 20), diff_fullci, label='adaptVQE')
plt.legend()

plt.show()