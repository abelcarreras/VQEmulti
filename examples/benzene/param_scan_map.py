from vqemulti.utils import get_hf_reference_in_fock_space, generate_reduced_hamiltonian
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.adapt_vqe import adaptVQE
from vqemulti.energy import get_adapt_vqe_energy
from vqemulti.energy.simulation import simulate_adapt_vqe_variance
from vqemulti.preferences import Configuration
from vqemulti.symmetry import get_symmetry_reduced_pool, symmetrize_molecular_orbitals
from openfermionpyscf import run_pyscf
from vqemulti.errors import NotConvergedError
from vqemulti.basis_projection import get_basis_overlap_matrix, project_basis, prepare_ansatz_for_restart
from vqemulti.density import get_density_matrix, density_fidelity
from vqemulti.optimizers import OptimizerParams
from vqemulti.optimizers import adam, rmsprop, sgd, cobyla_mod
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import Session
from openfermion import MolecularData, get_sparse_operator, get_fermion_operator
import numpy as np

from noise_model import get_noise_model

import warnings
warnings.filterwarnings("ignore")

config = Configuration()
config.verbose = 2
config.mapping = 'jw'


from vqemulti.utils import load_wave_function, load_hamiltonian
coeff, ansatz = load_wave_function(filename='wf_qubit.yml', qubit_op=True)
# coeff, ansatz = load_wave_function(filename='wf_fermion.yml', qubit_op=False)

print(coeff)
coeff = coeff[:4]
ansatz = ansatz[:4]

# normalize coefficients
from vqemulti.pool.tools import OperatorList
for i, c in enumerate(coeff):
    # print(np.linalg.norm(list(ansatz[i].terms.values())[0]))
    coeff[i] *= np.linalg.norm(list(ansatz[i].terms.values())[0])



ansatz = OperatorList(ansatz.get_quibits_list(normalize=True), normalize=True)


hamiltonian = load_hamiltonian(file='hamiltonian.npz')
hamiltonian = get_fermion_operator(hamiltonian)
hamiltonian.compress(1e-2)

n_electrons = 4
n_orbitals = 4

print('N electrons', n_electrons)
print('N Orbitals', n_orbitals)
# Choose specific pool of operators for adapt-VQE

# Get Hartree Fock reference in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, n_orbitals*2)
# hf_reference_fock = [0] * n_orbitals*2
print('hf_reference_fock: ', hf_reference_fock)

from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
# from vqemulti.simulators.penny_simulator import PennylaneSimulator as Simulator


x_range = np.linspace(-np.pi, np.pi, 20)

surface = []
for m1 in x_range:
    energies = []

    for m2 in x_range:

        print(ansatz)
        print(coeff)
        coeff[2] = m1
        coeff[3] = m2

        # run SP energy
        energy = get_adapt_vqe_energy(coeff,
                                      ansatz,
                                      hf_reference_fock,
                                      hamiltonian,
                                      None)

        energies.append(energy)
        print(energy)

    surface.append(energies)


import matplotlib.pyplot as plt

X, Y = np.meshgrid(x_range, x_range)

# Calculem la funció sin(x) * sin(y)
Z = np.array(surface)

# Creem el mapa de contorn
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, Z, levels=50, cmap="viridis")
contour = plt.contour(X, Y, Z, levels=8, colors="black")  # Només línies negres

plt.colorbar(contour, label="2 operators")
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$\theta_2$")
plt.title("Contour map")
plt.show()

