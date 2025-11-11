# example of H2 molecule using adaptVQE method and Pennylane simulator (10000 shots)
from vqemulti.utils import get_hf_reference_in_fock_space, generate_reduced_hamiltonian
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.adapt_vqe import adaptVQE
from vqemulti.analysis import get_info
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
from pyqchem.tools import get_geometry_from_pubchem

pyqchem_mol = get_geometry_from_pubchem('ammonia')
print(pyqchem_mol)

from posym import PointGroup, SymmetryMolecule, SymmetryGaussianLinear


# molecule definition
h2_molecule = MolecularData(geometry=[[s, c] for s, c in zip(pyqchem_mol.get_symbols(), pyqchem_mol.get_coordinates())],
                            basis='sto-3g',
                            multiplicity=1,
                            charge=0,
                            description='NH3')


#print(sm.orientation_angles)
#exit()

# molecule definition
h2_molecule = MolecularData(geometry=[['O', [ 0.0000000000, 0.000000000, -0.0428008531]],
                                      ['H', [-0.7581074140, 0.000000000, -0.6785995734]],
                                      ['H', [ 0.7581074140, 0.000000000, -0.6785995734]]],
                            basis='sto-3g',
                            multiplicity=1,
                            charge=0,
                            description='H2O')

# run classical calculation
molecule = run_pyscf(h2_molecule, run_fci=True)

mol_pyscf = molecule._pyscf_data['mol']
#print(mol_pyscf.nbas)
#exit()

#print(dir(mol_pyscf))
#print(mol_pyscf.atom_coords())
#print(mol_pyscf.atom_symbol(0))


sm = SymmetryMolecule('c2v', mol_pyscf.atom_coords(), ['O', 'H', 'H'])
print(sm.measure_pos)
print(sm.orientation_angles)
print('center: ', sm.center)


from vqemulti.preferences import Configuration
from posym.tools import get_basis_set_pyscf, build_density
import numpy as np

basis_set = get_basis_set_pyscf(mol_pyscf)

#print(molecule.n_orbitals)
density_matrix = np.diag([2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0])
#print(density_matrix)

#print('density mat: ', np.diag(density_matrix))
# f_density = build_density(basis_set, density_matrix)
#print('density integral: ', f_density.integrate)

# sm = SymmetryGaussianLinear('c2v', f_density, orientation_angles=[-90, 0, 0])
#sm = SymmetryGaussianLinear('c3v', f_density,
#                            orientation_angles=[168.86670622, 40.72374731, -7.05789814],
#                            center=[0.36443368, 0.05201471, 0.42013336])
#print(sm.center)
#print(sm.measure_pos, sm.measure)
# exit()
Configuration().kill_pyscf = basis_set
Configuration().verbose = True

# get additional info about electronic structure properties
# get_info(molecule, check_HF_data=False)

# get properties from classical SCF calculation
n_electrons = molecule.n_electrons
n_orbitals = 6 # molecule.n_orbitals
hamiltonian = molecule.get_molecular_hamiltonian()
print('n_orbitals: ', n_orbitals)
print('n_electrons: ', n_electrons)

# Choose specific pool of operators for adapt-VQE
pool = get_pool_singlet_sd(n_electrons=n_electrons, n_orbitals=n_orbitals)

# Get Hartree Fock reference in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

# define simulator paramters
simulator = Simulator(trotter=True,
                      trotter_steps=1,
                      shots=10000)

# run adaptVQE
result = adaptVQE(hamiltonian,
                  pool,
                  hf_reference_fock,
                  opt_qubits=True,
                  max_iterations=50,
                  energy_threshold=1e-2,  # energy tolerance for classical optimization function
                  # coeff_tolerance=1e-2,  # threshold value for which coefficient is assumed to be zero
                  # energy_simulator=simulator,  # comment this line to not use sampler simulator
                  # gradient_simulator=simulator,  # comment this line to not use sampler simulator
                  )

print("HF energy:", molecule.hf_energy)
print("VQE energy:", result["energy"])
print("FullCI energy:", molecule.fci_energy)

# Compare error vs FullCI calculation
error = result["energy"] - molecule.fci_energy
print("Error (respect to FullCI):", error)

# run results
# print("Ansatz:", result["ansatz"])
print("Indices:", result["indices"])
print("Coefficients:", result["coefficients"])
print("Num operators: {}".format(len(result["ansatz"])))


