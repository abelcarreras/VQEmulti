# example of H2 molecule using adaptVQE method and Pennylane simulator (10000 shots)
from vqemulti.utils import get_hf_reference_in_fock_space, generate_reduced_hamiltonian
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.adapt_vqe import adaptVQE
from vqemulti.analysis import get_info
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator


# molecule definition
h2_molecule = MolecularData(geometry=[['H', [0, 0, 0]],
                                      ['H', [0, 0, 0.8]]],
                            basis='sto-3g',
                            multiplicity=1,
                            charge=0,
                            description='H2')

# run classical calculation
molecule = run_pyscf(h2_molecule, run_fci=True)

# get additional info about electronic structure properties
# get_info(molecule, check_HF_data=False)

# get properties from classical SCF calculation
n_electrons = molecule.n_electrons
n_orbitals = molecule.n_orbitals_active
hamiltonian = molecule.get_molecular_hamiltonian()

# Choose specific pool of operators for adapt-VQE
operators_pool = get_pool_singlet_sd(n_electrons=n_electrons, n_orbitals=n_orbitals)

# Get Hartree Fock reference in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, hamiltonian.n_qubits)

# define simulator paramters
simulator = Simulator(trotter=True,
                      trotter_steps=1,
                      shots=10000)
from vqemulti.method.adapt_vanila import AdapVanilla

method = AdapVanilla(gradient_threshold=1e-6,
                     diff_threshold=0,
                     coeff_tolerance=1e-10,
                     gradient_simulator=simulator,
                     operator_update_number=1,
                     operator_update_max_grad=2e-2,
                     )

# run adaptVQE
result = adaptVQE(hamiltonian,  # fermionic hamiltonian
                  operators_pool,  # fermionic operators
                  hf_reference_fock,
                  energy_threshold=0.0001,
                  method=method,
                  max_iterations=20,
                  energy_simulator=simulator,
                  variance_simulator=simulator,
                  reference_dm=None,
                  optimizer_params=None)

print("HF energy:", molecule.hf_energy)
print("VQE energy:", result["energy"])
print("FullCI energy:", molecule.fci_energy)

# Compare error vs FullCI calculation
error = result["energy"] - molecule.fci_energy
print("Error (respect to FullCI):", error)

# run results
print("Ansatz:", result["ansatz"])
print("Indices:", result["indices"])
print("Coefficients:", result["coefficients"])
print("Num operators: {}".format(len(result["ansatz"])))


