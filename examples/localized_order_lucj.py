from openfermion import MolecularData
from openfermionpyscf import run_pyscf, store_orbitals_in_molden
from vqemulti.utils import get_hf_reference_in_fock_space, permute_hamiltonian
from vqemulti.ansatz.unitary_jastrow import UnitaryCoupledJastrowAnsatz
from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
from vqemulti.ansatz.generators.rotation import change_of_basis_orbitals
from vqemulti.ansatz.generators.basis import get_absolute_orbitals
from vqemulti.preferences import Configuration
from openfermion import get_fermion_operator
from copy import deepcopy
import numpy as np


Configuration().mapping = 'jw'
Configuration().verbose = False
Configuration().temp_dir = '/Users/abel/TEST/DICE'


def print_permutation(G_qpu, permutation):
    print('mapping original -> final')

    for i, p in enumerate(permutation):
        print(i, '->', p)

    pos = {permutation[i]: centers[i][:2] for i in range(n_orbitals)}

    permutation_inv = np.argsort(permutation).tolist()

    edge_labels = {}
    for i, j in G_qpu.edges:
        i2 = permutation_inv[i]
        j2 = permutation_inv[j]

        d = np.linalg.norm(centers[i2][:2] - centers[j2][:2])
        edge_labels[(i, j)] = f"{d:.2f}"


    labels = {i: permutation_inv[i] for i in G_qpu.nodes}

    nx.draw(G_qpu, pos, with_labels=False)
    nx.draw_networkx_edge_labels(G_qpu, pos, edge_labels=edge_labels)
    nx.draw_networkx_labels(G_qpu, pos,labels=labels)
    plt.show()

def optimize_mapping(G_qpu, interaction_matrix):
    from scipy.optimize import quadratic_assignment

    res = quadratic_assignment(interaction_matrix,
                               nx.to_numpy_array(G_qpu),
                               method="2opt",
                               options={"maximize": True})

    return res.col_ind, res.fun

butatriene = MolecularData(
    geometry=[
        ('C', [-0.5968,  0.3062,  0.0001 ]),
        ('C', [ 0.5968, -0.3063,  0.0000 ]),
        ('C', [-1.8503, -0.4127,  0.0000 ]),
        ('C', [ 1.8502,  0.4127, -0.0001 ]),
        ('C', [-3.0441,  0.1908,  0.0000 ]),
        ('C', [ 3.0442, -0.1908,  0.0000 ]),
        ('H', [-0.6504,  1.3927,  0.0000 ]),
        ('H', [ 0.6505, -1.3928,  0.0000 ]),
        ('H', [-1.8139, -1.4988, -0.0001 ]),
        ('H', [ 1.8138,  1.4988, -0.0002 ]),
        ('H', [-3.9547, -0.3990, -0.0002 ]),
        ('H', [-3.1431,  1.2712,  0.0001 ]),
        ('H', [ 3.9547,  0.3991, -0.0001 ]),
        ('H', [ 3.1431, -1.2712,  0.0002 ]),
    ],
    basis='3-21g',
    multiplicity=1,
    charge=0,
    description='tetracene'
)

# 44 total electrons
active = 6

# run reference calculation
molecule_loc = run_pyscf(deepcopy(butatriene),
                         run_fci=False,
                         nat_orb=False,
                         guess_mix=False,
                         verbose=True,
                         frozen_core=44//2-active//2,
                         n_orbitals=44//2+active//2,
                         #run_ccsd=True,
                         run_casci=True,
                         loc_boys=False,
                         loc_pipek=True,
                         permute_orbitals=[(24, 25)],
                         reference='HF')


# hamiltonian data
n_electrons = molecule_loc.n_electrons
n_orbitals = molecule_loc.n_orbitals
multiplicity = molecule_loc.multiplicity
centers = molecule_loc.orbital_centers

print('n_electrons: ', n_electrons)
print('n_orbitals: ', n_orbitals)
print('n_qubits: ', molecule_loc.n_qubits)
print('multiplicity', multiplicity)

# save orbitals
store_orbitals_in_molden(molecule_loc, filename='orbitals.molden')


# Build local hamiltonian
trans_mat = molecule_loc.canonical_local_trans_mat
hamiltonian_loc = molecule_loc.get_molecular_hamiltonian()


import networkx as nx
import matplotlib.pyplot as plt


# define topology
G_qpu = nx.Graph()
G_qpu.add_edges_from([(i, i+1) for i in range(n_orbitals-1)])


# based on 1e interaction
one_body_alpha_simple = hamiltonian_loc.one_body_tensor[::2, ::2]
permutation, score = optimize_mapping(G_qpu, np.abs(one_body_alpha_simple))

# based on distance
# distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
# permutation, score = optimize_mapping(G_qpu, -np.square(distances))

print('permutation:', permutation)

print_permutation(G_qpu, permutation)
permutation_inv = np.argsort(permutation).tolist()

hamiltonian_loc_ord = permute_hamiltonian(hamiltonian_loc, permutation_inv)

molecule = run_pyscf(deepcopy(butatriene),
                     run_fci=False,
                     nat_orb=False,
                     guess_mix=False,
                     verbose=True,
                     frozen_core=44//2-active//2,
                     n_orbitals=44//2+active//2,
                     run_ccsd=True,
                     run_casci=True,
                     loc_boys=False,
                     loc_pipek=False,
                     permute_orbitals=[(24, 25)],
                     reference='HF')


hamiltonian = molecule.get_molecular_hamiltonian()


do_compression_test = True
if do_compression_test:
    n_terms_can = []
    n_terms_loc = []
    range_c = np.logspace(-4, -1, 10)
    for compression in range_c:
        # compression = 1e-2
        hamiltonian_ = get_fermion_operator(hamiltonian)
        print('n_terms hamiltonian original: ', len(hamiltonian_.terms))

        hamiltonian_.compress(compression)
        print('n_terms hamiltonian: ', len(hamiltonian_.terms))
        n_terms_can.append(len(hamiltonian_.terms))

        hamiltonian_loc_ = get_fermion_operator(hamiltonian_loc)
        hamiltonian_loc_.compress(compression)
        print('n_terms hamiltonian_loc:', len(hamiltonian_loc_.terms))
        n_terms_loc.append(len(hamiltonian_loc_.terms))

        hamiltonian_loc_ord_ = get_fermion_operator(hamiltonian_loc_ord)
        hamiltonian_loc_ord_.compress(compression)
        #print('n_terms hamiltonian_loc_ord:', len(hamiltonian_loc_ord_.terms))

    plt.plot(range_c, n_terms_can, label='canonical')
    plt.plot(range_c, n_terms_loc, label='local')
    plt.xlabel('compression level')
    plt.ylabel('number of Hamiltonian terms')
    plt.xscale('log')
    plt.legend()
    plt.show()


do_dmrg_test = True
if do_dmrg_test:
    from vqemulti.utils import get_dmrg_energy

    ref_energy = molecule.casci_energy
    energy_list = []
    energy_list_loc = []
    energy_list_loc_ord = []
    bd_range = range(1, 100, 5)
    for bd in bd_range:

        energy_dmrg, extra = get_dmrg_energy(hamiltonian,
                                             molecule_loc.n_electrons,
                                             max_bond_dimension=bd,
                                             start_bond_dimension=10,
                                             reorder_sites=False,
                                             # sample=1e-6,
                                             stream_output=False,
                                             compute_density_matrix=True,
                                             # schedule=schedule,
                                             #max_solver_iterations=200
                                             )


        energy_dmrg_loc, extra = get_dmrg_energy(hamiltonian_loc,
                                                 molecule_loc.n_electrons,
                                                 max_bond_dimension=bd,
                                                 start_bond_dimension=10,
                                                 reorder_sites=False,
                                                 # sample=1e-6,
                                                 stream_output=False,
                                                 compute_density_matrix=True,
                                                 # schedule=schedule,
                                                 #max_solver_iterations=200
                                                 )

        energy_dmrg_loc_ord, extra = get_dmrg_energy(hamiltonian_loc_ord,
                                                     molecule_loc.n_electrons,
                                                     max_bond_dimension=bd,
                                                     start_bond_dimension=10,
                                                     reorder_sites=False,
                                                     # sample=1e-6,
                                                     stream_output=False,
                                                     compute_density_matrix=True,
                                                     # schedule=schedule,
                                                     #max_solver_iterations=200
                                                     )

        print('energy_dmrg', bd, energy_dmrg, energy_dmrg_loc, energy_dmrg_loc_ord)
        energy_list.append(energy_dmrg - ref_energy)
        energy_list_loc.append(energy_dmrg_loc - ref_energy)
        energy_list_loc_ord.append(energy_dmrg_loc_ord -ref_energy)

    plt.plot(bd_range, energy_list, label='canonical')
    plt.plot(bd_range, energy_list_loc, label='local')
    plt.plot(bd_range, energy_list_loc_ord, label='local_sorted')
    plt.yscale('log')
    plt.xlabel('bond dimension')
    plt.ylabel('energy error [Ha]')
    plt.legend()
    plt.show()


do_hci_test = True
if do_hci_test:
    from vqemulti.utils import get_selected_ci_energy_dice, load_hamiltonian

    configuration_HF = [
        [1] * n_electrons + [0] * (2 * n_orbitals - n_electrons)  # HF
    ]

    hci_energy_ref, extra_ref = get_selected_ci_energy_dice(configuration_HF,
                                                            hamiltonian_loc_ord,
                                                            # stream_output=True,
                                                            compute_variance=True,
                                                            #compute_ci_state=True,
                                                            #hci_schedule=[(0, 1e-3), (100, 1e-6), (200, 1e-8)]
                                                            )
    print('hci_energy_ref: ', hci_energy_ref)


molecule = run_pyscf(deepcopy(butatriene),
                     run_fci=False,
                     nat_orb=False,
                     guess_mix=False,
                     verbose=True,
                     frozen_core=44//2-active//2,
                     n_orbitals=44//2+active//2,
                     run_ccsd=True,
                     run_casci=True,
                     loc_boys=False,
                     loc_pipek=False,
                     permute_orbitals=[(24, 25)],
                     reference='HF')


hamiltonian = molecule.get_molecular_hamiltonian()

local = 2
n_terms = 1

# scale t1 to be noticeable (unphysical)
ccsd = molecule._pyscf_data.get('ccsd', None)
#ccsd.t1 = ccsd.t1 * 1000


from qiskit_ibm_runtime import QiskitRuntimeService
# list of backends

#service = QiskitRuntimeService()
#backend = service.backend('ibm_basquecountry')
#print('backend: ', backend)

simulator = Simulator(trotter=True,
                      trotter_steps=1,
                      test_only=False,
                      # hamiltonian_grouping=True,
                      # backend=backend,
                      use_ibm_runtime=True,
                      # use_estimator=True,
                      shots=100000)


# SQD parameters for Hi-VQE
sqd_params = {'recovery_type': 1,
              'max_configurations': 10000,
              'add_hf_configuration': True}

# Get reference Hartree Fock state
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, n_orbitals*2, multiplicity)
print('hf reference', hf_reference_fock)

if True:
    simulator_can = simulator.copy()
    # Build LUCJ ansatz using canonical basis
    ucja_ansatz = UnitaryCoupledJastrowAnsatz(ccsd.t1,
                                              ccsd.t2,
                                              hf_reference_fock,
                                              n_terms=n_terms,
                                              full_trotter=True,
                                              local=local)


    energy = ucja_ansatz.get_energy(ucja_ansatz.parameters, hamiltonian, None)
    #energy = ucja_ansatz.get_sampled_energy(ucja_ansatz.parameters, hamiltonian, simulator_can, sqd_params)
    print('LUCJ energy canonical: ', energy)
    simulator_can.print_statistics()

# Build LUCJ ansatz local
#ccsd = molecule._pyscf_data.get('ccsd', None)
T1_orb = get_absolute_orbitals(ccsd.t1)  # a_j a_i^ -> a_j  a_i^
T2_orb = get_absolute_orbitals(ccsd.t2)  # a_j a_l a_i^ a_k^ ->  a_j a_l a_i^ a_k^


# generate amplitudes in the local basis
T1_orb_loc, T2_orb_loc = change_of_basis_orbitals(T1_orb, T2_orb, trans_mat)


# Build LUCJ ansatz using local basis with absolute amplitudes
ucja_ansatz_loc = UnitaryCoupledJastrowAnsatz(T1_orb_loc,
                                              T2_orb_loc,
                                              hf_reference_fock,
                                              n_terms=n_terms,
                                              full_trotter=True,
                                              local=local,
                                              use_general=True,
                                              separate_spins=True,
                                              reference_basis=trans_mat,
                                              connectivity_graph=G_qpu
                                              )


energy_loc = ucja_ansatz_loc.get_energy(ucja_ansatz_loc.parameters, hamiltonian_loc, None)
#energy_loc = ucja_ansatz_loc.get_sampled_energy(ucja_ansatz_loc.parameters, hamiltonian_loc_ord, simulator, sqd_params)

print('LUCJ energy localized abs:', energy_loc)


# change permutation
from vqemulti.utils import permute_amplitudes
T1_orb_loc_ord, T2_orb_loc_ord = permute_amplitudes(T1_orb_loc, T2_orb_loc, permutation_inv)
trans_mat_ord = trans_mat[:, permutation_inv]



# Build LUCJ ansatz using local basis with absolute amplitudes
ucja_ansatz_loc_ord = UnitaryCoupledJastrowAnsatz(T1_orb_loc_ord,
                                                  T2_orb_loc_ord,
                                                  hf_reference_fock,
                                                  n_terms=n_terms,
                                                  full_trotter=True,
                                                  local=local,
                                                  use_general=True,
                                                  separate_spins=True,
                                                  reference_basis=trans_mat_ord,
                                                  connectivity_graph=G_qpu
                                                  )


energy_loc_ord = ucja_ansatz_loc_ord.get_energy(ucja_ansatz_loc_ord.parameters, hamiltonian_loc_ord, None)
#energy_loc_ord = ucja_ansatz_loc_ord.get_sampled_energy(ucja_ansatz_loc_ord.parameters, hamiltonian_loc_ord, simulator, sqd_params)

print('LUCJ energy localized abs ord:', energy_loc_ord)
