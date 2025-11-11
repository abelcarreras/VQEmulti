from vqemulti.utils import get_hf_reference_in_fock_space, generate_reduced_hamiltonian
from vqemulti.pool import get_pool_singlet_sd
from vqemulti.adapt_vqe import adaptVQE
from vqemulti.energy import get_adapt_vqe_energy
from vqemulti.energy.simulation import simulate_adapt_vqe_variance, simulate_energy_sqd
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
from vqemulti.gradient import compute_gradient_vector, simulate_gradient
from vqemulti.utils import store_hamiltonian, store_wave_function, load_wave_function, load_hamiltonian
from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
import numpy as np

import warnings

def get_n_particles_operator(n_orbitals, to_pauli=False):
    from openfermion import FermionOperator, jordan_wigner

    sz = FermionOperator()
    for i in range(0, n_orbitals * 2):
        a = FermionOperator((i, 1))
        a_dag = FermionOperator((i, 0))
        sz += a * a_dag

    if to_pauli:
        sz = jordan_wigner(sz)
    return sz

warnings.filterwarnings("ignore")

config = Configuration()
config.verbose = False
config.mapping = 'jw'

import warnings
warnings.filterwarnings("ignore")



def get_molecule(filename):
    """
    read molecule from xyz file

    :param filename: XYZ file
    :return: pyscf molecule
    """

    with open(filename, 'r') as f:
        lines = f.readlines()[2:]

    symbols = []
    coordinates = []

    for line in lines:
        symbols.append(line.split()[0])
        coordinates.append(line.split()[1:])

    coordinates = np.array(coordinates, dtype=float)

    geometry = []
    for s, c in zip(symbols, coordinates):
        geometry.append([s, [c[0], c[1], c[2]]])

    mol = MolecularData(geometry=geometry,
                        basis='sto-3g',
                        multiplicity=1,
                        charge=0,
                        description='molecule')

    return mol


# molecule definition

# from vqemulti.utils import generate_reduced_hamiltonian, get_uccsd_operators
# print('len UCC: ', len(get_uccsd_operators(4, 4).terms))

simulator = Simulator(trotter=True,
                      trotter_steps=1,
                      test_only=True,
                      hamiltonian_grouping=True,
                      use_estimator=True,
                      shots=200)
remake = False
if remake:
    benzene = get_molecule('benzene.xyz')

    # run classical calculation
    frozen_orb = 19
    n_total = 24
    molecule = run_pyscf(benzene, run_fci=False, nat_orb=True, guess_mix=False, verbose=True,
                         frozen_core=frozen_orb,
                         n_orbitals=n_total)

    print('hf_energy: ', molecule.hf_energy)
    # get properties from classical SCF calculation
    n_electrons = molecule.n_electrons - frozen_orb * 2
    n_orbitals = n_total - frozen_orb  # molecule.n_orbitals

    hamiltonian = molecule.get_molecular_hamiltonian()

    store_hamiltonian(hamiltonian, file='hamiltonian.npz')

    #hamiltonian = get_fermion_operator(hamiltonian)
    #print('H terms (not compress)', len(hamiltonian.terms))
    #hamiltonian.compress(1e-2)
    #print('H terms (compress)', len(hamiltonian.terms))

    matrix = get_sparse_operator(hamiltonian).toarray()
    print('lowest eigenvalues:', np.sort(np.linalg.eigvals(matrix))[:4])

    print('N electrons', n_electrons)
    print('N Orbitals', n_orbitals)
    # Choose specific pool of operators for adapt-VQE
    pool = get_pool_singlet_sd(n_electrons=n_electrons, n_orbitals=n_orbitals)

    sym_orbitals = symmetrize_molecular_orbitals(molecule, 'D6h', skip=True, frozen_core=frozen_orb, n_orbitals=n_total)
    pool = get_symmetry_reduced_pool(pool, sym_orbitals, threshold=0.5)  # using symmetry
    # pool = pool.get_quibits_list(normalize=True)

    print('len pool', len(pool))

    hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, n_orbitals*2)
    print('hf_reference_fock:', hf_reference_fock)

    # Start session
    print('Initialize Single point')

    precision = 1e-2

    # define update method
    from vqemulti.method.adapt_vanila import AdapVanilla

    adapt_method = AdapVanilla(operator_update_number=1,
                               operator_update_max_grad=precision,
                               gradient_threshold=1e-8,
                               # coeff_tolerance=precision*0.1,
                               # diff_threshold=0,
                               # gradient_simulator=simulator_grad,
                               diff_threshold=0,
                               min_iterations=0)
    try:
        result = adaptVQE(hamiltonian,
                          pool,
                          hf_reference_fock,
                          energy_threshold=precision,
                          max_iterations=1,
                          energy_simulator=simulator,
                          # variance_simulator=simulator_var,
                          optimizer_params=OptimizerParams(method='BFGS', options={'gtol': precision}),
                          method=adapt_method
                          )
    except NotConvergedError as e:
        result = e.results

    # run SP energy
    # simulator = None

    print(result['coefficients'])
    print(result['ansatz'])

    store_wave_function(result['coefficients'], result['ansatz'], filename='wf_fermion_1.yml')
    coefficients, ansatz = result['coefficients'], result['ansatz']

else:
    hamiltonian = load_hamiltonian('hamiltonian.npz')
    coefficients, ansatz = load_wave_function(filename='wf_fermion_1.yml', qubit_op=False)
    # hf_reference_fock = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    n_electrons = 4
    n_orbitals = 5
    hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, n_orbitals*2)


#print(hf_reference_fock)
#hf_reference_fock = [1, 0, 0, 1, 1, 0, 0, 1, 0, 0]
hf_energy = get_adapt_vqe_energy([], [], hf_reference_fock, hamiltonian, None)
print('energy HF: ', hf_energy)
# exit()

#coefficients = coefficients[:0]
#ansatz = ansatz[:0]

energy = get_adapt_vqe_energy(coefficients,
                              ansatz,
                              hf_reference_fock,
                              hamiltonian,
                              simulator
                              )

#print(simulator.get_circuits()[-1])
#print(simulator.print_statistics())
print('energy VQE: ', energy)

energy = simulate_energy_sqd(coefficients,
                             ansatz,
                             hf_reference_fock,
                             hamiltonian,
                             simulator,
                             n_electrons,
                             generate_random=True,
                             adapt=True)

print('final SDQ energy: ', energy)

# DMRG sampled SCI
from vqemulti.utils import get_dmrg_energy, get_selected_ci_energy_dice
for bd in range(1, 20, 1):
    dmrg_energy, configurations = get_dmrg_energy(hamiltonian, n_electrons, max_bond_dimension=bd, sample=0.001,
                                                  occupations=[2, 2, 0, 0, 0],
                                                  stream_output=False)
    # print(configurations)
    sci_energy = get_selected_ci_energy_dice(configurations, hamiltonian, stream_output=False)
    print('energy: {:5} {:5} {:20.12f} {:20.12f}'.format(bd, len(configurations), dmrg_energy, sci_energy))
    # exit()



for n_shots in range(100, 650, 50):
    simulator = Simulator(trotter=True,
                          trotter_steps=1,
                          test_only=True,
                          hamiltonian_grouping=True,
                          use_estimator=True,
                          shots=n_shots)

    # random sampling
    e_list = []
    for _ in range(1):
        energy = simulate_energy_sqd(coefficients,
                                     ansatz,
                                     hf_reference_fock,
                                     hamiltonian,
                                     simulator,
                                     n_electrons,
                                     generate_random=True,
                                     adapt=True)

        e_list.append(energy)
    print('energy SQD RND: ', n_shots, np.average(e_list))

    #print(simulator.get_circuits()[-1])
    #simulator.print_statistics()

