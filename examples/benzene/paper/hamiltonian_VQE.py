from vqemulti.utils import get_hf_reference_in_fock_space, generate_reduced_hamiltonian
from vqemulti.pool import get_pool_singlet_sd, get_pool_singlet_gsd
from vqemulti.adapt_vqe import adaptVQE
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
from noise_model import get_noise_model
from openfermion import MolecularData, get_sparse_operator, get_fermion_operator
from vqemulti.utils import get_uccsd_operators
import numpy as np
import time


import warnings
warnings.filterwarnings("ignore")

config = Configuration()
config.verbose = 1
config.mapping = 'jw'


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


benzene = get_molecule('benzene.xyz')

frozen_core = 19
n_orb = 23

n_active = 4
frozen_core = 21 - n_active//2
n_orb = 21 + n_active//2


# run classical calculation
molecule = run_pyscf(benzene, run_fci=False, nat_orb=True, guess_mix=False, verbose=True, frozen_core=frozen_core, n_orbitals=n_orb)

# get properties from classical SCF calculation
n_electrons = molecule.n_electrons - frozen_core * 2
n_orbitals = n_orb - frozen_core  # molecule.n_orbitals
print('n_orbitals', n_orbitals)

hamiltonian = molecule.get_molecular_hamiltonian()
from vqemulti.utils import store_hamiltonian
store_hamiltonian(hamiltonian, file='hamiltonian.npz')


hamiltonian = get_fermion_operator(hamiltonian)
#hamiltonian.compress(5e-1)
#print(hamiltonian)

from vqemulti.utils import get_selected_ci_energy_dice, get_dmrg_energy

hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, n_orbitals*2)
configurations = [hf_reference_fock]
print(configurations)

energy_ci = get_selected_ci_energy_dice(configurations, hamiltonian, stream_output=False)
print('energy_ci', energy_ci)
print('n_electrons', n_electrons)


#energy_dmrg = get_dmrg_energy(hamiltonian, n_electrons, max_bond_dimension=200, stream_output=True, spin=None)
#print('energy_dmrg', energy_dmrg)

#exit()
#hamiltonian = get_fermion_operator(hamiltonian)
print('H terms (not compress)', len(hamiltonian.terms))
hamiltonian.compress(1e-2)
print('H terms (compress)', len(hamiltonian.terms))


matrix = get_sparse_operator(hamiltonian).toarray()
print('lowest eigenvalues:', np.sort(np.linalg.eigvals(matrix))[:4])


print('mapping: ', config.mapping)

#from openfermion.transforms import jordan_wigner, bravyi_kitaev
#h_jw = jordan_wigner(hamiltonian)
#h_jw.compress(1e-2)
#print('H terms qubit JW', len(h_jw.terms))
#h_bk = bravyi_kitaev(hamiltonian)
#h_bk.compress(1e-2)
#print('H terms qubit BK', len(h_bk.terms))


print('N electrons', n_electrons)
print('N Orbitals', n_orbitals)
# Choose specific pool of operators for adapt-VQE
#pool = get_pool_singlet_sd(n_electrons=n_electrons, n_orbitals=n_orbitals)
pool = get_pool_singlet_gsd(n_orbitals=n_orbitals)
print('len pool', len(pool))
uccsd_pool = get_uccsd_operators(n_electrons=n_electrons, n_orbitals=n_orbitals)
print('len UCCSD pool', len(uccsd_pool.terms))
#exit()



sym_orbitals = symmetrize_molecular_orbitals(molecule, 'D6h', skip=True, frozen_core=19, n_orbitals=23)
pool = get_symmetry_reduced_pool(pool, sym_orbitals, threshold=0.5)  # using symmetry
pool = pool.get_quibits_list(normalize=True)

print('len pool (symmetrized)', len(pool))

print('pool in Qubits')
for p in pool:
    print(p)

from vqemulti.symmetry import get_pauli_symmetry_reduced_pool
pool = get_pauli_symmetry_reduced_pool(pool)

print('pool Qubits symmetrized')
for p in pool:
    print(p)

exit()

#from vqemulti.pool.tools import OperatorList
#pool = OperatorList([pool[i] for i in range(0, len(pool), 4)])
# print('len pool', len(pool))


print('\nADAPT_VQE POOL\n---------------------------')
pool.print_compact_representation()

# Get Hartree Fock reference in Fock space
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, n_orbitals*2)
from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
# from vqemulti.simulators.penny_simulator import PennylaneSimulator as Simulator

from qiskit_aer import AerSimulator
backend = AerSimulator()

#from qiskit_ibm_runtime.fake_provider import FakeTorino, FakeAlmadenV2, FakeKyiv
#backend = FakeTorino()

#service = QiskitRuntimeService()
#backend = service.least_busy(simulator=False, operational=True)
#backend = service.backend('ibm_fez')

print('backend: ', backend.name)

# Start session
print('Initialize VQE')
n_shots = 1000

energy_list = []
evaluation_list = []

for _ in range(30):

    with Session(backend=backend) as session:

        simulator = Simulator(trotter=True,
                              trotter_steps=1,
                              test_only=True,
                              hamiltonian_grouping=True,
                              use_estimator=True,
                              # noise_model=get_noise_model(n_qubits=n_orbitals * 2, multiple=2), # 15
                              # session=session,
                              shots=n_shots,
                              )

        #simulator.set_shots_model(shots_model_energy)

        simulator_grad = Simulator(trotter=True,
                                   trotter_steps=1,
                                   test_only=True,
                                   hamiltonian_grouping=True,
                                   use_estimator=True,
                                   # noise_model=get_noise_model(n_qubits=n_orbitals * 2, multiple=2), # 10
                                   # session=session,
                                   shots=n_shots)
        #simulator_grad = None

        simulator_var = simulator_grad

        # simulator = None
        precision = 1e-3
        print('precision', precision)
        st = time.time()

        # define optimizer
        opt_bfgs = OptimizerParams(method='BFGS', options={'gtol': precision})
        opt_cobyla = OptimizerParams(method='COBYLA', options={'rhobeg': 0.1, 'maxiter': 60})
        opt_adam = OptimizerParams(method=adam, options={'learning_rate': 0.1, 'gtol': 1e-1})
        # cobyla modified
        opt_cobyla_mod = OptimizerParams(method=cobyla_mod, options={'rhobeg': 0.1,
                                                                     'n_guess': 8,
                                                                     'guess_range': np.pi})
        # define update method
        from vqemulti.method.adapt_vanila import AdapVanilla

        adapt_method = AdapVanilla(operator_update_number=1,
                                   operator_update_max_grad=precision,
                                   coeff_tolerance=precision*0.1,
                                   # diff_threshold=0,
                                   gradient_simulator=simulator_grad,
                                   min_iterations=7
                    )

        # run adaptVQE
        try:
            result = adaptVQE(hamiltonian,
                              pool,
                              hf_reference_fock,
                              energy_threshold=precision,
                              max_iterations=8,
                              energy_simulator=simulator,
                              #variance_simulator=simulator_var,
                              optimizer_params=opt_cobyla_mod,
                              method=adapt_method
                              )

        except NotConvergedError as e:
            result = e.results

    print('simulation energy')
    simulator.print_statistics()

    print('simulation gradient')
    simulator_grad.print_statistics()

    print("HF energy:", molecule.hf_energy)
    print("Final adaptVQE energy:", result["energy"])
    #print("FullCI energy:", molecule.fci_energy)
    print("CASCI energy:", molecule.casci_energy)

    print("Error (respect to CASCI): {:10.3e}".format(result["energy"] - molecule.casci_energy))

    print(result)
    print(result.keys())
    from vqemulti.utils import store_wave_function, load_wave_function

    store_wave_function(result['coefficients'], result['ansatz'], filename='wf_fermion_trotter_8.yml')
    #print('stored')
    #coeff_new, ansatz_new = load_wave_function(filename='wf_fermion.yml', qubit_op=False)
    #print('loaded')

    #print(result['iterations']['f_evaluations'])
    #print(result['iterations']['energies'])

    energy_list.append(result['iterations']['energies'])
    evaluation_list.append(result['iterations']['f_evaluations'])
    exit()


print('\nenergies')
for e in energy_list:
    print(e)

print('\nevaluations')
for e in evaluation_list:
    print(e)

exit()
print(hf_reference_fock)
from vqemulti.energy import get_adapt_vqe_energy
energy = get_adapt_vqe_energy(result['coefficients'],
                              result['ansatz'],
                              hf_reference_fock,
                              hamiltonian,
                              simulator
                              )

print('Energy restart:', energy)
exit()
