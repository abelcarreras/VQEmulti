from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
from openfermionpyscf import run_pyscf
from openfermion import MolecularData
from vqemulti.ansatz.unitary_jastrow import crop_local_amplitudes, UnitaryCoupledJastrowAnsatz
from vqemulti.preferences import Configuration
from vqemulti.sqd import simulate_energy_sqd
from vqemulti.simulators.layout import LayoutModelSQD, LayoutModelLinear, LayoutModelDefault
from vqemulti.utils import get_selected_ci_energy_dice, get_dmrg_energy
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeTorino
from qiskit_aer import AerSimulator

config = Configuration()
config.verbose = 1
config.mapping = 'jw'
config.temp_dir = '/Users/abel/test_dice/'

# list of backends
backend = AerSimulator()
#backend = FakeTorino()
#service = QiskitRuntimeService()
#backend = service.least_busy(simulator=False, operational=True)
#backend = service.backend('ibm_basquecountry')
print('backend: ', backend)


hydrogen = MolecularData(geometry=[('N', [0.0, 0.0, 0.0]),
                                   ('N', [2.0, 0.0, 0.0])],
                         basis='sto-3g',
                         multiplicity=1,
                         charge=0,
                         description='molecule')

# run classical calculation
n_frozen_orb = 2  # nothing
n_total_orb = 10  # total orbitals

molecule = run_pyscf(hydrogen, run_fci=False, nat_orb=False, guess_mix=False, verbose=True,
                     frozen_core=n_frozen_orb, n_orbitals=n_total_orb, run_ccsd=True)

n_electrons = molecule.n_electrons - n_frozen_orb * 2
n_orbitals = n_total_orb - n_frozen_orb
print('N_electrons: ', n_electrons)
print('N_orb: ', n_orbitals)

# get hamiltonian
hamiltonian = molecule.get_molecular_hamiltonian()


# make selected-CI HCI calculation
configuration_HF = [
    [1]*n_electrons + [0]*(2*n_orbitals - n_electrons)  # HF
]
hci_energy = get_selected_ci_energy_dice(configuration_HF, hamiltonian,
                                         stream_output=False,
                                         hci_schedule=[(0, 1e-3),(100, 1e-6)])

print('HCI_energy:', hci_energy)

energy_dmrg = get_dmrg_energy(hamiltonian, n_electrons, max_bond_dimension=100)
print('dmrg_energy:', energy_dmrg)

ccsd = molecule._pyscf_data.get('ccsd', None)
t2 = crop_local_amplitudes(ccsd.t2, n_neighbors=3)
t1 = ccsd.t1

ucja = UnitaryCoupledJastrowAnsatz(None, t2, n_terms=1, full_trotter=True)

layout_model = LayoutModelSQD()
# layout_model = LayoutModelLinear()

#layout_model.plot_data(backend, ucja.n_qubits)
#exit()

simulator = Simulator(trotter=True,
                      trotter_steps=1,
                      test_only=False,
                      hamiltonian_grouping=True,
                      use_estimator=True, shots=100,
                      backend=backend,
                      use_ibm_runtime=True,
                      # layout_model=layout_model # use default
                      )


# analysis of UCJ initial guess state
if False:
    print('compute energy')
    energy = ucja.get_energy(ucja.parameters, hamiltonian, simulator)
    print('Jastrow energy: ', energy)

    simulator.print_statistics()
    print(simulator.get_circuits()[-1])

# SQD
print('SQD start')
energy, samples = simulate_energy_sqd(ucja,
                                      hamiltonian,
                                      simulator,
                                      n_electrons,
                                      generate_random=True,
                                      return_samples=True)

# order samples
print('\nsamples')
sorted_configurations = sorted(samples.items(),
                               key=lambda item: item[1],
                               reverse=True)
for k, v in sorted_configurations:
    print('{} {}'.format(k, v))

print('\nSQD energy', energy)
