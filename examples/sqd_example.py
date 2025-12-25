from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
from openfermionpyscf import run_pyscf
from openfermion import MolecularData
from qiskit_ibm_runtime.fake_provider import FakeTorino
from qiskit_aer import AerSimulator
from vqemulti.ansatz.unitary_jastrow import crop_local_amplitudes, UnitaryCoupledJastrowAnsatz
from vqemulti.preferences import Configuration
from vqemulti.sqd import simulate_energy_sqd

config = Configuration()
# config.verbose = 2
config.mapping = 'jw'

simulator = Simulator(trotter=True,
                      trotter_steps=1,
                      test_only=False,
                      hamiltonian_grouping=True,
                      use_estimator=True, shots=100000,
                      backend=FakeTorino(),
                      # use_ibm_runtime=True
                      )


hydrogen = MolecularData(geometry=[('N', [0.0, 0.0, 0.0]),
                                   ('N', [2.0, 0.0, 0.0])],
                         basis='sto-3g',
                         multiplicity=1,
                         charge=0,
                         description='molecule')

# run classical calculation
n_frozen_orb = 2  # nothing
n_total_orb = 8  # total orbitals
molecule = run_pyscf(hydrogen, run_fci=False, nat_orb=False, guess_mix=False, verbose=True,
                     frozen_core=n_frozen_orb, n_orbitals=n_total_orb, run_ccsd=True)

n_electrons = molecule.n_electrons - n_frozen_orb * 2
print('N_electrons: ', n_electrons)
print('N_orb: ', n_total_orb - n_frozen_orb)

hamiltonian = molecule.get_molecular_hamiltonian()

ccsd = molecule._pyscf_data.get('ccsd', None)
t2 = crop_local_amplitudes(ccsd.t2, n_neighbors=3)
t1 = ccsd.t1

ucja = UnitaryCoupledJastrowAnsatz(None, t2, n_terms=1, full_trotter=True)

print('compute energy')
energy = ucja.get_energy(ucja.parameters, hamiltonian, simulator)

simulator.print_statistics()
print(simulator.get_circuits()[-1])
print('Jastrow energy: ', energy)

# SQD
energy, samples = simulate_energy_sqd(ucja,
                                      hamiltonian,
                                      simulator,
                                      n_electrons,
                                      return_samples=True)


# order samples
print('\nsamples')
sorted_configurations = sorted(samples.items(),
                               key=lambda item: item[1],
                               reverse=True)
for k, v in sorted_configurations:
    print('{} {}'.format(k, v))

print('\nSQD energy', energy)
