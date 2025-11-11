from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
from vqemulti.ansatz import get_ucc_ansatz
from openfermionpyscf import run_pyscf
from openfermion import MolecularData
from vqemulti.utils import get_hf_reference_in_fock_space
from vqemulti.energy import get_vqe_energy, get_adapt_vqe_energy
from vqemulti.operators import n_particles_operator, spin_z_operator, spin_square_operator
from vqemulti.preferences import Configuration
from vqemulti.energy.simulation import simulate_energy_sqd
import numpy as np


def print_ranked_samples(samples, n_electrons, multiplicity=0):
    import numpy as np
    from vqemulti.utils import get_fock_space_vector

    alpha_electrons = (multiplicity + n_electrons)//2
    beta_electrons = (n_electrons - multiplicity)//2


    # print(list(samples.values()))
    values = list(samples.values())
    indices = np.argsort(values)

    correct_samples = 0
    keys = list(samples.keys())
    for i in indices[::-1]:

        bitstring = keys[i]
        fock_vector = get_fock_space_vector([1 if b == '1' else 0 for b in bitstring[::-1]])
        if np.sum(fock_vector[::2]) == alpha_electrons and np.sum(fock_vector[1::2]) == beta_electrons:
            print(keys[i][::-1], values[i])
            correct_samples += values[i]

    return correct_samples


#Configuration().verbose = 2

simulator = Simulator(trotter=True,
                      trotter_steps=1,
                      test_only=True,
                      hamiltonian_grouping=True,
                      use_estimator=True,
                      shots=100000)

hydrogen = MolecularData(geometry=[('H', [0.0, 0.0, 0.0]),
                                   ('H', [2.0, 0.0, 0.0]),
                                   ('H', [4.0, 0.0, 0.0]),
                                   ('H', [6.0, 0.0, 0.0])],
                         basis='sto-3g',
                         multiplicity=1,
                         charge=0,
                         description='molecule')

# run classical calculation
n_frozen_orb = 0
n_total_orb = 4
molecule = run_pyscf(hydrogen, run_fci=True, nat_orb=False, guess_mix=False, verbose=True,
                     frozen_core=n_frozen_orb, n_orbitals=n_total_orb, run_ccsd=True)

hamiltonian = molecule.get_molecular_hamiltonian()

n_electrons = molecule.n_electrons - n_frozen_orb * 2
n_orbitals = n_total_orb - n_frozen_orb  # molecule.n_orbitals
n_qubits = n_orbitals * 2

hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, n_qubits)

hf_energy = get_vqe_energy([], [], hf_reference_fock, hamiltonian, None)
print('energy HF: ', hf_energy)

coefficients, ansatz = get_ucc_ansatz(molecule.ccsd_single_amps, molecule.ccsd_double_amps, use_qubit=True)

# print initial energy
ucc_energy = get_adapt_vqe_energy(coefficients,
                                  ansatz,
                                  hf_reference_fock,
                                  hamiltonian,
                                  simulator)

print('UCC energy: ', ucc_energy)

energy, samples = simulate_energy_sqd(coefficients, ansatz, hf_reference_fock, hamiltonian, simulator, n_electrons,
                                      adapt=True,
                                      return_samples=True)

n_samples = print_ranked_samples(samples, n_electrons)
print('samples: ', n_samples)
print('11110000: ', samples['00001111'])
print('SQD energy: ', energy)
print('\n===========================\n')

# projector
from openfermion import QubitOperator


proj = QubitOperator()
for i in range(n_electrons):
    proj += QubitOperator('X{}'.format(i))


ansatz.append(-1.0j * proj)
coefficients.append(1.0)

ucc_energy_proj = get_adapt_vqe_energy(coefficients,
                                       ansatz,
                                       hf_reference_fock,
                                       hamiltonian,
                                       simulator)

print('UCC-proj energy: ', ucc_energy_proj)
print('UCC-proj + HF energy ', ucc_energy_proj + hf_energy)


energy, samples = simulate_energy_sqd(coefficients,
                                      ansatz,
                                      hf_reference_fock,
                                      hamiltonian,
                                      simulator,
                                      n_electrons,
                                      adapt=True,
                                      return_samples=True)


n_samples = print_ranked_samples(samples, n_electrons)
print('samples: ', n_samples)
try:
    hf_samples = samples['00001111']
except KeyError:
    hf_samples = 0

print('11110000: ', hf_samples)
print('SQD energy: ', energy)

exit()


sample_hf_list = []
sqd_energy_list = []
for theta in np.linspace(0, np.pi, 10, endpoint=True):
    coefficients, ansatz = get_ucc_ansatz(molecule.ccsd_single_amps, molecule.ccsd_double_amps, use_qubit=True)

    # theta = 1
    ansatz.append(-1.0j * theta * proj)
    coefficients.append(1)

    ucc_energy_proj = get_adapt_vqe_energy(coefficients,
                                           ansatz,
                                           hf_reference_fock,
                                           hamiltonian,
                                           simulator)

    print('UCC-proj energy: ', ucc_energy_proj)
    print('UCC-proj + HF energy ', ucc_energy_proj + hf_energy)


    energy, samples = simulate_energy_sqd(coefficients,
                                          ansatz,
                                          hf_reference_fock,
                                          hamiltonian,
                                          simulator,
                                          n_electrons,
                                          adapt=True,
                                          return_samples=True)


    n_samples = print_ranked_samples(samples, n_electrons)
    print('samples: ', n_samples)
    try:
        hf_samples = samples['00001111']
    except KeyError:
        hf_samples = 0

    print('11110000: ', hf_samples)
    print('SQD energy: ', energy)

    sample_hf_list.append(hf_samples)
    sqd_energy_list.append(energy)


import matplotlib.pyplot as plt
plt.plot(sample_hf_list)
plt.figure()

plt.plot(sqd_energy_list)
plt.show()