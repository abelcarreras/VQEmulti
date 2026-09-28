import matplotlib.pyplot as plt
import numpy as np
from openfermionpyscf import run_pyscf
from openfermion import MolecularData
from vqemulti.utils import get_hf_reference_in_fock_space
from vqemulti.utils import get_dmrg_energy, fermion_to_qubit
from vqemulti.ansatz.exponential import ExponentialAnsatz
from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeTorino
from openfermion import QubitOperator
from vqemulti.preferences import Configuration

def separate_identity_term(qubit_operator):

    result = QubitOperator()

    constant = 0
    for term, coefficient in qubit_operator.terms.items():
        if term:
            result += QubitOperator(term, coefficient)
        else:
            constant += coefficient

    return result, constant

def esprit(signal, n_components, dt):
    """
    Estimate angular frequencies of a complex signal using ESPRIT.

    Parameters
    ----------
    signal : array_like
        Complex-valued signal samples x[n].
    n_components : int
        Number of exponential components to extract.
    dt : float
        Sampling interval.

    Returns
    -------
    frequencies : ndarray
        Estimated angular frequencies in radians per unit of time.
    amplitudes : ndarray
        Estimated complex amplitudes.
    """
    signal = np.asarray(signal, dtype=complex)
    n_samples = len(signal)

    if n_components >= n_samples // 2:
        raise ValueError("n_components must be smaller than len(signal) / 2.")

    n_rows = n_samples // 2
    n_cols = n_samples - n_rows + 1

    X = np.empty((n_rows, n_cols), dtype=complex)

    for i in range(n_rows):
        X[i, :] = signal[i:i + n_cols]

    U, _, _ = np.linalg.svd(X, full_matrices=False)
    U_signal = U[:, :n_components]

    U_1 = U_signal[:-1, :]
    U_2 = U_signal[1:, :]

    Psi = np.linalg.pinv(U_1) @ U_2

    eigenvalues, _ = np.linalg.eig(Psi)

    frequencies = np.angle(eigenvalues) / dt

    order = np.argsort(frequencies)
    frequencies = frequencies[order]

    n = np.arange(n_samples)
    Vandermonde = np.exp(1j * np.outer(n * dt, frequencies))

    amplitudes, *_ = np.linalg.lstsq(
        Vandermonde,
        signal,
        rcond=None
    )

    return frequencies, amplitudes

# Define the time values
dt = 0.5  # time evolution steps


# simulator
#config = Configuration()
#config.verbose = 5

#backend = FakeTorino()
# service = QiskitRuntimeService()
# backend = service.backend('ibm_basquecountry')
backend = AerSimulator()

simulator = Simulator(trotter=True,
                      trotter_steps=6,
                      test_only=False,
                      shots=100,
                      backend=backend,
                      #use_ibm_runtime=True
                      )


distance = 2.0 # 0.74
hydrogen = MolecularData(geometry=[('H', [0.0, 0.0, 0.0]),
                                   ('H', [0.0, 0.0, distance])],
                         basis='sto-3g',
                         multiplicity=1,
                         charge=0,
                         description='molecule')

# run reference calculation
molecule = run_pyscf(hydrogen,
                     run_fci=False, nat_orb=False, guess_mix=False, verbose=True,
                     frozen_core=0, n_orbitals=4, run_ccsd=False, run_casci=True)

n_electrons = molecule.n_electrons
n_orbitals = molecule.n_orbitals
n_qubits = molecule.n_qubits

print('N_electrons: ', n_electrons)
print('N_orb: ', n_orbitals)

# get hamiltonian
hamiltonian = molecule.get_molecular_hamiltonian()
hamiltonian_te = fermion_to_qubit(hamiltonian)
hamiltonian_te, constant = separate_identity_term(hamiltonian_te)
print('H terms:', len(hamiltonian_te.terms))
#hamiltonian_te.compress(1e-1)
print('H terms compress:', len(hamiltonian_te.terms))
print('Constant:', constant)

# FCI energy
e_fci = molecule.casci_energy
print('e_fci: ', e_fci)

# reference
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, n_qubits)
ref = ExponentialAnsatz([], [], hf_reference_fock)

def get_overlap(i, simulator=None):

    generator = [-1j * hamiltonian_te]
    psi = ExponentialAnsatz([dt * i], generator, hf_reference_fock)

    return ref.get_overlap(psi, simulator)

n_time_steps = 50
overlap_exact = []
overlap_sim = []

for i in range(n_time_steps):
    overlap_exact.append(get_overlap(i))
    overlap_sim.append(get_overlap(i, simulator))

plt.figure()
plt.title('Overlap')
plt.plot([dt * i for i in range(n_time_steps)], np.real(overlap_exact), '-', label='exact real')
plt.plot([dt * i for i in range(n_time_steps)], np.imag(overlap_exact), '-', label='exact imag')

plt.title('Overlap')
plt.plot([dt * i for i in range(n_time_steps)], np.real(overlap_sim), '--', label='sim real')
plt.plot([dt * i for i in range(n_time_steps)], np.imag(overlap_sim), '--', label='sim imag')

plt.legend()
plt.ylim(-1.01, 1.01)
plt.xlabel('time')
plt.show()

print('\n------- Exact -------')
frequencies, amplitudes = esprit(overlap_exact, 2, dt)
for i, (f, a) in enumerate(zip(frequencies[::-1], amplitudes[::-1])):
    print('\nState', i+1)
    print(' Frequency: ', f)
    print(' Amplitude: ', a.real)
    print(' Energy: ', -f + constant)

print('\n------- Simulator -------')
frequencies, amplitudes = esprit(overlap_sim, 2, dt)
for i, (f, a) in enumerate(zip(frequencies[::-1], amplitudes[::-1])):
    print('\nState', i+1)
    print(' Frequency: ', f)
    print(' Amplitude: ', a.real)
    print(' Energy: ', -f + constant)
