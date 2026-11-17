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
from vqemulti.utils import odmd
import matplotlib.pyplot as plt
import numpy as np


def separate_identity_term(qubit_operator):

    result = QubitOperator()

    constant = 0
    for term, coefficient in qubit_operator.terms.items():
        if term:
            result += QubitOperator(term, coefficient)
        else:
            constant += coefficient

    return result, constant

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

distance = 0.74
hydrogen = MolecularData(geometry=[('H', [0.0, 0.0, 0.0]),
                                   ('H', [0.0, 0.0, distance])],
                         basis='sto-3g',
                         multiplicity=1,
                         charge=0,
                         description='molecule')

# run reference calculation
molecule = run_pyscf(hydrogen,
                     run_fci=True, nat_orb=False, guess_mix=False, verbose=True,
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

# ODMD analysis
print("========== ODMD RESULT =========")

# Signal
delay = n_time_steps//2
time = [dt * i for i in range(n_time_steps)]

print(f" Signal: exact overlap")
C = overlap_exact
postproc = odmd.ODMD(d=delay, tol=0.05, svd_rank=-1, sorted_eigs="abs", reduced=True)
postproc.fit(overlap_exact)

for i, lam in enumerate(postproc.eigs):
    mu = np.log(lam) / dt
    print(f"[ODMD] {i}: decay={mu.real}, omega={mu.imag}")

selected_eigs = np.asarray(postproc.eigs)
print(selected_eigs)
fig, (ax_signal, ax_eigs) = plt.subplots(1, 2, figsize=(14, 5))

if hasattr(postproc, "reconstructed_data"):
    ax_signal.plot(time, C, "x-", linewidth=1, label="Input Data")
    ax_signal.plot(
        time, postproc.reconstructed_data[0].real, "x-", linewidth=1,
        label="ODMD reconstruction",
    )
    ax_signal.legend()

ax_signal.set_title(f"ODMD Result - Signal exact overlap")
ax_signal.set_xlabel("Time")
ax_signal.set_ylabel("Signal")
ax_signal.grid(True)

ax_eigs.scatter(selected_eigs.real, selected_eigs.imag, color='red', zorder=3)
ax_eigs.add_patch(
    plt.Circle((0, 0), 1, fill=False, color="black", linestyle="--")
)
ax_eigs.axhline(0, color="black", linewidth=0.7)
ax_eigs.axvline(0, color="black", linewidth=0.7)

ax_eigs.set_aspect("equal", adjustable="datalim")
ax_eigs.set_title("ODMD Eigenvalues")
ax_eigs.set_xlabel("Real")
ax_eigs.set_ylabel("Imaginary")
ax_eigs.grid(True)
fig.tight_layout()
plt.show()