import os
import numpy as np
from pyscf import ao2mo, fci
from openfermion import InteractionOperator, jordan_wigner, get_sparse_operator
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.linalg import jw_get_ground_state_at_particle_number
from qiskit.quantum_info import SparsePauliOp

# FUNCTIONS
# ----------------------------------------------------------------------------
#Read .npz
# ----------------------------------------------------------------------------
def load_ints(fname):
    data = np.load(fname)
    h0 = float(data["hc"])
    h1 = np.asarray(data["h1e"])
    h2 = np.asarray(data["h2e"])
    norb = h1.shape[0]
    # ao2mo returns a compact (npair, npair) array -> expand to (n, n, n, n)
    if h2.ndim != 4:
        h2 = ao2mo.restore(1, h2, norb)
    return h0, h1, h2

# ----------------------------------------------------------------------------
#OpenFermion fermionic + qubit Hamiltonian
# ----------------------------------------------------------------------------
def build_openfermion_hamiltonian(h0, h1, h2_chem):
    #OpenFermion stores the coefficient of $a^\dagger_p a^\dagger_q a_r a_s$ (with the ½ in front),
    # which is the physicists’ integral ⟨pq|sr⟩ = (ps|qr).
    # That’s here weneed to transpose(0, 2, 3, 1) before building the InteractionOperator.
    h2_phys = np.asarray(h2_chem.transpose(0, 2, 3, 1), order="C")
    one_body, two_body = spinorb_from_spatial(h1, h2_phys)
    fermion_op = InteractionOperator(h0, one_body, 0.5 * two_body)
    qubit_op = jordan_wigner(fermion_op)
    qubit_op.compress()                       # drop ~0 terms
    return fermion_op, qubit_op

def openfermion_to_qiskit(qubit_op, n_qubits):
    """
    from_sparse_list takes explicit qubit indices, so Qiskit's little-endian
    label ordering is handled automatically (OpenFermion qubit i -> Qiskit qubit i).
    """
    sparse_list = []
    for term, coeff in qubit_op.terms.items():
        paulis = "".join(p for _, p in term)
        indices = [i for i, _ in term]
        sparse_list.append((paulis, indices, coeff))   # term == () -> identity
    op = SparsePauliOp.from_sparse_list(sparse_list, num_qubits=n_qubits)
    op = op.simplify()
    if np.allclose(op.coeffs.imag, 0.0):
        op = SparsePauliOp(op.paulis, op.coeffs.real)
    return op


# Construct Hamiltonian from stored integrals in npz file
# Stored integrals are in Chemist's notation, and are converted to Physicist's notation for use in OpenFermion.
integrals_dir = "He_series_RHF_integrals_sto-3g"

# For Lithium cation
h0, h1, h2 = load_ints(os.path.join(integrals_dir, f"Li1+_sto-3g_integrals.npz"))
norb = h1.shape[0]
n_qubits = 2 * norb

# We know the number of electrons for Li1+ is 2 (1s^2)
n_electrons = 2

# NOTE: WE WANT TO EXCLUDE THE CONSTANT TERM , WE CAN MAKE IT EQUAL TO ZERO
# THE COMPRESS FUNCTION WILL REMOVE THE CONSTANT TERM FROM THE HAMILTONIAN
print("Constant term (h0):", h0)

print("Number of qubits:", n_qubits)
#fermion_op, qubit_op = build_openfermion_hamiltonian(h0, h1, h2)
fermion_op, qubit_op = build_openfermion_hamiltonian(0.0, h1, h2)
print("Number of terms in QubitOperator:", len(qubit_op.terms))
#H_qiskit = openfermion_to_qiskit(qubit_op, n_qubits)
#print("Number of terms in Qiskit SparsePauliOp:", len(H_qiskit.paulis))
#print(H_qiskit)

# From here we can continue with the C(t) construction using VQEmulti
from vqemulti.utils import get_hf_reference_in_fock_space
from vqemulti.ansatz.exponential import ExponentialAnsatz
from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt

# Define the time values
dt = 0.106  # time evolution steps
n_time_steps = 30
shots = 10000
times = [dt * i for i in range(n_time_steps)]

# simulator
# config = Configuration()
# config.verbose = 5
# backend = FakeTorino()
# service = QiskitRuntimeService()
# backend = service.backend('ibm_basquecountry')
backend = AerSimulator()

simulator = Simulator(trotter=True,
                      trotter_steps=6,
                      test_only=False,
                      shots=shots,
                      backend=backend,
                      # use_ibm_runtime=True
                      )

def get_overlap(i, Hamiltonian, hf_reference_fock, ref, simulator=None):

    generator = [-1j * Hamiltonian]
    psi = ExponentialAnsatz([dt * i], generator, hf_reference_fock)

    return ref.get_overlap(psi, simulator)


# Compress?
#Hamiltonian.compress(4e-2)

# Reference
hf_reference_fock = get_hf_reference_in_fock_space(n_electrons, n_qubits)
ref = ExponentialAnsatz([], [], hf_reference_fock)

overlap_exact = []
overlap_sim = []
for i in range(n_time_steps):
    overlap_exact.append(get_overlap(i, qubit_op, hf_reference_fock, ref))
    overlap_sim.append(get_overlap(i, qubit_op, hf_reference_fock, ref, simulator))

# --- Plot (saved, not shown) ---
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
print('\ndt          Re(overlap)     Im(overlap)')
for i in range(n_time_steps):
    print(f"{dt * i:.2f}    {np.real(overlap_exact[i]):.6f}    {np.imag(overlap_exact[i]):.6f}")

print('\n------- Simulator -------')
print('\ndt          Re(overlap)     Im(overlap)')
for i in range(n_time_steps):
    print(f"{dt * i:.2f}    {np.real(overlap_sim[i]):.6f}    {np.imag(overlap_sim[i]):.6f}")