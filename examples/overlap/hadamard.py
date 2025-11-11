import numpy as np
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
#from qiskit_ibm_runtime import SamplerV1
from qiskit_ibm_runtime.fake_provider import FakeTorino
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Estimator, Sampler
import matplotlib.pyplot as plt

# ------------------------------------

def simulate_expectation(qc, n_shots=100000):
    # Prepare and run the circuit in simulator
    backend = AerSimulator()
    qc = qc.decompose()

    result = backend.run(qc, shots=n_shots).result()

    counts = result.get_counts()

    print("Counts simulator:", counts)
    p0 = counts['0'] / n_shots if '0' in counts else 0.0
    p1 = counts['1'] / n_shots if '1' in counts else 0.0

    overlap = p0 - p1
    return overlap



def build_state_psi(qc, params_psi):
    #qc.x(0)
    qc.rz(params_psi[0]*2, 0)

def build_state_phi(qc, params_psi):
    #qc.x(0)
    qc.rz(params_psi[0]*2, 0)


real_list = []
imag_list = []

for theta in np.linspace(0.0, 2*np.pi, 50):

    nqubits = 3

    # |psi>
    q = QuantumRegister(nqubits, 'qubit')
    circ_psi = QuantumCircuit(q)
    params_psi = [theta]
    build_state_psi(circ_psi, params_psi)
    print('circ_psi')
    print(circ_psi)


    # |phi>
    q = QuantumRegister(nqubits, 'qubit')
    circ_phi = QuantumCircuit(q)
    params_phi = [-theta]
    build_state_phi(circ_phi, params_phi)
    print('circ_phi')
    print(circ_phi)
    #print(circ_phi.inverse())

    # Append it to the existing psi circuit, then extract it
    U_phi_dagger = circ_phi.inverse()
    U_circ = circ_psi.copy()
    U_circ.compose(U_phi_dagger, inplace=True)
    print('U_circ')
    print(U_circ)

    # Then create a controlled gate out of it
    U_controlled_gate = (U_circ.to_gate(label="U")).control(1)


    # create circuit with 3 qubits and 1 ancilla
    anc = QuantumRegister(1, 'ancilla')
    q = QuantumRegister(3, 'qubit')
    c_reg = ClassicalRegister(1, 'c_bit')

    qc = QuantumCircuit(anc, q, c_reg)

    qc.h(anc[0])
    qc.append(U_controlled_gate, [anc[0], q[0], q[1], q[2]])
    qc.h(anc[0])
    qc.measure(0, c_reg)

    print(qc)


    real_overlap = simulate_expectation(qc, n_shots=10000)

    # Creem un circuit amb 3 qubits: c (control), a, b (target)
    qc = QuantumCircuit(anc, q, c_reg)

    qc.h(anc[0])
    qc.s(anc[0])
    qc.append(U_controlled_gate, [anc[0], q[0], q[1], q[2]])
    qc.h(anc[0])
    qc.measure(0, c_reg)


    imaginary_overlap = simulate_expectation(qc, n_shots=10000)

    print('overlap: ', real_overlap + imaginary_overlap * 1j)

    real_list.append(real_overlap)
    imag_list.append(imaginary_overlap)

plt.plot(real_list, label='real')
plt.plot(imag_list, label='imag')
plt.legend()
plt.show()
