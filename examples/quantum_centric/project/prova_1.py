from qiskit import QuantumCircuit #execute
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.visualization import plot_histogram
from qiskit_aer import Aer
import numpy as np

# 4 qubits, HF = |1010>
n_qubits = 4
hf_state = [1,0,1,0]  # ocupacions de HF

# angle de reducció parcial (menor que pi)
theta = np.pi / 2  # ajusta per quant vols reduir HF

qc = QuantumCircuit(n_qubits)


n_qubits = 4
hf_state = [1,0,1,0]
theta = np.pi / 3  # controla quant amplitud transferim a l'ancilla

# 5 qubits total (4 + 1 ancilla)
qc = QuantumCircuit(n_qubits + 1)

# 1️⃣ Prepara un estat de prova
qc.h(range(n_qubits))

# 2️⃣ Porta HF a |1111>
for q, bit in enumerate(hf_state):
    if bit == 0:
        qc.x(q)

# 3️⃣ Controlled-Ry parcial sobre ancilla
# Objectiu: qubit ancilla (últim qubit)
controls = list(range(n_qubits))
target = n_qubits
qc.mcry(2*theta, controls, target)

# 4️⃣ Desfer X gates
for q, bit in enumerate(hf_state):
    if bit == 0:
        qc.x(q)

print(qc)
# 5️⃣ Mesurar
qc.measure_all()

# Simulació
# Prepare and run the circuit in simulator
backend = BasicSimulator()
result = backend.run(qc, shots=100000).result()
counts = result.get_counts()



# Mostrem el resultat
import matplotlib.pyplot as plt
plot_histogram(counts)
plt.show()