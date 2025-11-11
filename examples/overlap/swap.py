from qiskit.providers.basic_provider import BasicSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
#from qiskit_ibm_runtime import SamplerV1
from qiskit_ibm_runtime.fake_provider import FakeTorino
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Estimator, Sampler


# ------------------------------------



# Creem un circuit amb 3 qubits: c (control), a, b (target)
qc = QuantumCircuit(3, 1)
qc.h(0)

qc.barrier()
qc.h(1)

qc.x(2)
qc.h(2)

qc.barrier()

# Fredkin gate
qc.cx(2, 1)
qc.ccx(0, 1, 2)
qc.cx(2, 1)

qc.barrier()

qc.h(0)

qc.measure(0, 0)
print(qc)

# Visualitzem el circuit




# Prepare and run the circuit in simulator
backend = AerSimulator()
n_shots = 100000
result = backend.run(qc, shots=n_shots).result()
print(result)
counts = result.get_counts()

print("Counts simulator:", counts)
p0 = counts['0']/n_shots if '0' in counts else 0.0
overlap = 2 * p0 - 1
print('overlap:', overlap)




# Executem el circuit en un simulador
#simulator = Aer.get_backend('statevector_simulator')
#result = execute(qc, simulator).result()
#statevector = result.get_statevector()
#print(statevector)