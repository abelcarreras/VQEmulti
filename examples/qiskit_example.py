from qiskit.providers.basic_provider import BasicSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
#from qiskit_ibm_runtime import SamplerV1
from qiskit_ibm_runtime.fake_provider import FakeTorino
from qiskit import QuantumCircuit

# Initialising a circuit with 2 qubits and 2 classical bits
gc = QuantumCircuit(2,2)

# Some gates
gc.h([0,1])
gc.barrier()
gc.cx(0,1)
gc.barrier()
gc.h([0,1])
gc.cx(0,1)
gc.h([0,1])

# Measure qbits [0,1] into classical bits [0, 1]
gc.measure([0,1], [0,1])

# Draw the circuit
# gc.draw(output = 'mpl')
print(gc)

# ------------------------------------


# Prepare and run the circuit in simulator
backend = BasicSimulator()
result = backend.run(gc, shots=1024).result()
counts = result.get_counts()

print("Counts simulator:", counts)


# ------------------------------------


# fake provider (use for test)
backend = FakeTorino()

# real provider (use for real)
service = QiskitRuntimeService()
#backend = service.least_busy(simulator=False, operational=True)
#backend = service.backend('ibm_torino')



print('backend:', backend.name)

pm = generate_preset_pass_manager(backend=backend,
                                  optimization_level=1,
                                  # initial_layout=layout,
                                  # layout_method='dense'
                                  )
isa_circuit = pm.run(gc)

# use sampler primitive
sampler = Sampler(backend=backend)

job = sampler.run([isa_circuit], shots=1000)
pub_result = job.result()[0]
counts = pub_result.data.c.get_counts()


print("Counts real hardware:", counts)

sampler = SamplerV1(backend=backend)

job = sampler.run([isa_circuit], shots=1000)
pub_result = job.result()
print(dir(pub_result))
print(pub_result.quasi_dists)
counts = pub_result.metadata
print('countsV1: ', counts)
