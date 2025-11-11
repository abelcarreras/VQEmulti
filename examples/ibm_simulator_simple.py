# Use the following code instead if you want to run on a simulator:
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Create a new circuit with two qubits
qc = QuantumCircuit(2)

# Add a Hadamard gate to qubit 0
qc.h(0)

# Perform a controlled-X gate on qubit 1, controlled by qubit 0
qc.cx(0, 1)

# Return a drawing of the circuit using MatPlotLib ("mpl"). This is the
# last line of the cell, so the drawing appears in the cell output.
# Remove the "mpl" argument to get a text drawing.
print(qc.draw())



# Set up six different observables.
from qiskit.quantum_info import SparsePauliOp

observables_labels = ["IZ", "IX", "ZI", "XI", "ZZ", "XX"]
observables = [SparsePauliOp(label) for label in observables_labels]

from qiskit_ibm_runtime import QiskitRuntimeService

# If you did not previously save your credentials, use the following line instead:
# service = QiskitRuntimeService(channel="ibm_quantum", token="<MY_IBM_QUANTUM_TOKEN>")
service = QiskitRuntimeService()

from qiskit_ibm_runtime.fake_provider import FakeAlmadenV2
#backend = FakeAlmadenV2()
backend = service.least_busy(simulator=False, operational=True)
#backend = service.backend('ibm_cusco')

# Convert to an ISA circuit and layout-mapped observables.
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_circuit = pm.run(qc)

print(isa_circuit.draw(idle_wires=False))


mapped_observables = [
    observable.apply_layout(isa_circuit.layout) for observable in observables
]


if True:
    from qiskit_ibm_runtime import EstimatorV1 as Estimator
    from qiskit_ibm_runtime import Options

    options = Options(optimization_level=1)
    estimator = Estimator(backend)

    job = estimator.run(circuits=[isa_circuit]*len(mapped_observables), observables=mapped_observables, shots=10000)  # , abelian_grouping=True)

    # This is the result of the entire submission.  You submitted one Pub,
    # so this contains one inner result (and some metadata of its own).
    job_result = job.result()

    import numpy as np
    values = job_result.values
    errors = [np.sqrt(meta['variance']) for meta in job_result.metadata]

else:
    from qiskit_ibm_runtime import EstimatorV2 as Estimator
    estimator = Estimator(backend)

    job = estimator.run([(isa_circuit, mapped_observables)], precision=1e-3)
    pub_result = job.result()[0]
    values = pub_result.data.evs
    errors = pub_result.data.stds


# plotting graph
from matplotlib import pyplot as plt
plt.plot(observables_labels, values, '-o')
plt.xlabel('Observables')
plt.ylabel('Values')
plt.show()