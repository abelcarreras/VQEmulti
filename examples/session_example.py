from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator, Session
from qiskit_ibm_provider import IBMProvider
from qiskit.circuit.random import random_circuit
import qiskit


# provider = IBMProvider(instance='ibm-q-ikerbasque/basq-ws-june23/project')

# service = QiskitRuntimeService(channel="ibm_cloud")


circuit_test = qiskit.QuantumCircuit(4)
circuit_test.x(2)
circuit_test.x(3)

print(circuit_test)


circuit = random_circuit(2, 2, seed=0, measure=True).decompose(reps=1)


psi1 = RealAmplitudes(num_qubits=2, reps=2)

H_test = SparsePauliOp.from_list([("IIII", 1)])

H1 = SparsePauliOp.from_list([("II", 1), ("IZ", 2), ("XI", 3)])
H2 = SparsePauliOp.from_list([("IZ", 1)])
H3 = SparsePauliOp.from_list([("ZI", 1), ("ZZ", 1)])

with Session(backend="ibmq_qasm_simulator") as session:
    estimator = Estimator(session=session)

    psi1_test = estimator.run(circuits=[circuit_test], observables=[H_test], shots=1000)
    print(psi1_test.result())


    theta1 = [0, 1, 1, 2, 3, 5]

    # calculate [ <psi1(theta1)|H1|psi1(theta1)> ]
    # psi1_H1 = estimator.run(circuits=[psi1], observables=[H1], parameter_values=[theta1], shots=100)
    psi1_H1 = estimator.run(circuits=[circuit], observables=[H1], shots=1000)

    print(psi1_H1.result())

    print(psi1_H1.result().values[0])

    # calculate [ <psi1(theta1)|H2|psi1(theta1)>, <psi1(theta1)|H3|psi1(theta1)> ]
    psi1_H23 = estimator.run(circuits=[psi1, psi1], observables=[H2, H3], parameter_values=[theta1]*2, shots=100)
    print(psi1_H23.result())
    # Close the session only if all jobs are finished
    # and you don't need to run more in the session
    session.close()