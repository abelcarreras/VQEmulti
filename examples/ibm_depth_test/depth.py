import numpy as np
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2, Session
from qiskit.circuit.random import random_circuit
from qiskit_ibm_runtime import Estimator, Options, EstimatorV2, EstimatorV1
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from vqemulti.simulators.backend_opt import get_backend_opt_layout
import matplotlib.pyplot as plt
import qiskit



def get_circuit(n_qubits=4, n_blocks=1):
    circuit_test = qiskit.QuantumCircuit(n_qubits)

    for _ in range(n_blocks):

        for i in range(n_qubits):
            circuit_test.h(i)

        circuit_test.barrier()

        for i in range(0, n_qubits-1):
            circuit_test.cx(i, i+1)

        for i in range(n_qubits - 2, -1, -1):
            circuit_test.cx(i, i + 1)

        circuit_test.barrier()

        for i in range(n_qubits):
            circuit_test.h(i)

        circuit_test.barrier()

    return circuit_test


def generate_pauli_string(n_qubits, index):
    import itertools
    pauli_operators = ['X', 'Y', 'Z']
    total_combinations = 3 ** n_qubits

    if index >= total_combinations:
        raise ValueError(
            "Index out of range. Maximum index for {} qubits is {}.".format(n_qubits, total_combinations - 1))

    # Generate all possible combinations of 'X', 'Y', 'Z' for the given number of qubits
    all_combinations = list(itertools.product(pauli_operators, repeat=n_qubits))

    # Get the specific combination based on the index
    return ''.join(all_combinations[index])


depth_list = []
value_list = []
variance_list = []

real = 0


# fake provider (use for test)
from qiskit_ibm_runtime.fake_provider import FakeTorino
backend = FakeTorino()

# real provider (use for real)
#service = QiskitRuntimeService()
#backend = service.least_busy(simulator=False, operational=True)
# backend = service.backend('ibm_cusco')
print('backend: ', backend.name)

with Session(backend=backend) as session:

    #for n_blocks in range(200, 1000, 10):
    for n_oper in range(6, 2000, 10):
        n_blocks = 1
        n_qubits = 4

        print('n_blocks: ', n_blocks)
        print('n_operators: ', n_oper)

        #circuit = random_circuit(n_qubits, 2, seed=0, measure=False).decompose(reps=1)

        circuit = get_circuit(n_qubits, n_blocks)
        # print(circuit)

        #H_test = SparsePauliOp.from_list([(generate_pauli_string(n_qubits, i), float(i)) for i in range(n_oper)])
        H_test = SparsePauliOp.from_list([('X'*n_qubits, 1)])
        print(H_test)

        measure_op = H_test
        if real == 0:

            layout = get_backend_opt_layout(backend, n_qubits)
            print('layout: ', layout)

            pm = generate_preset_pass_manager(backend=backend,
                                              optimization_level=0,
                                              initial_layout=layout,
                                              # layout_method='dense'
                                              )
            isa_circuit = pm.run(circuit)

            depth = isa_circuit.depth()
            print('depth: ', depth)

            mapped_observables = measure_op.apply_layout(isa_circuit.layout)

            estimator = EstimatorV2(session=session)
            #estimator = EstimatorV2(backend=backend)

            estimator.options.default_shots = 10000000
            estimator.options.resilience.zne_mitigation = False
            estimator.options.update(default_shots=10000000, optimization_level=0)

            job = estimator.run([(isa_circuit, mapped_observables)], precision=None)

            print(job.result())
            std = job.result()[0].data.stds
            variance = std ** 2
            expectation_value = job.result()[0].data.evs

            print('expectation: ', expectation_value)
            print('variance: ', variance)
            print('std: ', std)

            value_list.append(expectation_value)
            variance_list.append(variance)
            depth_list.append(depth)

        if real == 1:

            layout = get_backend_opt_layout(backend, n_qubits)
            print('layout: ', layout)

            pm = generate_preset_pass_manager(backend=backend,
                                              optimization_level=0,
                                              initial_layout=layout,
                                              # layout_method='dense'
                                              )
            isa_circuit = pm.run(circuit)

            depth = isa_circuit.depth()
            print('depth: ', depth)

            mapped_observables = measure_op.apply_layout(isa_circuit.layout)

            estimator = EstimatorV1(session=session)

            job = estimator.run(circuits=[isa_circuit], observables=[mapped_observables], shots=10000000)
            variance = sum([meta['variance'] for meta in job.result().metadata])
            expectation_value = sum(job.result().values)

            print('expectation: ', expectation_value)
            print('variance: ', variance)
            print('std: ', np.sqrt(variance))

            value_list.append(expectation_value)
            variance_list.append(variance)
            depth_list.append(depth)

        elif real == 2:
            from qiskit_aer.primitives import Estimator

            estimator = Estimator(abelian_grouping=False)

            depth = circuit.depth()
            print('depth: ', depth)

            job = estimator.run(circuits=[circuit], observables=[measure_op], shots=10000000)

            expectation_value = sum(job.result().values)

            # get variance
            variance = sum([meta['variance'] for meta in job.result().metadata])

            print('expectation: ', expectation_value)
            print('variance: ', variance)

            value_list.append(expectation_value)
            variance_list.append(variance)
            depth_list.append(depth)

        elif real == 3:
            from qiskit_aer.primitives import Estimator

            estimator = Estimator(abelian_grouping=False)

            depth = circuit.depth()
            print('depth: ', depth)

            len_piece = 5
            n_blocks = int(np.ceil(len(measure_op)/len_piece))
            print('n_block', n_blocks)

            expectation_value = 0
            variance = 0

            for i in range(n_blocks):
                print('pas', i)
                measure_op_i = measure_op[i*len_piece: (i+1)*len_piece]
                print(measure_op_i)
                job = estimator.run(circuits=[circuit], observables=[measure_op_i], shots=10000000)

                expectation_value += sum(job.result().values)

                # get variance
                variance += sum([meta['variance'] for meta in job.result().metadata])

            print('expectation: ', expectation_value)
            print('variance: ', variance)

            value_list.append(expectation_value)
            variance_list.append(variance)
            depth_list.append(depth)
        exit()


print(depth_list)
print('{:^10} {:^10} {:^10}'.format('depth', 'value_list', 'variance_list'))
for i, depth in enumerate(depth_list):
    print('{:10} {:10.5f} {:10.5f}'.format(depth, value_list[i], variance_list[i]))

plt.plot(depth_list, value_list, label='expectation')
plt.plot(depth_list, variance_list, label='variance')
plt.legend()
plt.show()