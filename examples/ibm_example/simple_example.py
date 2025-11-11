from qiskit_ibm_runtime import Session, QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeTorino
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2
from qiskit_ibm_runtime import EstimatorV1, Options
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit import qpy
from qiskit_aer.primitives import Estimator as EstimatorV1Aer
from qiskit_aer.primitives import EstimatorV2 as EstimatorV2Aer
import numpy as np
import json

import warnings
warnings.filterwarnings("ignore")

# define backend
backend = FakeTorino()


# service = QiskitRuntimeService()
# backend = service.least_busy(simulator=False, operational=True)
# backend_ibm = service.backend('ibm_torino')
# noise_model = NoiseModel.from_backend(backend_ibm)
# backend = AerSimulator()


print('backend: ', backend.name)
print()

with Session(backend=backend) as session:

    shots = 1000

    # load observables
    with open('observables.json', "r") as f:
        data = json.load(f)

    measure_op = SparsePauliOp.from_list(data)

    # load circuit
    with open('circuit.qpy', 'rb') as f:
        circuit = qpy.load(f)[0]

    # Aer estimator V1
    estimator = EstimatorV1Aer(abelian_grouping=True)
    job = estimator.run(circuits=[circuit], observables=[measure_op], shots=shots)

    expectation_value = sum(job.result().values)
    variance = sum([meta['variance'] for meta in job.result().metadata])

    print('Estimator qiskit_aer V1')
    print('expectation V1: ', expectation_value)
    print('variance V1: ', variance)
    print()

    # Aer estimator V2
    estimator = EstimatorV2Aer()
    estimator.options.default_shots = shots
    estimator.options.resilience_level = 0
    estimator.options.optimization_level = 0

    job = estimator.run([(circuit, measure_op)], precision=None)

    std = job.result()[0].data.stds * np.sqrt(shots)
    variance = std**2

    expectation_value = job.result()[0].data.evs

    print('Estimator qiskit_aer V2')
    print('expectation V2: ', expectation_value)
    print('variance V2: ', variance)
    print()


    # transpile circuit
    layout = [0, 1, 2, 3, 4, 5, 6, 7]
    pm = generate_preset_pass_manager(backend=backend,
                                      optimization_level=3,
                                      initial_layout=layout,
                                      # layout_method='dense'
                                      )

    isa_circuit = pm.run(circuit)

    mapped_observables = measure_op.apply_layout(isa_circuit.layout)

    # Using Estimator V1
    estimator = EstimatorV1(session=session, options=Options(optimization_level=0, resilience_level=0))
    job = estimator.run(circuits=[isa_circuit], observables=[mapped_observables], shots=shots)
    variance = sum([meta['variance'] for meta in job.result().metadata])
    expectation_value = sum(job.result().values)

    print('Estimator runtime V1')
    print('expectationV1: ', expectation_value)
    print('varianceV1: ', variance)
    print()

    # Using Estimator V2
    estimator = EstimatorV2(session=session)
    estimator.options.default_shots = shots
    estimator.options.resilience_level = 0
    estimator.options.optimization_level = 0

    job = estimator.run([(isa_circuit, mapped_observables)], precision=None)

    std = job.result()[0].data.stds * np.sqrt(shots)
    variance = std**2

    expectation_value = job.result()[0].data.evs

    print('Estimator runtime V2')
    print('expectationV2:', expectation_value)
    print('varianceV2: ', variance)


