import numpy as np
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from vqemulti.utils import log_message, log_section
from vqemulti.simulators.layout import accumulated_errors


def get_isa_layout(isa_circuit):
    """
    get original layout from isa circuit

    :param isa_circuit: Qiskit ISA circuit
    :return: list of qubits
    """
    if isa_circuit.layout is None:
        return None

    list_qubits = np.zeros(len(isa_circuit.qubits), dtype=int)
    n_qubits = 0
    for i, q in enumerate(isa_circuit.layout.initial_layout):
        if q._register.name != 'ancilla':
            # print(q, q._index, q._register, i)
            list_qubits[q._index] =  i
            n_qubits += 1
        if i > len(isa_circuit.qubits) - 2:
            break

    return list_qubits[:n_qubits].tolist()


class RHESampler:

    def __init__(self, backend, session, layout=None):

        self._pm = generate_preset_pass_manager(backend=backend,
                                                optimization_level=3,
                                                initial_layout=layout,
                                                # layout_method='dense'
                                                )

        self._session = session
        self._backend = backend

    def run(self, circuit, shots=1000, memory=True):

        isa_circuit = self._pm.run(circuit)

        from qiskit_ibm_runtime import SamplerV2
        mode = self._backend if self._session is None else self._session
        sampler = SamplerV2(mode=mode)
        job = sampler.run([isa_circuit], shots=shots)
        pub_result = job.result()[0]
        # counts_total = pub_result.data.meas.get_counts()

        log_message('depth: ', circuit.depth(), log_level=1)
        log_message('isa_depth: ', isa_circuit.depth(), log_level=1)
        log_message(str(isa_circuit.draw(fold=-1, reverse_bits=True, idle_wires=False)), log_level=1)
        log_message('isa_layout', get_isa_layout(isa_circuit), log_level=1)
        if log_section(log_level=1):
            accumulated_errors(self._backend, isa_circuit, print_data=True)


        # emulate Qiskit Sampler result interface
        class Result:
            def result(self):
                return pub_result.data.meas

        return Result()


class RHEstimator:

    def __init__(self, backend, session, layout=None, limit_hw=50000):

        self.pm = generate_preset_pass_manager(backend=backend,
                                               optimization_level=3,
                                               initial_layout=layout,
                                               # layout_method='dense'
                                               )
        self._session = session
        self._backend = backend
        self._limit_hw = limit_hw

        log_message('layout: ', layout, log_level=1)

    def run(self, circuit, measure_op, shots=10000):

        isa_circuit = self.pm.run(circuit)

        n_chunks = max([1, isa_circuit.depth() * len(measure_op) // self._limit_hw])
        n_chunks = min([n_chunks, len(measure_op)])
        chunck_size = int(np.ceil(len(measure_op)/n_chunks))

        log_message('depth: ', circuit.depth(), log_level=1)
        log_message('isa_depth: ', isa_circuit.depth(), log_level=1)
        log_message(str(isa_circuit.draw(fold=-1, reverse_bits=True, idle_wires=False)), log_level=1)
        log_message('isa_layout', get_isa_layout(isa_circuit), log_level=1)
        log_message('circuit gate count: ', dict(circuit.count_ops()), log_level=1)
        log_message('isa_circuit gate count: ', dict(isa_circuit.count_ops()), log_level=1)
        # log_message('observable: ', measure_op, log_level=1)
        log_message('n_observable: ', len(measure_op), log_level=1)
        log_message('n_chunks: ', n_chunks, log_level=1)
        log_message('chunck_size: ', chunck_size, log_level=1)
        if log_section(log_level=1):
            accumulated_errors(self._backend, isa_circuit, print_data=True)

        from qiskit_ibm_runtime import EstimatorV2

        try:
            # old version of qiskit
            estimator = EstimatorV2(session=self._session)
        except:
            mode = self._backend if self._session is None else self._session
            estimator = EstimatorV2(mode=mode)

        estimator.options.default_shots = shots
        estimator.options.resilience_level = 2
        # estimator.options.optimization_level = 0
        # estimator.options.dynamical_decoupling.enable = False
        # estimator.options.resilience.zne_mitigation = False
        # estimator.options.twirling.enable_measure = False
        # estimator.options.twirling.enable_gates = False
        # estimator.options.update(default_shots=shots, optimization_level=0)

        variance = 0
        expectation_value = 0

        for i in range(n_chunks):
            # print(i*chunck_size, (i+1)*chunck_size)

            measure_op_i = measure_op[i*chunck_size: (i+1)*chunck_size]
            if len(measure_op_i) == 0:
                continue

            mapped_observables = measure_op_i.apply_layout(isa_circuit.layout)

            precision = np.sqrt(0.04/shots)
            #print('precision: ', precision)

            job = estimator.run([(isa_circuit, mapped_observables)], precision=None)
            #print('metadata: ', job.result()[0].metadata)

            # shots = 4000  # current hypotesis shots are ignored and always uses this
            std = job.result()[0].data.stds # * np.sqrt(shots)
            variance += std ** 2 * shots * n_chunks

            expectation_value += job.result()[0].data.evs
            # print('expectationV2:', expectation_value)
            # print('varianceV2: ', variance, '/ ', job.result()[0].data.stds)
            # print('shots: ', shots)
            log_message(i+1, '/', n_chunks, log_level=1)
            log_message('partial expectation: ', expectation_value, log_level=1)
            log_message('metadata: ', job.result()[0].metadata, log_level=1)

        std = np.sqrt(variance/shots)
        log_message('Final Estimator value: ', expectation_value, log_level=1)

        # emulate Qiskit Estimator result interface
        class ResultData:
            def __init__(self):
                self.values = [expectation_value]
                self.metadata = [{'std_error': std}]

        class Result:
            def result(self):
                return ResultData()

        return Result()

