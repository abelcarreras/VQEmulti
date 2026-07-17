# Run the circuit using Sampler
from qiskit_ibm_catalog import QiskitFunctionsCatalog
from vqemulti.utils import log_message
from vqemulti.simulators.cache import JobCache


class QCTRLJob:
    def __init__(self, job):
        self._job = job

    def result(self):
        return self._job.result()[0].data.meas

    def status(self):
        return self._job.status()

    def job_id(self):
        return self._job.job_id()



class QCTRLSampler:
    def __init__(self, backend):
        self._backend = backend

        log_message('using Q-CRTL qiskit function', log_level=1)

        catalog = QiskitFunctionsCatalog()
        # print(catalog.list())
        self._cache = JobCache()


        self._perf_mgmt = catalog.load("q-ctrl/performance-management")


    def run(self, circuit, shots=1000, memory=True):

        qctrl_sampler_job = self._cache.get_job(circuit, {'n_shots': shots}, 'sampler_qctrl', self._backend.name)

        if qctrl_sampler_job is None:

            sampler_pubs = [(circuit,)]

            qctrl_sampler_job = self._perf_mgmt.run(primitive="sampler",
                                                    pubs=sampler_pubs,
                                                    options={"default_shots": shots},
                                                    backend_name=self._backend.name)

            #log_message('job ID: ', qctrl_sampler_job.job_id, log_level=1)
            self._cache.store_job(qctrl_sampler_job, circuit, {'n_shots': shots}, 'sampler_qctrl', self._backend.name)

        return QCTRLJob(qctrl_sampler_job)
