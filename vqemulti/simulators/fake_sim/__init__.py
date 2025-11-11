import os
from qiskit_ibm_runtime.fake_provider import fake_backend


class FakeIdeal(fake_backend.FakeBackendV2):
    """A fake 133 qubit backend."""

    dirname = os.path.dirname(__file__)  # type: ignore
    conf_filename = "conf_ideal.json"  # type: ignore
    props_filename = None # "props_ideal.json"  # type: ignore
    defs_filename = None #"defs_ideal.json"  # type: ignore
    backend_name = "fake_ideal"  # type: ignore
