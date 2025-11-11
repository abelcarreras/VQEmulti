set_token = False
check_status = True


if set_token:
    from qiskit_ibm_runtime import QiskitRuntimeService

    # Save an IBM Quantum account and set it as your default account.
    QiskitRuntimeService.save_account(
        # channel="ibm_quantum",
        instance='Generic project_on-prem',
        token="RD7ZY6dkSZbCWTUSdIk56AmIwam0dWQ6AuGjBFYc7DaI",
        set_as_default=True,
        # Use `overwrite=True` if you're updating your token.
        overwrite=True,
    )

    # Load saved credentials
    service = QiskitRuntimeService()


if check_status:

    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService()

    for backend in service.backends():
        #print(backend.name, "-", backend.backend_name, "-", backend.status().status_msg)
        print(f"\n  Backend: {backend.name}")
        print(f"  Number of qubits: {backend.configuration().n_qubits}")

        status = backend.status()
        print(f"  Status: {status.status_msg}")
        print(f"  Active: {status.operational}")
        print(f"  Pending job: {status.pending_jobs}")

