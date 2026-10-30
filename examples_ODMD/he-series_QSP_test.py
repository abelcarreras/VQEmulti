#= = = IMPORTS = = =
import numpy as np
from atomsystem import AtomSystem
from vqemulti.utils import get_hf_reference_in_fock_space
from vqemulti.ansatz.exponential import ExponentialAnsatz
from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt

"""
 Here calculations of quantum signals for He-series based on protocols implemented in VQEmulti are shown.
 """

if __name__ == "__main__":
    # Example for the whole he-series
    atoms = {
        "He" : AtomSystem([("He", [0.0, 0.0, 0.0])], charge=0, spin=0, basis="sto-3g"),
        "Li1+": AtomSystem([("Li", [0.0, 0.0, 0.0])], charge=1, spin=0, basis="sto-3g"),
        "Be2+":AtomSystem([("Be", [0.0, 0.0, 0.0])], charge=2, spin=0, basis="sto-3g"),
        "B3+": AtomSystem([("B", [0.0, 0.0, 0.0])], charge=3, spin=0, basis="sto-3g"),
        "C4+": AtomSystem([("C", [0.0, 0.0, 0.0])], charge=4, spin=0, basis="sto-3g"),
        "N5+": AtomSystem([("N", [0.0, 0.0, 0.0])], charge=5, spin=0, basis="sto-3g"),
        "O6+": AtomSystem([("O", [0.0, 0.0, 0.0])], charge=6, spin=0, basis="sto-3g"),
        "F7+": AtomSystem([("F", [0.0, 0.0, 0.0])], charge=7, spin=0, basis="sto-3g"),
        "Ne8+":AtomSystem([("Ne", [0.0, 0.0, 0.0])], charge=8, spin=0, basis="sto-3g")
    }

    symbol_lst = ["He", "Li1+", "Be2+", "B3+", "C4+", "N5+", "O6+", "F7+", "Ne8+"]

    fci_corr_lst = []
    dft_corr_lst = []
    e_dft_lst = []
    e_hf_lst = []
    e_fci_lst = []

    e_hf = atoms["Ne8+"].run_rhf()
    e_fci = atoms["Ne8+"].run_fci()
    e_dft = atoms["Ne8+"].run_dft("b3lyp")

    print("RHF energy:", e_hf)
    print("FCI energy:", e_fci)
    print("DFT energy:", e_dft)

    # Get Hamiltonian and constant term for Ne8+
    Hamiltonian, constant_term = atoms["Ne8+"].get_hamiltonian()
    print("Hamiltonian:")
    for term, coeff in Hamiltonian.terms.items():
        print(f"{term}: {coeff}")

    print("Constant term:", constant_term)
    print("Number of qubits:", atoms["Ne8+"].n_qubits)
    print("Number of Terms:", len(Hamiltonian.terms))

    # Compress?
    #Hamiltonian.compress(4e-2)

    # Reference
    hf_reference_fock = get_hf_reference_in_fock_space(atoms["Ne8+"].n_electrons,
                                                       atoms["Ne8+"].n_qubits)

    print(hf_reference_fock)
    ref = ExponentialAnsatz([], [], hf_reference_fock)
    print("ExponentialAnsatz:", ref)

    # Define the time values
    dt = 0.01  # time evolution steps
    n_time_steps = 50
    overlap_exact = []
    overlap_sim = []

    # simulator
    # config = Configuration()
    # config.verbose = 5
    # backend = FakeTorino()
    # service = QiskitRuntimeService()
    # backend = service.backend('ibm_basquecountry')
    backend = AerSimulator()

    simulator = Simulator(trotter=True,
                          trotter_steps=4,
                          test_only=False,
                          shots=10000,
                          backend=backend,
                          # use_ibm_runtime=True
                          )

    def get_overlap(i, simulator=None):

        generator = [-1j * Hamiltonian]
        psi = ExponentialAnsatz([dt * i], generator, hf_reference_fock)

        return ref.get_overlap(psi, simulator)


    for i in range(n_time_steps):
        overlap_exact.append(get_overlap(i))
        overlap_sim.append(get_overlap(i, simulator))

    plt.figure()
    plt.title('Overlap')
    plt.plot([dt * i for i in range(n_time_steps)], np.real(overlap_exact), '-', label='exact real')
    plt.plot([dt * i for i in range(n_time_steps)], np.imag(overlap_exact), '-', label='exact imag')

    plt.title('Overlap')
    plt.plot([dt * i for i in range(n_time_steps)], np.real(overlap_sim), '--', label='sim real')
    plt.plot([dt * i for i in range(n_time_steps)], np.imag(overlap_sim), '--', label='sim imag')

    plt.legend()
    plt.ylim(-1.01, 1.01)
    plt.xlabel('time')
    plt.show()

    print('\n------- Exact -------')
    print(symbol_lst[1])
    print('\ndt          Re(overlap)     Im(overlap)')
    for i in range(n_time_steps):
        print(f"{dt * i:.2f}    {np.real(overlap_exact[i]):.6f}    {np.imag(overlap_exact[i]):.6f}")

    print('\n------- Simulator -------')
    print(symbol_lst[1])
    print('\ndt          Re(overlap)     Im(overlap)')
    for i in range(n_time_steps):
        print(f"{dt * i:.2f}    {np.real(overlap_sim[i]):.6f}    {np.imag(overlap_sim[i]):.6f}")










