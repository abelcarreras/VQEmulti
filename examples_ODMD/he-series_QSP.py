#= = = IMPORTS = = =
import os
import numpy as np
from atomsystem import AtomSystem
from vqemulti.utils import get_hf_reference_in_fock_space
from vqemulti.ansatz.exponential import ExponentialAnsatz
from vqemulti.simulators.qiskit_simulator import QiskitSimulator as Simulator
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt

"""
 Here calculations of quantum signals for He-series based on protocols implemented in VQEmulti are shown.
 Run only when enough memory is available, as the Circuit simulation may take too long.
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

    # Output folder for txt files and plots
    out_dir = "he_series_results"
    os.makedirs(out_dir, exist_ok=True)

    fci_corr_lst = []
    dft_corr_lst = []
    e_dft_lst = []
    e_hf_lst = []
    e_fci_lst = []

    # Define the time values
    dt = 0.35  # time evolution steps
    n_time_steps = 50
    shots = 10000
    times = [dt * i for i in range(n_time_steps)]

    # simulator
    # config = Configuration()
    # config.verbose = 5
    # backend = FakeTorino()
    # service = QiskitRuntimeService()
    # backend = service.backend('ibm_basquecountry')
    backend = AerSimulator()

    simulator = Simulator(trotter=True,
                          trotter_steps=6,
                          test_only=False,
                          shots=shots,
                          backend=backend,
                          # use_ibm_runtime=True
                          )

    def get_overlap(i, Hamiltonian, hf_reference_fock, ref, simulator=None):

        generator = [-1j * Hamiltonian]
        psi = ExponentialAnsatz([dt * i], generator, hf_reference_fock)

        return ref.get_overlap(psi, simulator)


    for element in symbol_lst:
        print(f"\n========== {element} ==========")

        e_hf = atoms[element].run_rhf()
        e_fci = atoms[element].run_fci()
        e_dft = atoms[element].run_dft("b3lyp")

        print("RHF energy:", e_hf)
        print("FCI energy:", e_fci)
        print("DFT energy:", e_dft)

        corr = atoms[element].compare_energies()
        fci_corr_lst.append(corr["FCI"])
        dft_corr_lst.append(corr["DFT"])

        e_dft_lst.append(e_dft)
        e_hf_lst.append(e_hf)
        e_fci_lst.append(e_fci)

        # Get Hamiltonian and constant term
        Hamiltonian, constant_term = atoms[element].get_hamiltonian()
        print("Constant term:", constant_term)
        print("Number of qubits:", atoms[element].n_qubits)
        print("Number of Terms:", len(Hamiltonian.terms))

        # Compress?
        #Hamiltonian.compress(4e-2)

        # Reference
        hf_reference_fock = get_hf_reference_in_fock_space(atoms[element].n_electrons,
                                                           atoms[element].n_qubits)
        ref = ExponentialAnsatz([], [], hf_reference_fock)

        overlap_exact = []
        overlap_sim = []
        for i in range(n_time_steps):
            overlap_exact.append(get_overlap(i, Hamiltonian, hf_reference_fock, ref))
            overlap_sim.append(get_overlap(i, Hamiltonian, hf_reference_fock, ref, simulator))

        # --- Plot (saved, not shown) ---
        plt.figure()
        plt.title(f'Overlap {element}')
        plt.plot(times, np.real(overlap_exact), '-', label='exact real')
        plt.plot(times, np.imag(overlap_exact), '-', label='exact imag')
        plt.plot(times, np.real(overlap_sim), '--', label='sim real')
        plt.plot(times, np.imag(overlap_sim), '--', label='sim imag')
        plt.legend()
        plt.ylim(-1.01, 1.01)
        plt.xlabel('time')
        plt.savefig(os.path.join(out_dir, f"overlap_{element}_dt_{dt}_shots_{shots}.png"), dpi=300)
        plt.close()

        # --- Text files (same format as the printed output) ---
        with open(os.path.join(out_dir, f"overlap_{element}_dt_{dt}_exact.txt"), "w") as f:
            f.write('------- Exact -------\n')
            f.write(f'{element}\n')
            f.write('\ndt          Re(overlap)     Im(overlap)\n')
            for i in range(n_time_steps):
                f.write(f"{dt * i:.2f}    {np.real(overlap_exact[i]):.6f}    {np.imag(overlap_exact[i]):.6f}\n")

        with open(os.path.join(out_dir, f"overlap_{element}_dt_{dt}_shots_{shots}.txt"), "w") as f:
            f.write(f'------- Simulator ------- shots = {shots}\n')
            f.write(f'{element}\n')
            f.write('\ndt          Re(overlap)     Im(overlap)\n')
            for i in range(n_time_steps):
                f.write(f"{dt * i:.2f}    {np.real(overlap_sim[i]):.6f}    {np.imag(overlap_sim[i]):.6f}\n")

        print(f"Saved results for {element} in '{out_dir}/'")