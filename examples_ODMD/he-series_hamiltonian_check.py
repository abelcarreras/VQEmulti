"""
 This script can be used to test hamiltonian_diagnostics.py
 and get prior information about the Hamiltonian of a given system.
"""
import numpy as np
from atomsystem import AtomSystem
from vqemulti.utils import get_hf_reference_in_fock_space
from hamiltonian_diagnostics import diagnose

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

    e_hf = atoms["Li1+"].run_rhf()
    e_fci = atoms["Li1+"].run_fci()
    e_dft = atoms["Li1+"].run_dft("b3lyp")

    print("RHF energy:", e_hf)
    print("FCI energy:", e_fci)
    print("DFT energy:", e_dft)

    # Get Hamiltonian and constant term for Li1+
    Hamiltonian, constant_term = atoms["Li1+"].get_hamiltonian()

    # Reference
    hf = get_hf_reference_in_fock_space(atoms["Li1+"].n_electrons,
                                                       atoms["Li1+"].n_qubits)

    for sig in ("complex", "real"):
        print("=" * 72)
        print(f"Diagnosing Hamiltonian for {sig} signals")
        diagnose(Hamiltonian, hf, signal=sig).report(dt_current=None, trotter_bias=1e-9)