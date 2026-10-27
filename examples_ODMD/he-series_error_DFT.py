#= = = IMPORTS = = =
import numpy as np
from atomsystem import AtomSystem
"""
 Here we show the performance of DFT calculations for the atomic He-series and show the
 errors associated to the lack of electron correlation
"""

if __name__ == "__main__":
    # Example for the whole he-series
    atoms = {
        "He": AtomSystem([("He", [0.0, 0.0, 0.0])], charge=0, spin=0, basis="sto-3g"),
        "Li1+": AtomSystem([("Li", [0.0, 0.0, 0.0])], charge=1, spin=0, basis="sto-3g"),
        "Be2+": AtomSystem([("Be", [0.0, 0.0, 0.0])], charge=2, spin=0, basis="sto-3g"),
        "B3+": AtomSystem([("B", [0.0, 0.0, 0.0])], charge=3, spin=0, basis="sto-3g"),
        "C4+": AtomSystem([("C", [0.0, 0.0, 0.0])], charge=4, spin=0, basis="sto-3g"),
        "N5+": AtomSystem([("N", [0.0, 0.0, 0.0])], charge=5, spin=0, basis="sto-3g"),
        "O6+": AtomSystem([("O", [0.0, 0.0, 0.0])], charge=6, spin=0, basis="sto-3g"),
        "F7+": AtomSystem([("F", [0.0, 0.0, 0.0])], charge=7, spin=0, basis="sto-3g"),
        "Ne8+": AtomSystem([("Ne", [0.0, 0.0, 0.0])], charge=8, spin=0, basis="sto-3g")
    }
    symbol_lst = ["He", "Li1+", "Be2+", "B3+", "C4+", "N5+", "O6+", "F7+", "Ne8+"]

    fci_corr_lst = []
    dft_corr_lst = []
    e_dft_lst = []
    e_hf_lst = []
    e_fci_lst = []
    for element in symbol_lst:
        e_hf = atoms[element].run_rhf()
        #print("RHF energy:", e_hf)
        e_dft = atoms[element].run_dft("b3lyp")
        print("DFT (B3LYP) energy:", e_dft)
        print(e_dft)
        e_fci = atoms[element].run_fci()
        #print("FCI energy:", e_fci)

        corr = atoms[element].compare_energies()
        fci_corr_lst.append(corr["FCI"])
        dft_corr_lst.append(corr["DFT"])

        e_dft_lst.append(e_dft)
        e_hf_lst.append(e_hf)
        e_fci_lst.append(e_fci)

    # Plot energies
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(np.arange(2, len(symbol_lst)+2), e_hf_lst, "o-", label="RHF")
    plt.plot(np.arange(2, len(symbol_lst)+2), e_fci_lst, "r*", label="FCI")
    plt.plot(np.arange(2, len(symbol_lst)+2), e_dft_lst, "k-",label="B3LYP")
    plt.legend()
    plt.xlabel("Atomic number")
    plt.ylabel("Energy (a.u.)")
    plt.title("He-series")
    plt.show()

    plt.figure()
    plt.plot(np.arange(2, len(symbol_lst)+2), fci_corr_lst, "r*", linewidth=2, label="FCI")
    plt.plot(np.arange(2,len(symbol_lst)+2), dft_corr_lst, "k-",linewidth=2, label="B3LYP")
    plt.xlabel("Atomic number")
    plt.ylabel("Correlation Energy (a.u.)")
    plt.title("He-series Correlation Energy")
    plt.legend()
    plt.show()

    print(fci_corr_lst)
    print(dft_corr_lst)









