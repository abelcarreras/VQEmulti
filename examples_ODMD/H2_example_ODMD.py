import cmath
import numpy as np
import matplotlib.pyplot as plt
import tools
import odmd

from pydmd import HankelDMD
from pydmd.preprocessing import hankel_preprocessing
from pydmd.plotter import plot_eigs

# This example uses quantum signal input data precomputed for H2
# Runs the ODMD algorithm and presents the results for the estimation of
# the ground state energy.

# READ DATA from files (only the real part of the signal)
times = []
signals = []

titles = ["H2_exact_evol.txt",
          "H2_AER_500_shots.dat",
          "H2_AER_1000_shots.dat",
          "H2_FakeTorino_500_shots.dat",
          "H2_FakeTorino_1000_shots.dat",]

for title in titles:
    time, C, ReC, ImC = tools.read_signal_file(title)
    times.append(time)
    signals.append(ReC)

delay = 25
dt = 0.5

# Plot all signals
fig = plt.figure(figsize=(10, 6), dpi=200)
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
for time, signal, title in zip(times, signals, titles):
    plt.plot(time, signal, 'x-',label=title, linewidth=1.5)
plt.xlabel("Time")
plt.ylabel("Re(C(t))")
plt.title("H2 Real-time evolution signals")
plt.tight_layout()
plt.legend()
plt.show()

# Placeholder for results
for idx, sig in zip(titles, signals):
    # ----------------------------
    # 1) Baseline: Hankel DMD  with rank 0+ Hankel preprocessing
    #   form pydmd for comparsion
    # ----------------------------
    dmd = HankelDMD(svd_rank=0, exact=True)
    delay_dmd = hankel_preprocessing(dmd, d=delay)
    # Sanity check:
    X = np.asarray(sig).reshape(1, -1)
    N = X.shape[1]  # number of time points in original series
    delay_dmd.fit(X)

    ref_modes = []
#    print("================================")
#    print("======= Hankel DMD RESULT ======")
#    print(f" Signal: {idx}")
#    print("================================")
#    for i, lam in enumerate(delay_dmd.eigs):
#        mu = cmath.log(lam) / dt
#        ref_modes.append(
#            {"mode": i, "lambda": lam, "decay": mu.real, "omega": mu.imag}
#        )
#        print(f"[DMD] {i}: lambda={lam}, decay={mu.real}, omega={mu.imag}")
#
#    fig, (ax_signal, ax_eigs) = plt.subplots(1, 2, figsize=(14, 5))
#    ax_signal.plot(time, sig, "-", label="Data")
#    ax_signal.plot(
#        time, delay_dmd.reconstructed_data[0].real, "--",
#        label="DMD reconstruction",
#    )
#    ax_signal.legend()
#    ax_signal.set_title(f"Hankel DMD Result - Signal {idx}")
#    ax_signal.set_xlabel("Time")
#    ax_signal.set_ylabel("Signal")
#    ax_signal.grid(True)
#
#    eigs = np.asarray(delay_dmd.eigs)
#    ax_eigs.scatter(eigs.real, eigs.imag)
#    ax_eigs.add_patch(plt.Circle((0, 0), 1, fill=False, color="black", linestyle="--"))
#    ax_eigs.axhline(0, color="black", linewidth=0.7)
#    ax_eigs.axvline(0, color="black", linewidth=0.7)
#    ax_eigs.set_aspect("equal", adjustable="datalim")
#    ax_eigs.set_title("Hankel DMD Eigenvalues")
#    ax_eigs.set_xlabel("Real")
#    ax_eigs.set_ylabel("Imaginary")
#    ax_eigs.grid(True)
#    fig.tight_layout()
#    plt.show()

    # Placeholder for DMD algorithm implementation
    postproc = odmd.ODMD(d=delay, tol=0.05, svd_rank=-1, sorted_eigs="abs", reduced=True)
    postproc.fit(sig)
    print("========== ODMD RESULT =========")
    print(f" Signal: {idx}")
    print("================================")
    for i, lam in enumerate(postproc.eigs):
        mu = cmath.log(lam) / dt
        print(f"[ODMD] {i}: decay={mu.real}, omega={mu.imag}")
        #print(f"[ODMD] {i}: lambda={lam}, decay={mu.real}, omega={mu.imag}")

    selected_eigs = np.asarray(postproc.eigs[:-1])
    fig, (ax_signal, ax_eigs) = plt.subplots(1, 2, figsize=(14, 5))

    if hasattr(postproc, "reconstructed_data"):
        ax_signal.plot(time, sig, "x-", linewidth=1, label="Input Data")
        ax_signal.plot(
            time, postproc.reconstructed_data[0].real, "x-", linewidth=1,
            label="ODMD reconstruction",
        )
        ax_signal.legend()
    ax_signal.set_title(f"ODMD Result - Signal {idx}")
    ax_signal.set_xlabel("Time")
    ax_signal.set_ylabel("Signal")
    ax_signal.grid(True)

    ax_eigs.scatter(selected_eigs.real, selected_eigs.imag)
    ax_eigs.add_patch(
        plt.Circle((0, 0), 1, fill=False, color="black", linestyle="--")
    )
    ax_eigs.axhline(0, color="black", linewidth=0.7)
    ax_eigs.axvline(0, color="black", linewidth=0.7)
    ax_eigs.set_aspect("equal", adjustable="datalim")
    ax_eigs.set_title("ODMD Eigenvalues")
    ax_eigs.set_xlabel("Real")
    ax_eigs.set_ylabel("Imaginary")
    ax_eigs.grid(True)
    fig.tight_layout()
    plt.show()