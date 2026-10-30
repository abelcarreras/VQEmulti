import cmath
import numpy as np
import matplotlib.pyplot as plt
import tools
import odmd

#
# General purpose script to estimate the frequencies from a correlation function using ODMD.
#

# READ DATA from files (only the real part of the signal)
#signame = "Ne_exact_signal.txt"
signame = "Ne_test_signal.txt"
time, C, ReC, ImC = tools.read_signal_file(signame)
delay = 25
dt = time[1] - time[0]

# Plot all signals
fig = plt.figure(figsize=(10, 6), dpi=200)
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
plt.plot(time, ReC, 'x-',label="Input Signal", linewidth=1.5)
plt.plot(time, ImC, 'y-',label="Input Signal", linewidth=1.5)
plt.xlabel("Time")
plt.ylabel("Re(C(t))")
plt.tight_layout()
plt.legend()
plt.show()

# ODMD postprocessing
postproc = odmd.ODMD(d=delay, tol=0.05, svd_rank=-1, sorted_eigs="abs", reduced=True)
postproc.fit(C)
print("========== ODMD RESULT =========")
print(f" Signal: {signame}")
print("================================")
for i, lam in enumerate(postproc.eigs):
    mu = cmath.log(lam) / dt
    print(f"[ODMD] {i}: decay={mu.real}, omega={mu.imag}")

selected_eigs = np.asarray(postproc.eigs[:-1])
fig, (ax_signal, ax_eigs) = plt.subplots(1, 2, figsize=(14, 5))

if hasattr(postproc, "reconstructed_data"):
    ax_signal.plot(time, ReC, "x-", linewidth=1, label="Input Data")
    ax_signal.plot(
        time, postproc.reconstructed_data[0].real, "x-", linewidth=1,
        label="ODMD reconstruction",
    )
    ax_signal.legend()

ax_signal.set_title(f"ODMD Result - Signal {signame}")
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