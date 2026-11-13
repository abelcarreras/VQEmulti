import warnings
import numpy as np
warnings.filterwarnings("ignore")
#--------------------------------------
# Utilities
#--------------------------------------

def _as_signal(X):
    """
    Accept shape (n,), (1, n) or (n, 1) and return a 1D complex array
    """
    X = np.asarray(X)
    if X.ndim == 2 and 1 not in X.shape:
        raise ValueError(
            f"ODMD expects a single sequence, got 2D data of shape {X.shape}"
        )
    if X.ndim > 2:
        raise ValueError(f"ODMD expects 1D or single-row data, got ndim={X.ndim}")
    sig = X.astype(complex).ravel()
    if sig.size == 0:
        raise ValueError("Input data cannot be empty")
    if not np.all(np.isfinite(sig)):
        raise ValueError("Input data contains NaN or infinity")
    return sig

def hankel_pair(signal, d):
    """
    Delay-embed a 1D signal into shifted Hankel blocks

    Returns
    -------
    X : (d, n - d) array, columns signal[j : j + d] (earlier block, "rhs")
    Y : (d, n - d) array, columns signal[j + 1 : j + 1 + d] (shifted block, "lhs")
    """
    signal = np.asarray(signal).ravel()
    n = signal.size
    if n <= d:
        raise ValueError(f"Need len(signal) > d. Got len(signal)={n}, d={d}")
    num_cols = n - d
    idx = np.arange(d)[:, None] + np.arange(num_cols)[None, :]
    return signal[idx], signal[idx + 1]


def compute_rank(s, shape, svd_rank=-1):
    """
    Rank selection with the same conventions as pydmd.utils.compute_rank.

    svd_rank = -1      : no truncation
    svd_rank = 0       : optimal hard threshold (Gavish & Donoho 2014)
    0 < svd_rank < 1   : smallest rank reaching that fraction of energy
    int svd_rank >= 1  : fixed rank (capped at the number of singular values)
    """
    if svd_rank == 0:
        beta = min(shape) / max(shape)
        omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
        rank = int(np.sum(s > np.median(s) * omega))
        if rank == 0:
            warnings.warn("Optimal rank is 0; using rank 1 instead.")
            rank = 1
    elif 0 < svd_rank < 1:
        energy = np.cumsum(s**2) / np.sum(s**2)
        rank = int(np.searchsorted(energy, svd_rank) + 1)
    elif svd_rank >= 1 and float(svd_rank).is_integer():
        rank = int(svd_rank)
    elif svd_rank == -1:
        rank = s.size
    else:
        raise ValueError(f"Invalid svd_rank: {svd_rank}")
    return min(rank, s.size)

def truncated_svd(X, svd_rank=-1, tol=None):
    """
    SVD of X truncated by svd_rank and, optionally, by a relative cutoff
    s_i > tol * s_0. When both are given the smaller rank wins
    """
    U, s, Vh = np.linalg.svd(X, full_matrices=False)
    if s.size == 0 or s[0] == 0:
        raise ValueError("SVD failed: matrix has no non-zero singular values.")

    rank = compute_rank(s, X.shape, svd_rank)
    if tol is not None:
        keep = np.flatnonzero(s > tol * s[0])
        if keep.size == 0:
            raise ValueError("All singular values were truncated; reduce tol.")
        rank = min(rank, keep[-1] + 1)

    return U[:, :rank], s[:rank], Vh[:rank, :], s

def _sort_eigs(eigs, vecs, how):
    if not how:
        return eigs, vecs
    if how == "abs":
        order = np.argsort(np.abs(eigs))[::-1]
    elif how == "real":
        order = np.lexsort((eigs.imag, eigs.real))[::-1]
    elif how == "phase":
        order = np.argsort(np.angle(eigs))[::-1]
    else:
        raise ValueError(f"Invalid how variable at sorted_eigs: {how}")
    return eigs[order], vecs[:, order]

#--------------------------------------
# OPERATOR CLASS
#--------------------------------------
class ODMDOperator:
    """
    Builds the ODMD operator A ~ Y X^+ from a 1D overlap sequence.

    Parameters
    ----------
    d : int
        Delay (embedding) dimension.
    tol : float or None
        Relative singular-value cutoff s_i > tol * s_0.
    svd_rank : int or float
        Optional extra truncation, pydmd conventions (-1 = none).
    sorted_eigs : {False, 'abs', 'real', 'phase'}
    reduced : bool
        False (default): eigendecompose the full d x d operator, exactly
        (d - r eigenvalues are ~0 by construction).
        True: eigendecompose the r x r projected operator U_r^H A U_r instead.
        Its eigenvalues are the r non-zero eigenvalues of the full operator,
        and the modes are the exact DMD modes Y V S^-1 W.
    """

    def __init__(
            self,
            d,
            tol=0.1,
            svd_rank=-1,
            sorted_eigs=False,
            reduced=False
    ):
        self.d = int(d)
        self.tol = None if tol is None else float(tol)
        self.svd_rank = svd_rank
        self.sorted_eigs = sorted_eigs
        self.reduced = bool(reduced)

        self._Atilde = None
        self._eigenvalues = None
        self._eigenvectors = None
        self._modes = None
        self._svd_vals = None
        self._rank = None


    def compute_operator(self, overlaps):
        """
        Build the ODMD operator from a 1D overlap sequence.
        Parameters
        ----------
        overlaps : array-like, shape (n_samples,)
            Sequence of overlap values.
        Returns
        -------
        modes : np.ndarray
        eigs : np.ndarray
        s_all : np.ndarray Singular values of the RHS block.
        """
        overlaps = _as_signal(overlaps)
        X, Y = hankel_pair(overlaps, self.d)  # X = rhs, Y = lhs

        U_r, s_r, Vh_r, s_all = truncated_svd(X, self.svd_rank, self.tol)
        YVSinv = Y @ Vh_r.conj().T @ np.diag(1.0 / s_r)

        if self.reduced:
            Atilde = U_r.conj().T @ YVSinv  # (r, r)
            eigs, W = np.linalg.eig(Atilde)
            eigs, W = _sort_eigs(eigs, W, self.sorted_eigs)
            modes = YVSinv @ W
            eigenvectors = W
        else:
            Atilde = YVSinv @ U_r.conj().T  # (d, d)
            eigs, W = np.linalg.eig(Atilde)
            eigs, W = _sort_eigs(eigs, W, self.sorted_eigs)
            modes = W
            eigenvectors = W

        self._Atilde = Atilde
        self._eigenvalues = eigs
        self._eigenvectors = eigenvectors
        self._modes = modes
        self._svd_vals = s_all
        self._rank = s_r.size
        return modes, eigs, s_all

    # accessors
    @property
    def as_numpy_array(self):
        return self._Atilde

    @property
    def shape(self):
        return None if self._Atilde is None else self._Atilde.shape

    @property
    def eigenvalues(self):
        return self._eigenvalues

    @property
    def eigenvectors(self):
        return self._eigenvectors

    @property
    def modes(self):
        return self._modes

    @property
    def svd_vals(self):
        return self._svd_vals

    @property
    def rank(self):
        return self._rank

#--------------------------------------
# ODMD CLASS (Outer API)
#--------------------------------------
class ODMD:
    """
    ODMD outer API
    Class that implements ODMD method, the implementation and usage are
    slightly different from those implemented in standard DMD methods inside pyDMD

    Usage
    -----
       odmd = ODMD(d=25, tol=0.01)  (recommended for some data with 50 total snapshots)
       odmd.fit(sig)     # sig shape (n,) or (1,n)
       odmd.eigs, odmd.modes, odmd.reconstructed_data[0]

    Parameters
    -----
    d : int
    tol : float
    svd_rank : int of float (see compute_rank function)
    sorted_eigs : bool
    reduced : bool (see ODMDOperator)
    """

    def __init__(self,
                 d=10,
                 tol=0.1,
                 svd_rank=-1,
                 sorted_eigs=False,
                 reduced=False):

        self.d = int(d)
        self.tol = tol
        self.svd_rank = svd_rank
        self.sorted_eigs = sorted_eigs
        self.reduced = reduced
        self._reset()

    def _reset(self):
        self._operator = None
        self._overlaps = None
        self._b = None
        self._original_time = None
        self._dmd_time = None

    #- - - - Fitting - - -
    def fit(self,X, t0=0, dt=1):
        """
        Fit ODMD to a 1D overlap sequence.

        X  : shape (n,) or (1, n)
        t0, dt : time origin and step used for dynamics/reconstruction.
                 Default assume t0 = 0, dt = 1, i.e. step index.
        """
        self._reset()
        overlaps = _as_signal(X)
        self._overlaps = overlaps

        self._operator = ODMDOperator(
            d = self.d,
            tol = self.tol,
            svd_rank = self.svd_rank,
            sorted_eigs = self.sorted_eigs,
            reduced = self.reduced,
        )
        modes, _, _ = self._operator.compute_operator(overlaps)

        tend = t0 + (overlaps.size -1) * dt
        self._original_time = {"t0" : t0, "tend": tend, "dt": dt}
        self._dmd_time = dict(self._original_time)

        # Amplitudes from the firs Hankel column
        x0 = overlaps[: self.d]
        self._b = np.linalg.lstsq(modes, x0, rcond=None)[0]
        return  self

    def _check_fitted(self):
        if self._operator is None:
            raise RuntimeError("Call fit() before using ODMD results.")

    # properties
    @property
    def operator(self):
        return self._operator

    @property
    def snapshots(self):
        """The fitted sequence as a (1,n) array"""
        return None if self._overlaps is None else self._overlaps[None, :]

    @property
    def eigs(self):
        self._check_fitted()
        return self._operator.eigenvalues

    @property
    def modes(self):
        self._check_fitted()
        return self._operator.modes

    @property
    def amplitudes(self):
        self._check_fitted()
        return self._b

    @property
    def singular_values(self):
        self._check_fitted()
        return self._operator._svd_vals

    @property
    def original_time(self):
        return self._original_time

    @property
    def dmd_time(self):
        return self._dmd_time

    @dmd_time.setter
    def dmd_time(self, value):
        """Set e.g {'t0':0, 'tend' : 500, 'dt' : 1} to forecast/intrapolate"""
        self._check_fitted()
        new = dict(self._dmd_time)
        new.update(value)
        self._dmd_time = new

    @property
    def original_timesteps(self):
        t = self._original_time
        return t["t0"] + t["dt"] * np.arange(self._overlaps.size)

    @property
    def dmd_timesteps(self):
        t = self._dmd_time
        n = int(np.floor((t["tend"] - t["t0"]) / t["dt"] + 1e-9)) + 1
        return t["t0"] + t["dt"] * np.arange(n)

    @property
    def dynamics(self):
        """b_k * lambda_k^((t - t0_orig) / dt_orig), shape (n_modes, n_times)."""
        self._check_fitted()
        steps = (self.dmd_timesteps - self._original_time["t0"]) / self._original_time["dt"]
        return self.amplitudes[:, None] * np.power(self.eigs[:, None], steps[None, :])

    @property
    def reconstructed_data(self):
        """modes @ dynamics, shape (d, n_times); row 0 is the signal itself."""
        return self.modes @ self.dynamics

    @property
    def frequency(self):
        """Cycles per unit time, same definition as pydmd."""
        return np.log(self.eigs).imag / (2 * np.pi * self._original_time["dt"])

    @property
    def growth_rate(self):
        return np.log(self.eigs).real / self._original_time["dt"]

    # ---- ODMD-specific -------------
    def ground_phase(self):
        """Negative phase of the eigenvalue with the largest phase."""
        eigs = np.asarray(self.eigs)
        idx = np.argmax(np.angle(eigs))
        return -np.angle(eigs[idx])


#------------------------------------
# TEST
#------------------------------------
if __name__ == "__main__":
    import cmath
    import numpy as np
    import matplotlib.pyplot as plt
    import tools
    from pydmd import HankelDMD
    from pydmd.preprocessing import hankel_preprocessing
    from pydmd.plotter import plot_eigs

    # Read data:
    time, C, ReC, ImC = tools.read_signal_file(
        "H2_exact_evol.txt"
    )

    C1 = np.asarray(ReC)  # , dtype=complex).ravel()
    C2 = np.asarray(ImC)  # , dtype=complex).ravel()
    print("DATA C1 SHAPE ", C1.shape)
    print("DATA C2 SHAPE ", C1.shape)

    signals = [C1]
    delay = 25
    dt = time[1] - time[0]

    # Plot signal to compare:
    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.plot(time, ReC, 'k-', label='Re Original', linewidth=1.0)
    plt.plot(time, ImC, 'r-', label='Im Original', linewidth=1.0)
    plt.xlabel("t")
    plt.ylabel("C(t)")
    plt.title("The original signal")
    plt.tight_layout()
    plt.legend()
    plt.show()


    results = {"hankel_dmd": [], "odmd": []}

    for idx, sig in enumerate(signals):
        print(f"\nSignal {idx}")

        # ----------------------------
        # 1) Baseline: Hankel DMD  with rank 0+ Hankel preprocessing
        #   form pydmd for comparsion
        # ----------------------------
        dmd = HankelDMD(svd_rank=0, exact=True)
        delay_dmd = hankel_preprocessing(dmd, d=delay)
        # Sanity check:
        X = np.asarray(sig).reshape(1,-1)
        N = X.shape[1]  # number of time points in original series
        delay_dmd.fit(X)

        ref_modes = []
        print("========== Hankel DMD RESULT =========")
        for i, lam in enumerate(delay_dmd.eigs):
            mu = cmath.log(lam) / dt
            ref_modes.append(
                {"mode": i, "lambda": lam, "decay": mu.real, "omega": mu.imag}
            )
            print(f"[DMD] {i}: lambda={lam}, decay={mu.real}, omega={mu.imag}")

        results["hankel_dmd"].append(ref_modes)

        plt.figure(figsize=(8, 4))
        plt.plot(time, sig, "-", label="Data")
        plt.plot(time, delay_dmd.reconstructed_data[0].real, "--", label="DMD reconstruction")
        plt.legend()
        plt.title(f"Hankel DMD Result - Signal {idx}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plot_eigs(delay_dmd, show_axes=True, show_unit_circle=True, figsize=(8, 8))
        plt.show()

        # ----------------------------
        # 2) ODMD
        # ----------------------------
        odmd = ODMD(d=delay, tol=None, svd_rank=0, sorted_eigs="abs", reduced=True)
        odmd.fit(sig)

        odmd_modes = []
        print("========== ODMD RESULT =========")
        #print("Ground Phase :", odmd.ground_phase())
        for i, lam in enumerate(odmd.eigs):
            mu = cmath.log(lam) / dt
            odmd_modes.append(
                {"mode": i, "lambda": lam, "decay": mu.real, "omega": mu.imag}
            )
            print(f"[ODMD] {i}: lambda={lam}, decay={mu.real}, omega={mu.imag}")

        results["odmd"].append(odmd_modes)

        if hasattr(odmd, "reconstructed_data"):
            plt.figure(figsize=(8, 4))
            plt.plot(time, sig, "-", label="Data")
            plt.plot(time, odmd.reconstructed_data[0].real, "--", label="ODMD reconstruction")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        # ODMD eigenvalues to plot
        selected_eigs = odmd.eigs[:10]
        tools.plot_selected_eigs(selected_eigs, show_axes=True, show_unit_circle=True, figsize=(8, 8))
        plt.show()

