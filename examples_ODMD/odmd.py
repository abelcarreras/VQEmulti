import warnings
from pydmd.dmdbase import DMDBase
from pydmd.dmdoperator import DMDOperator
from pydmd.snapshots import Snapshots

warnings.filterwarnings(
    "ignore",
    message=r"Legend does not support handles for list instances\.",
    category=UserWarning,
)

class ODMDOperator(DMDOperator):
    """
    Operator for the overlap-based ODMD.

    This builds the companion-like A operator from a 1D overlap data sequence.
    """

    def __init__(
            self,
            d,
            tol=0.1,
            svd_rank=-1,
            exact=True,
            forward_backward=False,
            rescale_mode=None,
            sorted_eigs=False,
            tikhonov_regularization=None,
    ):
        super().__init__(
            svd_rank,
            exact,
            forward_backward,
            rescale_mode,
            sorted_eigs,
            tikhonov_regularization,
        )
        self.d = int(d)
        self.tol = float(tol)

    def compute_odmd_operator(self, overlaps):
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
        svals : np.ndarray
            Singular values of the RHS block.
        """
        overlaps = np.asarray(overlaps, dtype=complex).ravel()

        if overlaps.size == 0:
            raise ValueError("overlaps cannot be empty")

        if overlaps.size <= self.d:
            raise ValueError(
                f"Need len(overlaps) > d. Got len(overlaps)={overlaps.size}, d={self.d}"
            )

        num_cols = overlaps.size - self.d
        lhs = np.zeros((self.d, num_cols), dtype=complex)
        rhs = np.zeros((self.d, num_cols), dtype=complex)

        for col in range(num_cols):
            lhs[:, col] = overlaps[col + 1: col + 1 + self.d]
            rhs[:, col] = overlaps[col: col + self.d]

        # SVD truncation by relative cutoff
        U, s, Vh = np.linalg.svd(rhs, full_matrices=False)

        if s.size == 0:
            raise ValueError("SVD failed: RHS has no singular values.")

        keep = np.where(s > self.tol * s[0])[0]
        if keep.size == 0:
            raise ValueError("All singular values were truncated; reduce tol.")

        r = keep[-1] + 1
        U_r = U[:, :r]
        s_r = s[:r]
        Vh_r = Vh[:r, :]

        # ODMD Atilde
        Atilde = lhs @ Vh_r.conj().T @ np.diag(1.0 / s_r) @ U_r.conj().T

        eigs, modes = np.linalg.eig(Atilde)

        # Store everything in the operator object so PyDMD properties can see it
        self._Atilde = Atilde
        self._eigenvalues = eigs
        self._eigenvectors = modes
        self._modes = modes
        self._Lambda = eigs

        return modes, eigs, s

    def compute_operator(self, X, Y=None):
        """
        Compatibility wrapper with the standard DMDOperator signature.
        The second argument is ignored for ODMD.
        """
        return self.compute_odmd_operator(X)

class ODMD(DMDBase):
    """
    ODMD outer API built on DMDBase.
    Class that implements ODMD method, the implementation and usage are
    slightly different from those implemented in standard DMD methods inside pyDMD


    New parameters:
    :param d: data delay dimension (used to build Hankel matrices), ODMD constructs
              Hankel matrices based on the Givens theorem.
    :type int
    :param tol: corresponds to the tolerance parameter introduced instead of rank truncation.
    :type float, default 0.1, use with svd_rank = -1 to avoid conflicts with pyDMD

    Standard pyDMD parameters:
    :param svd_rank: the rank for the truncation; If 0, the method computes the
    optimal rank and uses it for truncation; if positive interger, the
    method uses the argument for the truncation; if float between 0 and 1,
    the rank is the number of the biggest singular values that are needed
    to reach the 'energy' specified by `svd_rank`; if -1, the method does
    not compute truncation.
    :type svd_rank: int or float
    :param int tlsq_rank: rank truncation computing Total Least Square. Default
        is 0, that means TLSQ is not applied.
    :param bool exact: flag to compute either exact DMD or projected DMD.
        Default is False.
    :param opt: argument to control the computation of DMD modes amplitudes.
        See :class:`DMDBase`. Default is False.
    :type opt: bool or int
    :param rescale_mode: Scale Atilde as shown in
            10.1016/j.jneumeth.2015.10.010 (section 2.4) before computing its
            eigendecomposition. None means no rescaling, 'auto' means automatic
            rescaling using singular values, otherwise the scaling factors.
    :type rescale_mode: {'auto'} or None or numpy.ndarray
    :param bool forward_backward: If True, the low-rank operator is computed
        like in fbDMD (reference: https://arxiv.org/abs/1507.02264). Default is
        False.
    :param sorted_eigs: Sort eigenvalues (and modes/dynamics accordingly) by
        magnitude if `sorted_eigs='abs'`, by real part (and then by imaginary
        part to break ties) if `sorted_eigs='real'`. Default: False.
    :type sorted_eigs: {'real', 'abs'} or False
    :param tikhonov_regularization: Tikhonov parameter for the regularization.
        If `None`, no regularization is applied, if `float`, it is used as the
        lambda tikhonov parameter.
    :type tikhonov_regularization: int or float

    usage: odmd = ODMD(d=10, tol=0.1, exact=True, svd_rank=-1)
           odmd.fit(sig.reshape(1, -1))

    This allows the user to make use of pyDMD machinery.
    """

    def __init__(
        self,
        d=10,
        tol=0.1,
        svd_rank=0,
        tlsq_rank=0,
        exact=False,
        opt=False,
        rescale_mode=None,
        forward_backward=False,
        sorted_eigs=False,
        tikhonov_regularization=None,
    ):
        super().__init__(
            svd_rank=svd_rank,
            tlsq_rank=tlsq_rank,
            exact=exact,
            opt=opt,
            rescale_mode=rescale_mode,
            forward_backward=forward_backward,
            sorted_eigs=sorted_eigs,
        )
        self._sorted_eigs = sorted_eigs
        self._forward_backward = forward_backward
        self._rescale_mode = rescale_mode
        self._svd_rank = svd_rank
        self.d = int(d)
        self.tol = float(tol)
        self._tikhonov_regularization = tikhonov_regularization
        self._odmd_operator = None

    @property
    def operator(self):
        """
        Make DMDBase.eigs and other base-class machinery use the fitted ODMD operator.
        """
        return self._odmd_operator

    def fit(self, X):
        """
        Fit ODMD to a 1D overlap sequence or a single-row snapshot array.

        Parameters
        ----------
        X : array-like
            Shape (n_samples,) or (1, n_samples).
        """
        self._reset()
        self._snapshots_holder = Snapshots(X)

        # Snapshots gives a 2D array; ODMD is built on a single sequence.
        overlaps = np.asarray(self.snapshots, dtype=complex).ravel()

        if overlaps.size == 0:
            raise ValueError("Input data cannot be empty")

        # Build and fit the custom operator
        self._odmd_operator = ODMDOperator(
            d=self.d,
            tol=self.tol,
            svd_rank=self._svd_rank,
            exact=self._exact,
            forward_backward=self._forward_backward,
            rescale_mode=self._rescale_mode,
            sorted_eigs=self._sorted_eigs,
            tikhonov_regularization=self._tikhonov_regularization,
        )

        print("overlaps before compute_operator:",
              np.asarray(overlaps).shape)

        print("overlaps dimensions:",
              np.asarray(overlaps).ndim)

        modes, eigs, _ = self._odmd_operator.compute_operator(overlaps, None)

        # Mirror fitted state into the DMDBase object
        self._Atilde = self._odmd_operator
        self._modes = modes
        self._eigs = eigs

        # Standard PyDMD time bookkeeping
        self._set_initial_time_dictionary(
            {"t0": 0, "tend": overlaps.size - 1, "dt": 1}
        )

        # Initial amplitude fit in the reduced ODMD space
        x0 = overlaps[: self._modes.shape[0]].reshape(-1)
        self._b = np.linalg.lstsq(self._modes, x0, rcond=None)[0]

        return self

    def ground_phase(self):
        """
        Return the negative phase of the eigenvalue with the largest phase.
        """
        eigs = np.asarray(self.eigs)
        idx = np.argmax(np.angle(eigs))
        return -np.angle(eigs[idx])


# TEST
if __name__ == "__main__":
    import cmath
    import numpy as np
    import matplotlib.pyplot as plt

    import tools
    from pydmd import DMD
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


    results = {"standard_dmd": [], "odmd": []}

    for idx, sig in enumerate(signals):
        print(f"\nSignal {idx}")

        # ----------------------------
        # 1) Baseline: standard DMD + Hankel preprocessing
        # ----------------------------
        dmd = DMD(svd_rank=8, exact=True)
        delay_dmd = hankel_preprocessing(dmd, d=delay)
        delay_dmd.fit(sig.reshape(1, -1))

        ref_modes = []
        print("========== DMD RESULT =========")
        for i, lam in enumerate(delay_dmd.eigs):
            mu = cmath.log(lam) / dt
            ref_modes.append(
                {"mode": i, "lambda": lam, "decay": mu.real, "omega": mu.imag}
            )
            print(f"[DMD] {i}: lambda={lam}, decay={mu.real}, omega={mu.imag}")

        results["standard_dmd"].append(ref_modes)

        plt.figure(figsize=(8, 4))
        plt.plot(time, sig, "-", label="Data")
        plt.plot(time, delay_dmd.reconstructed_data[0].real, "--", label="DMD reconstruction")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plot_eigs(delay_dmd, show_axes=True, show_unit_circle=True, figsize=(8, 8))
        plt.show()

        # ----------------------------
        # 2) ODMD
        # ----------------------------
        odmd = ODMD(d=delay, tol=0.01, exact=True, svd_rank=-1)
        odmd.fit(sig.reshape(1, -1))
        #print("eigs:", odmd.eigs)
        #print("modes shape:", odmd.modes.shape)

        odmd_modes = []
        print("========== ODMD RESULT =========")
        for i, lam in enumerate(odmd.eigs[:6]):
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
        selected_eigs = odmd.eigs[:6]
        tools.plot_selected_eigs(selected_eigs, show_axes=True, show_unit_circle=True, figsize=(8, 8))
        plt.show()

