import numpy as np
import matplotlib.pyplot as plt

def read_signal_file(filename):
    """
    Reads a text file with columns:
    dt, Re(C(t)), Im(C(t)), Re(F(t)), Im(F(t))

    Returns:
        dt        : numpy array
        C         : complex numpy array (C = Re + i Im)
        F         : complex numpy array (F = Re + i Im)
        ReC, ImC  : numpy arrays
        ReF, ImF  : numpy arrays
    """
    data = np.loadtxt(filename, skiprows=2)

    dt  = data[:, 0]
    ReC = data[:, 1]
    ImC = data[:, 2]

    C = ReC + 1j * ImC
    return dt, C, ReC, ImC

def plot_selected_eigs(
    eigs,
    show_axes=True,
    show_unit_circle=True,
    figsize=(8, 8),
    title="",
    dpi=None,
    filename=None,
):
    """
    Plot the eigenvalues.

    :param eigs: numpy array with eigenvalues from DMD calculation.
    :param bool show_axes: if True, the axes will be showed in the plot.
        Default is True.
    :param bool show_unit_circle: if True, the circle with unitary radius
        and center in the origin will be showed. Default is True.
    :param tuple(int,int) figsize: tuple in inches defining the figure
        size. Default is (8, 8).
    :param str title: title of the plot.
    :param dpi int: If not None, the given value is passed to
        ``plt.figure``.
    :param str filename: if specified, the plot is saved at `filename`.
    """

    if eigs is None:
        raise ValueError(
            "The eigenvalues have not been computed."
            "You have to call the fit() method."
        )

    if dpi is not None:
        plt.figure(figsize=figsize, dpi=dpi)
    else:
        plt.figure(figsize=figsize)

    plt.title(title)
    plt.gcf()
    ax = plt.gca()

    points = ax.plot(eigs.real, eigs.imag, "bo", label="Eigenvalues")


    # set limits for axis
    limit = np.max(np.ceil(np.absolute(eigs)))

    ax.set_xlim((-limit, limit))
    ax.set_ylim((-limit, limit))

    # x and y axes
    if show_axes:
        ax.annotate(
            "",
            xy=(np.max([limit * 0.8, 1.0]), 0.0),
            xytext=(np.min([-limit * 0.8, -1.0]), 0.0),
            arrowprops=dict(arrowstyle="->"),
        )
        ax.annotate(
            "",
            xy=(0.0, np.max([limit * 0.8, 1.0])),
            xytext=(0.0, np.min([-limit * 0.8, -1.0])),
            arrowprops=dict(arrowstyle="->"),
        )

    plt.ylabel("Imaginary part")
    plt.xlabel("Real part")

    if show_unit_circle:
        unit_circle = plt.Circle(
            (0.0, 0.0),
            1.0,
            color="green",
            fill=False,
            label="Unit circle",
            linestyle="--",
        )
        ax.add_artist(unit_circle)

    # Dashed grid
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    for line in gridlines:
        line.set_linestyle("-.")
    ax.grid(True)

    # legend
    if show_unit_circle:
        ax.add_artist(
            plt.legend(
                [points, unit_circle],
                ["Eigenvalues", "Unit circle"],
                loc="best",
            )
        )
    else:
        ax.add_artist(plt.legend([points], ["Eigenvalues"], loc="best"))

    ax.set_aspect("equal")

    if filename:
        plt.savefig(filename)
    else:
        plt.show()
