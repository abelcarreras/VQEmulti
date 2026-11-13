# ODMD validation suite (unittest)

```
VQEmulti/
├── examples_ODMD/
│   └── odmd.py                        # the standalone ODMD
└── tests_odmd/                        # this folder (any name works)
    ├── __init__.py                    # makes the folder a package
    ├── _imports.py                    # puts examples_ODMD/ and this folder on sys.path
    ├── signals.py                     # synthetic signals with known ground truth
    ├── helpers.py                     # metrics, log/report, plots, ODMDTestCase
    ├── test_01_exact.py               # sin+cos, complex exponentials, rank criteria
    ├── test_02_gaussian_noise.py      # SNR sweep, choice of d, rank selection under noise
    ├── test_03_random_signal.py       # white noise / random walk (negative controls)
    ├── test_04_damped_and_quantum.py  # decay rates, H eigenvalues, populations, Re-only
    ├── test_05_limits.py              # d too small, close energies, aliasing, bad input
    ├── test_06_forecast.py            # physical time units, forecasting unseen data
    ├── test_07_compare_pydmd.py       # cross-check vs pydmd (skipped if not installed)
    └── run_all.py                     # discovers and runs everything
```

If `odmd.py` moves, edit `EXAMPLES_DIR` in `_imports.py`; nothing else changes.

## Running

From inside `tests_odmd/`:

```bash
python run_all.py                                   # everything; prints which odmd.py is used
python -m unittest discover -v                      # same, without the odmd.py check
python -m unittest test_05_limits -v                # one file
python test_05_limits.py                            # one file, as a script
python -m unittest test_04_damped_and_quantum.TestQuantumOverlap -v              # one class
python -m unittest test_04_damped_and_quantum.TestQuantumOverlap.test_energies   # one test
python -m unittest discover -k noise -v             # tests whose name contains "noise"
python -m unittest discover -f                      # stop at first failure
```

From the project root (`VQEmulti/`):

```bash
python -m unittest discover -s tests_odmd -v
python -m unittest tests_odmd.test_04_damped_and_quantum -v
```

Options (environment variables, since unittest has no custom flags):

```bash
ODMD_PLOT=1 python run_all.py        # enable the plot tests
ODMD_VERBOSE=0 python run_all.py     # hide diagnostic numbers, keep pass/fail
```

## Reading the results

| Result | Meaning |
|---|---|
| `ok` | all assertions passed |
| `FAIL` | an assertion was false: the code ran but a number is out of tolerance |
| `ERROR` | an unexpected exception: the test itself crashed |
| `skipped '...'` | not run on purpose (plots off, pydmd missing) |

## Using ODMD from your own scripts

```python
import numpy as np
from odmd import ODMD          # odmd.py must be on sys.path (see _imports.py)

dt = 0.1
t = dt * np.arange(400)
sig = 0.7 * np.exp(1.1j * t) + 0.3 * np.exp(0.4j * t)

model = ODMD(d=100, svd_rank=0, tol=None).fit(sig, t0=t[0], dt=dt)

energies = -np.angle(model.eigs) / dt     # E_k
E0 = model.ground_phase() / dt            # lowest energy
recon = model.reconstructed_data[0]       # fitted signal

model.dmd_time = {"tend": 80.0}           # forecast to t = 80
forecast = model.reconstructed_data[0]
```

Check which file is actually imported with `import odmd; print(odmd.__file__)`.
