# ODMD validation scripts

```
odmd_tests/
├── odmd.py                      # the standalone ODMD (copy or import it, see below)
├── signals.py                   # synthetic signals with known ground truth
├── helpers.py                   # error metrics, reports, plots, mini test runner
├── test_01_exact.py             # sin+cos, complex exponentials, rank criteria
├── test_02_gaussian_noise.py    # SNR sweep, choice of d, rank selection under noise
├── test_03_random_signal.py     # white noise / random walk (negative controls)
├── test_04_damped_and_quantum.py# decay rates, H eigenvalues, populations, Re-only signal
├── test_05_limits.py            # d too small, close energies, aliasing, bad input
├── test_06_forecast.py          # physical time units, forecasting unseen data
├── test_07_compare_pydmd.py     # cross-check vs pydmd (skipped if not installed)
└── run_all.py                   # runs everything, prints which odmd.py is used
```

## Running

```bash
python run_all.py                 # everything
python run_all.py --plot          # with figures
python test_02_gaussian_noise.py  # one script
pytest -q                         # also works if pytest is installed
```

## Using ODMD

```python
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from odmd import ODMD

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

`from odmd import ODMD` only works if Python can find `odmd.py`. Options, from
quickest to cleanest:

1. **Same folder**: keep `odmd.py` at the same place as the tests
2. **Path at runtime**:
   ```python
   import sys
   sys.path.insert(0, "path/to/folder/with/odmd")
   from odmd import ODMD
   ```

Whichever you use, check which file is actually imported:

```python
import odmd; print(odmd.__file__)
```
