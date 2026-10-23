"""
Test 7 (optional): cross-check against pydmd's Hankel DMD.

Skipped automatically if pydmd is not installed. Run it in the environment
where pydmd is available to confirm the standalone version reproduces it:

    ODMD(d, svd_rank=r, tol=None)  ==  hankel_preprocessing(DMD(svd_rank=r, exact=True), d=d)

    python test_07_compare_pydmd.py
"""
import sys
import numpy as np

from odmd import ODMD
import signals
from helpers import run_tests, SkipTest


def _pydmd_eigs(sig, d, svd_rank):
    try:
        from pydmd import DMD
        from pydmd.preprocessing import hankel_preprocessing
    except ImportError:
        raise SkipTest("pydmd not installed")
    model = hankel_preprocessing(DMD(svd_rank=svd_rank, exact=True), d=d)
    model.fit(np.asarray(sig).reshape(1, -1))
    return np.asarray(model.eigs)


def _compare(sig, d, svd_rank, label):
    ref = np.sort_complex(_pydmd_eigs(sig, d, svd_rank))
    mine = np.sort_complex(ODMD(d=d, svd_rank=svd_rank, tol=None,sorted_eigs="abs", reduced=True).fit(sig).eigs)
    print(f"    {label}: pydmd {ref.size} eigs, ODMD {mine.size} eigs")
    assert ref.size == mine.size, "different ranks selected"
    diff = np.max(np.abs(ref - mine))
    print(f"        max |lambda_pydmd - lambda_odmd| = {diff:.1e}")
    assert diff < 1e-6


def test_match_pydmd_real_noisy():
    _, sig, _ = signals.sin_cos()
    sig = signals.add_noise(sig, 10, seed=0)
    for r in (0, 2, 4, 6, 8): #Test PASS within these truncations, for this example
        _compare(sig, d=25, svd_rank=r, label=f"sin+cos @40dB, svd_rank={r}")


def test_match_pydmd_complex():
    _, sig, _ = signals.quantum_overlap()
    sig = signals.add_noise(sig, 40, seed=0)
    for d in (25, 100):
        _compare(sig, d=d, svd_rank=0, label=f"quantum overlap @40dB, d={d}")


if __name__ == "__main__":
    ok = run_tests(globals())
    sys.exit(0 if ok else 1)
