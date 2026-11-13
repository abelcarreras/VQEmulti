"""
A priori diagnostics for sampling C(t) = <psi0| exp(-iHt) |psi0>.

Works for any qubit Hamiltonian (molecules, spin and Hubbard models) given as an
OpenFermion QubitOperator or a plain dict {((q, 'X'), (r, 'Z'), ...): coeff},
and any computational-basis reference (HF, Neel, ...), given as a 0/1 list
indexed by qubit.

Pipeline
  1. Pauli table      : identity coefficient, L1, L2, term count, max weight
  2. Krylov from psi0 : k Lanczos steps on a sparse bitstring dictionary
                        -> E_ref, sigma, Ritz values + weights (approximate
                        spectral measure of psi0), conserved Z-type charges
  3. Populated window : [lo, hi] from Ritz values +/- residuals, clipped to
                        the rigorous L1 bound (exact if the Krylov space closes)
  4. Time step        : largest alias-free dt for the signal type
                        (complex / real) and shift strategy
  5. Alias check      : where each Ritz frequency lands for a given dt
  6. Trotter          : first-order sub-steps per sample for a target bias

Energies refer to the operator that is actually evolved: if the constant was
removed from H, they are energies of H - c0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def _popcount(x: int) -> int:
    return x.bit_count()


# =============================================================================
# 1. Pauli table
# =============================================================================
class PauliTable:
    """
    Symplectic representation: for each string, flip mask (X or Y) and
    phase mask (Z or Y), so that  P|x> = i^nY (-1)^popcount(x & zy) |x ^ flip>.
    """

    def __init__(self, hamiltonian, n_qubits: int | None = None):
        terms = getattr(hamiltonian, "terms", hamiltonian)
        self.identity = 0.0
        flips, zys, coeffs, weights = [], [], [], []
        nq = 0
        for term, c in terms.items():
            c = complex(c)
            if abs(c.imag) > 1e-10:
                raise ValueError("Complex Pauli coefficient: H is not Hermitian.")
            if len(term) == 0:
                self.identity += c.real
                continue
            f = zy = ny = 0
            for q, p in term:
                nq = max(nq, q + 1)
                if p == "X":
                    f |= 1 << q
                elif p == "Y":
                    f |= 1 << q
                    zy |= 1 << q
                    ny += 1
                elif p == "Z":
                    zy |= 1 << q
                else:
                    raise ValueError(f"Unknown Pauli {p!r}")
            flips.append(f)
            zys.append(zy)
            coeffs.append(c.real * (1j ** ny))
            weights.append(len(term))

        self.n_qubits = max(nq, n_qubits or 0)
        self.flip = flips
        self.zy = zys
        self.coeff = np.array([abs(c) for c in coeffs])  # |c_j|
        self.max_weight = max(weights, default=0)

        # group by flip mask: H = sum_m D_m X^m with D_m diagonal
        groups: dict[int, list[tuple[int, complex]]] = {}
        for f, zy, c in zip(flips, zys, coeffs):
            groups.setdefault(f, []).append((zy, c))
        self.groups = groups

    @property
    def n_terms(self):
        return len(self.flip)

    @property
    def l1(self):
        return float(self.coeff.sum())

    @property
    def l2(self):
        return float(np.sqrt((self.coeff ** 2).sum()))

    def apply(self, state: dict[int, complex], drop: float = 1e-14) -> dict[int, complex]:
        out: dict[int, complex] = {}
        identity = self.identity
        for x, a in state.items():
            if identity:
                out[x] = out.get(x, 0.0) + identity * a
            for f, lst in self.groups.items():
                amp = 0.0
                for zy, c in lst:
                    amp += -c if _popcount(x & zy) & 1 else c
                if amp != 0.0:
                    y = x ^ f
                    out[y] = out.get(y, 0.0) + a * amp
        return {k: v for k, v in out.items() if abs(v) > drop}


def bits_to_int(bits) -> int:
    return sum(1 << q for q, b in enumerate(bits) if b)


# =============================================================================
# 2. Lanczos from a basis state (sparse dictionaries)
# =============================================================================
def _dot(u, v):
    if len(u) > len(v):
        u, v = v, u
    return sum(np.conj(a) * v[x] for x, a in u.items() if x in v)


def _axpy(w, c, v):
    for x, a in v.items():
        w[x] = w.get(x, 0.0) + c * a


@dataclass
class KrylovResult:
    alphas: list
    betas: list
    ritz: np.ndarray
    weights: np.ndarray
    residuals: np.ndarray
    closed: bool              # Krylov space invariant -> Ritz values exact
    truncated: bool           # stopped by max_states
    n_states: int
    charges: dict = field(default_factory=dict)


def lanczos_from_reference(table: PauliTable, x0: int, steps: int = 8,
                           max_states: int = 200_000, tol: float = 1e-10):
    V = [{x0: 1.0 + 0j}]
    alphas, betas = [], []
    closed = truncated = False
    seen_n, seen_sz = set(), set()
    even = sum(1 << q for q in range(0, table.n_qubits, 2))
    odd = sum(1 << q for q in range(1, table.n_qubits, 2))
    n_states = 1

    for j in range(steps):
        w = table.apply(V[j])
        for x in w:
            seen_n.add(_popcount(x))
            seen_sz.add(_popcount(x & even) - _popcount(x & odd))
        n_states = max(n_states, len(w))
        alphas.append(float(np.real(_dot(V[j], w))))
        for _ in range(2):                       # full reorthogonalisation
            for v in V:
                _axpy(w, -_dot(v, w), v)
        b = math.sqrt(sum(abs(a) ** 2 for a in w.values()))
        betas.append(b)
        if b < tol:
            closed = True
            break
        if len(w) > max_states:
            truncated = True
            break
        if j < steps - 1:
            V.append({x: a / b for x, a in w.items()})

    k = len(alphas)
    T = np.diag(alphas) + np.diag(betas[:k - 1], 1) + np.diag(betas[:k - 1], -1)
    theta, S = np.linalg.eigh(T)
    weights = S[0, :] ** 2
    residuals = np.zeros(k) if closed else betas[k - 1] * np.abs(S[-1, :])
    charges = {
        "N conserved (Z-magnetisation for spin models)": len(seen_n) == 1,
        "Sz conserved (interleaved a/b)": len(seen_sz) == 1,
    }
    return KrylovResult(alphas, betas, theta, weights, residuals,
                        closed, truncated, n_states, charges)


# =============================================================================
# 3-5. Window, step size, alias check
# =============================================================================
def populated_window(table: PauliTable, kr: KrylovResult, mode: str = "lanczos"):
    """[lo, hi] assumed to contain every eigenvalue with weight in psi0."""
    rig = (table.identity - table.l1, table.identity + table.l1)
    if mode == "rigorous":
        return rig
    lo = float(np.min(kr.ritz - kr.residuals))
    hi = float(np.max(kr.ritz + kr.residuals))
    return max(lo, rig[0]), min(hi, rig[1])


def recommend_step(lo: float, hi: float, signal: str = "complex",
                   shift: str = "classical", safety: float = 0.9):
    """
    complex signal : frequencies E - Es in (-pi/dt, pi/dt).
                     shift 'classical' or 'circuit' centres the window.
    real signal    : only |E - Es| is observable, in [0, pi/dt].
                     shift 'circuit' puts Es below the window (sign-safe);
                     shift 'none' requires all populated E of one sign.
    Returns dict(dt, e_shift, sign, warnings).
    """
    R = hi - lo
    warnings = []
    if signal == "complex":
        if shift == "none":
            es, W = 0.0, max(abs(lo), abs(hi))
            dt = safety * math.pi / W
        else:
            es = 0.5 * (lo + hi)
            dt = safety * 2.0 * math.pi / R
        sign = None
    elif signal == "real":
        if shift == "classical":
            raise ValueError("A real signal cannot be shifted classically; "
                             "use shift='circuit' or 'none'.")
        if shift == "none":
            es, W = 0.0, max(abs(lo), abs(hi))
            dt = safety * math.pi / W
            sign = -1 if lo < 0 else +1
            if lo < 0 < hi:
                warnings.append("Populated window straddles 0: +E and -E are "
                                "indistinguishable in a real signal. "
                                "Use shift='circuit'.")
        else:
            dt = safety * math.pi / R
            band = math.pi / dt
            es = lo - 0.5 * (band - R)           # equal margins on both sides
            sign = +1
    else:
        raise ValueError("signal must be 'complex' or 'real'")
    return {"dt": dt, "e_shift": es, "sign": sign, "warnings": warnings,
            "signal": signal, "shift": shift}


def apparent_energy(E, dt, e_shift=0.0, signal="complex", sign=None):
    """Where frequency E appears after sampling at dt (aliasing included)."""
    phi = math.remainder((E - e_shift) * dt, 2.0 * math.pi)   # in [-pi, pi]
    if signal == "complex":
        return e_shift + phi / dt
    return e_shift + (sign if sign is not None else -1) * abs(phi) / dt


def alias_table(kr: KrylovResult, dt, e_shift=0.0, signal="complex", sign=None):
    rows = []
    for E, w in zip(kr.ritz, kr.weights):
        Ea = apparent_energy(E, dt, e_shift, signal, sign)
        rows.append((E, w, Ea, abs(Ea - E) > 1e-8 * max(1.0, abs(E))))
    return rows


# =============================================================================
# 6. Trotter
# =============================================================================
def _popcount_u64(a):
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(a)
    a = a - ((a >> np.uint64(1)) & np.uint64(0x5555555555555555))
    a = (a & np.uint64(0x3333333333333333)) + ((a >> np.uint64(2)) & np.uint64(0x3333333333333333))
    a = (a + (a >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (a * np.uint64(0x0101010101010101)) >> np.uint64(56)


def trotter_constant(table: PauliTable, chunk: int = 1024):
    """
    C1 = sum_{j<k, anticommuting} |c_j c_k|.
    First-order product formula with step delta: energy bias <~ delta * C1
    (from (delta/2) sum_{j<k} ||[H_j, H_k]||, with ||[P_j, P_k]|| = 2).
    """
    if table.n_qubits > 63 or table.n_terms == 0:
        return None
    x = np.array(table.flip, dtype=np.uint64)
    z = np.array(table.zy, dtype=np.uint64)
    c = table.coeff
    total = 0.0
    for s in range(0, len(x), chunk):
        xs, zs, cs = x[s:s + chunk, None], z[s:s + chunk, None], c[s:s + chunk, None]
        anti = (_popcount_u64((xs & z[None, :]) ^ (zs & x[None, :])) & np.uint64(1)).astype(bool)
        total += float((cs * c[None, :] * anti).sum())
    return 0.5 * total


# =============================================================================
# Driver
# =============================================================================
@dataclass
class Diagnostics:
    table: PauliTable
    krylov: KrylovResult
    window: tuple
    window_rigorous: tuple
    step: dict
    trotter_c1: float | None
    constant: float = 0.0

    def trotter_substeps(self, dt, bias):
        if self.trotter_c1 is None:
            return None
        return max(1, math.ceil(dt * self.trotter_c1 / bias))

    def report(self, dt_current=None, trotter_bias=1e-3):
        t, k, s = self.table, self.krylov, self.step
        print("== Pauli table ==")
        print(f"qubits={t.n_qubits}  terms={t.n_terms}  max weight={t.max_weight}")
        print(f"identity={t.identity:.6f}  L1={t.l1:.6f}  L2={t.l2:.6f}  L1/L2={t.l1 / t.l2:.2f}")
        print("\n== Krylov space of the reference ==")
        print(f"steps={len(k.alphas)}  closed={k.closed}  truncated={k.truncated}  "
              f"largest vector={k.n_states} bitstrings")
        print(f"E_ref={k.alphas[0]:.8f}  sigma={k.betas[0]:.6f}")
        for name, ok in k.charges.items():
            print(f"  {name}: {ok}")
        print("  Ritz value       weight      residual")
        for E, w, r in zip(k.ritz, k.weights, k.residuals):
            print(f"  {E:14.8f}  {w:10.3e}  {r:10.3e}")
        lo, hi = self.window
        rlo, rhi = self.window_rigorous
        print("\n== Populated window ==")
        print(f"estimate  [{lo:.6f}, {hi:.6f}]  width={hi - lo:.6f}")
        print(f"rigorous  [{rlo:.6f}, {rhi:.6f}]  width={rhi - rlo:.6f}")
        print(f"\n== Recommended step ({s['signal']} signal, shift={s['shift']}) ==")
        print(f"dt={s['dt']:.6f}  e_shift={s['e_shift']:.6f}"
              + (f"  assumed sign={s['sign']:+d}" if s['sign'] is not None else ""))
        for msg in s["warnings"]:
            print("  WARNING:", msg)
        if self.trotter_c1 is not None:
            m = self.trotter_substeps(s["dt"], trotter_bias)
            print(f"Trotter C1={self.trotter_c1:.4f}  -> <= {m} first-order sub-steps per "
                  f"sample for bias <= {trotter_bias:g}")
            print("  (upper bound, linear in delta; for real H and real eigenstates the "
                  "actual bias is O(delta^2) -- calibrate on a small instance)")
        if dt_current is not None:
            if s["signal"] == "real" and s["shift"] == "none":
                es_cur, sign_cur = 0.0, s["sign"]
            else:
                es_cur, sign_cur = s["e_shift"], s["sign"]
            print(f"\n== Check of dt={dt_current} (same signal/shift) ==")
            ratio = s['dt'] / dt_current
            if ratio >= 1:
                print(f"recommended / current = {ratio:.1f}x "
                      f"(that many times fewer samples for the same total time)")
            else:
                print(f"recommended / current = {ratio:.2f}: current dt exceeds the "
                      f"alias-free limit")
            for E, w, Ea, al in alias_table(k, dt_current, es_cur, s["signal"], sign_cur):
                flag = "  ALIASED" if al else ""
                print(f"  E={E:14.8f}  w={w:9.2e}  appears at {Ea:14.8f}{flag}")


def diagnose(hamiltonian, reference_bits, signal="complex", shift=None,
             window="lanczos", lanczos_steps=8, max_states=200_000,
             safety=0.9, constant=0.0):
    if shift is None:
        shift = "classical" if signal == "complex" else "circuit"
    table = PauliTable(hamiltonian, n_qubits=len(reference_bits))
    kr = lanczos_from_reference(table, bits_to_int(reference_bits),
                                steps=lanczos_steps, max_states=max_states)
    win = populated_window(table, kr, window)
    rig = populated_window(table, kr, "rigorous")
    step = recommend_step(*win, signal=signal, shift=shift, safety=safety)
    return Diagnostics(table, kr, win, rig, step, trotter_constant(table), constant)


# =============================================================================
if __name__ == "__main__":
    # H2 / STO-3G at 0.7414 A, Jordan-Wigner (OpenFermion ordering)
    H2 = {
        (): -0.09706626816762845,
        ((0, "Z"),): 0.17141282644776884, ((1, "Z"),): 0.17141282644776884,
        ((2, "Z"),): -0.22343153690813564, ((3, "Z"),): -0.22343153690813564,
        ((0, "Z"), (1, "Z")): 0.16868898170361205,
        ((0, "Z"), (2, "Z")): 0.12062523483390425,
        ((0, "Z"), (3, "Z")): 0.16592785033770352,
        ((1, "Z"), (2, "Z")): 0.16592785033770352,
        ((1, "Z"), (3, "Z")): 0.12062523483390425,
        ((2, "Z"), (3, "Z")): 0.17441287612261583,
        ((0, "X"), (1, "X"), (2, "Y"), (3, "Y")): -0.0453026155037993,
        ((0, "X"), (1, "Y"), (2, "Y"), (3, "X")): 0.0453026155037993,
        ((0, "Y"), (1, "X"), (2, "X"), (3, "Y")): 0.0453026155037993,
        ((0, "Y"), (1, "Y"), (2, "X"), (3, "X")): -0.0453026155037993,
    }
    hf = [1, 1, 0, 0]
    for sig in ("complex", "real"):
        print("=" * 72)
        diagnose(H2, hf, signal=sig).report(dt_current=2.6)