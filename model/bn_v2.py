"""model/bn_v2.py -- discrete Bayesian network with *real* structure learning.

Replaces ``model/bn_model.py`` (referenced by ``train_bn.py``), which did not
learn a structure at all: it hard-coded a chain ``col[0] -> col[1] -> ... ->
col[n-1]`` in dataframe column order and fitted the CPTs of that chain on the
raw, undiscretised columns.  That is the direct cause of the invalid records --
``Age`` entered the network with 106 states, so a single Age->X CPT already had
~1400 free parameters, and the chain's independence assumptions (X_i _||_ X_j |
X_{i-1} for every non-adjacent pair) let ancestral sampling emit combinations
that are impossible in the PUMS universe even though every *marginal* value was
in support.

What this module provides
-------------------------
``learn_structure``   greedy hill climbing over DAGs with add / remove / reverse
                      moves, a decomposable score (BIC or BDeu), a max-parents
                      cap and an explicit acyclicity test.  Returns the parent
                      sets plus the full score trajectory.
``DiscreteBN``        CPTs with Dirichlet (Laplace / BDeu) smoothing and
                      ancestral sampling in topological order.

Everything works on **integer codes**: the caller is responsible for
discretisation.  ``train_bn_v2.py`` gets those codes from
``utils.eval_v2.EvalConfig`` so that the BN's state space is byte-identical to
the paper's matching space.

pgmpy seam
----------
pgmpy is deliberately *not* a dependency (it is not installed in this
environment and must not be installed).  The swap-in point is a single
function-table entry::

    STRUCTURE_BACKENDS["pgmpy_hc"] = my_pgmpy_wrapper

A backend is any callable with the signature

    backend(codes: np.ndarray[int32, (n, m)],
            cards: Sequence[int],
            names: Sequence[str],
            *,
            max_parents: int,
            score: str,
            ess: float,
            weights: Optional[np.ndarray],
            fixed_edges, forbidden_edges, max_iter, tol, verbose) -> StructureResult

returning a :class:`StructureResult` whose ``parents`` maps *column name* ->
list of parent column names.  ``DiscreteBN.fit`` consumes exactly that, so a
pgmpy ``HillClimbSearch``/``BicScore`` wrapper (or ``PC``, or ``ExhaustiveSearch``)
drops in without touching the trainer.  Note that pgmpy would have to be given
the same integer codes and the same cardinalities, otherwise the state space
stops matching the frozen evaluation config.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.special import gammaln

SCORES = ("bic", "bdeu")


# --------------------------------------------------------------------------- #
# counting
# --------------------------------------------------------------------------- #
def _config_index(codes: np.ndarray, cards: Sequence[int],
                  parents: Sequence[int]) -> Tuple[np.ndarray, int]:
    """Mixed-radix index of the parent configuration of every row.

    Returns ``(idx, q)`` with ``q = prod(cards[p] for p in parents)`` -- the size
    of the *full Cartesian* parent-configuration space (unobserved configurations
    included, which is what the BIC/BDeu penalties are defined over).
    """
    if not parents:
        return np.zeros(codes.shape[0], dtype=np.int64), 1
    idx = np.zeros(codes.shape[0], dtype=np.int64)
    q = 1
    for p in parents:
        idx = idx * int(cards[p]) + codes[:, p].astype(np.int64)
        q *= int(cards[p])
    return idx, int(q)


def contingency(codes: np.ndarray, cards: Sequence[int], child: int,
                parents: Sequence[int],
                weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int]:
    """``(q_obs, r)`` count table restricted to *observed* parent configs.

    Only observed configurations are materialised (a full ``q x r`` table would
    blow up at max_parents=3 with 19-state columns for no benefit: every
    unobserved configuration contributes 0 to the log-likelihood and a constant
    to the BDeu prior term, both handled analytically below).  The second return
    value is the full Cartesian ``q``.
    """
    r = int(cards[child])
    j, q = _config_index(codes, cards, parents)
    uniq, inv = np.unique(j, return_inverse=True)
    key2 = inv.astype(np.int64) * r + codes[:, child].astype(np.int64)
    cnt = np.bincount(key2, weights=weights, minlength=uniq.size * r)
    return cnt.reshape(uniq.size, r).astype(np.float64), q


# --------------------------------------------------------------------------- #
# decomposable scores
# --------------------------------------------------------------------------- #
def local_score(codes: np.ndarray, cards: Sequence[int], child: int,
                parents: Sequence[int], score: str = "bic", ess: float = 10.0,
                weights: Optional[np.ndarray] = None,
                n_eff: Optional[float] = None) -> float:
    """Decomposable local score of ``child`` given ``parents`` (higher = better)."""
    n_ijk, q = contingency(codes, cards, child, parents, weights)
    r = int(cards[child])
    n_ij = n_ijk.sum(axis=1)
    n_tot = float(n_eff if n_eff is not None else n_ijk.sum())

    if score == "bic":
        nz = n_ijk > 0
        ll = float(np.sum(n_ijk[nz] * np.log(n_ijk[nz])))
        pos = n_ij > 0
        ll -= float(np.sum(n_ij[pos] * np.log(n_ij[pos])))
        penalty = 0.5 * math.log(max(n_tot, 2.0)) * q * (r - 1)
        return ll - penalty

    if score == "bdeu":
        a_ij = float(ess) / q
        a_ijk = float(ess) / (q * r)
        # observed configurations
        term = float(np.sum(gammaln(a_ij) - gammaln(a_ij + n_ij)))
        term += float(np.sum(gammaln(a_ijk + n_ijk) - gammaln(a_ijk)))
        # unobserved parent configurations contribute exactly 0 (all N=0)
        return term

    raise ValueError(f"unknown score {score!r} (expected one of {SCORES})")


class ScoreCache:
    """Memoises ``local_score`` on ``(child, frozenset(parents))``."""

    def __init__(self, codes: np.ndarray, cards: Sequence[int], score: str = "bic",
                 ess: float = 10.0, weights: Optional[np.ndarray] = None):
        self.codes = codes
        self.cards = list(cards)
        self.score = score
        self.ess = float(ess)
        self.weights = weights
        self.n_eff = float(codes.shape[0] if weights is None else np.sum(weights))
        self._cache: Dict[Tuple[int, frozenset], float] = {}
        self.n_calls = 0
        self.n_hits = 0

    def __call__(self, child: int, parents: Sequence[int]) -> float:
        key = (int(child), frozenset(int(p) for p in parents))
        self.n_calls += 1
        hit = self._cache.get(key)
        if hit is not None:
            self.n_hits += 1
            return hit
        v = local_score(self.codes, self.cards, child, sorted(key[1]), self.score,
                        self.ess, self.weights, n_eff=self.n_eff)
        self._cache[key] = v
        return v


# --------------------------------------------------------------------------- #
# graph helpers
# --------------------------------------------------------------------------- #
def _children_map(parents: Dict[int, Set[int]], m: int) -> Dict[int, Set[int]]:
    ch: Dict[int, Set[int]] = {i: set() for i in range(m)}
    for v, ps in parents.items():
        for p in ps:
            ch[p].add(v)
    return ch


def _reachable(children: Dict[int, Set[int]], src: int, dst: int) -> bool:
    """Is ``dst`` reachable from ``src`` following parent->child edges?"""
    if src == dst:
        return True
    stack = [src]
    seen = {src}
    while stack:
        u = stack.pop()
        for w in children[u]:
            if w == dst:
                return True
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return False


def topological_order(parents: Dict[int, Sequence[int]], m: int) -> List[int]:
    """Kahn's algorithm; raises if the graph is cyclic."""
    indeg = {i: len(parents.get(i, ())) for i in range(m)}
    children = _children_map({i: set(parents.get(i, ())) for i in range(m)}, m)
    ready = sorted(i for i in range(m) if indeg[i] == 0)
    order: List[int] = []
    while ready:
        u = ready.pop(0)
        order.append(u)
        for w in sorted(children[u]):
            indeg[w] -= 1
            if indeg[w] == 0:
                ready.append(w)
        ready.sort()
    if len(order) != m:
        raise ValueError("graph is cyclic")
    return order


# --------------------------------------------------------------------------- #
# structure learning
# --------------------------------------------------------------------------- #
@dataclass
class StructureResult:
    parents: Dict[str, List[str]]
    edges: List[Tuple[str, str]]
    trajectory: List[Dict[str, Any]]
    score_start: float
    score_final: float
    score: str
    ess: float
    max_parents: int
    n_iter: int
    converged: bool
    backend: str = "numpy_hc"
    wall_clock_s: float = 0.0
    cache_stats: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "score": self.score,
            "ess": self.ess,
            "max_parents": self.max_parents,
            "n_edges": len(self.edges),
            "edges": [list(e) for e in self.edges],
            "parents": {k: list(v) for k, v in self.parents.items()},
            "score_start": self.score_start,
            "score_final": self.score_final,
            "score_improvement": self.score_final - self.score_start,
            "n_iter": self.n_iter,
            "converged": self.converged,
            "wall_clock_s": self.wall_clock_s,
            "cache_stats": dict(self.cache_stats),
            "trajectory": self.trajectory,
        }


def _learn_structure_numpy_hc(
    codes: np.ndarray,
    cards: Sequence[int],
    names: Sequence[str],
    *,
    max_parents: int = 3,
    score: str = "bic",
    ess: float = 10.0,
    weights: Optional[np.ndarray] = None,
    fixed_edges: Sequence[Tuple[str, str]] = (),
    forbidden_edges: Sequence[Tuple[str, str]] = (),
    max_iter: int = 400,
    tol: float = 1e-6,
    verbose: bool = True,
) -> StructureResult:
    """Greedy hill climbing over DAGs (add / remove / reverse).

    Starts from the empty graph, and at every iteration evaluates *all* legal
    single-edge moves, taking the one with the largest score gain.  Stops when
    no move improves the score by more than ``tol`` (or after ``max_iter``
    accepted moves).  ``fixed_edges`` are inserted first and never removed or
    reversed; ``forbidden_edges`` are never added.
    """
    t0 = time.time()
    m = len(names)
    name_to_i = {n: i for i, n in enumerate(names)}
    sc = ScoreCache(codes, cards, score=score, ess=ess, weights=weights)

    parents: Dict[int, Set[int]] = {i: set() for i in range(m)}
    forb = {(name_to_i[a], name_to_i[b]) for a, b in forbidden_edges}
    fixed = {(name_to_i[a], name_to_i[b]) for a, b in fixed_edges}

    for (u, v) in sorted(fixed):
        if len(parents[v]) >= max_parents:
            raise ValueError(f"fixed edge {names[u]}->{names[v]} exceeds max_parents")
        ch = _children_map(parents, m)
        if _reachable(ch, v, u):
            raise ValueError(f"fixed edge {names[u]}->{names[v]} would create a cycle")
        parents[v].add(u)

    local = {i: sc(i, parents[i]) for i in range(m)}
    score_start = float(sum(local.values()))
    total = score_start
    if verbose:
        print(f"[bn] hill climbing: m={m} n={codes.shape[0]} score={score} "
              f"max_parents={max_parents} start={score_start:.2f}", flush=True)

    trajectory: List[Dict[str, Any]] = [{
        "iter": 0, "move": "init", "edge": None, "delta": 0.0,
        "total_score": score_start, "n_edges": int(sum(len(p) for p in parents.values())),
    }]
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        children = _children_map(parents, m)
        best = (tol, None)  # (delta, (kind, u, v, new_local))
        for v in range(m):
            pv = parents[v]
            # ---- remove ----
            for u in pv:
                if (u, v) in fixed:
                    continue
                new_pv = pv - {u}
                d = sc(v, new_pv) - local[v]
                if d > best[0]:
                    best = (d, ("remove", u, v, {v: sc(v, new_pv)}))
            # ---- add ----
            if len(pv) < max_parents:
                for u in range(m):
                    if u == v or u in pv or (u, v) in forb:
                        continue
                    if _reachable(children, v, u):     # v ->...-> u  =>  cycle
                        continue
                    new_pv = pv | {u}
                    d = sc(v, new_pv) - local[v]
                    if d > best[0]:
                        best = (d, ("add", u, v, {v: sc(v, new_pv)}))
            # ---- reverse (u->v becomes v->u) ----
            for u in pv:
                if (u, v) in fixed or (v, u) in forb:
                    continue
                if len(parents[u]) >= max_parents:
                    continue
                tmp = {k: set(p) for k, p in parents.items()}
                tmp[v].discard(u)
                if _reachable(_children_map(tmp, m), u, v):
                    continue
                new_pv = pv - {u}
                new_pu = parents[u] | {v}
                d = (sc(v, new_pv) - local[v]) + (sc(u, new_pu) - local[u])
                if d > best[0]:
                    best = (d, ("reverse", u, v,
                                {v: sc(v, new_pv), u: sc(u, new_pu)}))
        delta, move = best
        if move is None:
            converged = True
            it -= 1
            break
        kind, u, v, new_local = move
        if kind == "add":
            parents[v].add(u)
        elif kind == "remove":
            parents[v].discard(u)
        else:
            parents[v].discard(u)
            parents[u].add(v)
        local.update(new_local)
        total += delta
        rec = {"iter": it, "move": kind, "edge": [names[u], names[v]],
               "delta": float(delta), "total_score": float(total),
               "n_edges": int(sum(len(p) for p in parents.values()))}
        trajectory.append(rec)
        if verbose and (it % 5 == 0 or it == 1):
            print(f"[bn]   it={it:3d} {kind:7s} {names[u]}->{names[v]} "
                  f"delta={delta:+.2f} total={total:.2f} edges={rec['n_edges']}",
                  flush=True)

    # numerical drift guard: recompute the total from scratch
    total = float(sum(sc(i, parents[i]) for i in range(m)))
    edges = sorted((names[u], names[v]) for v in range(m) for u in parents[v])
    res = StructureResult(
        parents={names[v]: sorted(names[u] for u in parents[v]) for v in range(m)},
        edges=edges,
        trajectory=trajectory,
        score_start=score_start,
        score_final=total,
        score=score,
        ess=float(ess),
        max_parents=int(max_parents),
        n_iter=int(it),
        converged=converged,
        backend="numpy_hc",
        wall_clock_s=round(time.time() - t0, 2),
        cache_stats={"calls": sc.n_calls, "hits": sc.n_hits, "entries": len(sc._cache)},
    )
    if verbose:
        print(f"[bn] done: {len(edges)} edges, score {score_start:.2f} -> {total:.2f} "
              f"(+{total - score_start:.2f}) in {res.wall_clock_s}s, "
              f"converged={converged}", flush=True)
    return res


#: swap-in point documented in the module docstring (e.g. a pgmpy HillClimbSearch)
STRUCTURE_BACKENDS: Dict[str, Callable[..., StructureResult]] = {
    "numpy_hc": _learn_structure_numpy_hc,
}


def learn_structure(codes: np.ndarray, cards: Sequence[int], names: Sequence[str],
                    backend: str = "numpy_hc", **kw) -> StructureResult:
    if backend not in STRUCTURE_BACKENDS:
        raise ValueError(f"unknown structure backend {backend!r}; "
                         f"available: {sorted(STRUCTURE_BACKENDS)}")
    return STRUCTURE_BACKENDS[backend](codes, cards, names, **kw)


@dataclass
class NumericDecoder:
    """Inverse of ``EvalConfig``'s discretisation for one numeric column.

    A BN sampled in the frozen matching space only knows the *bin* of a numeric
    attribute.  Two decoders are offered:

    ``uniform``    draw uniformly between the bin's edges (DEFAULT since 2026-08-26).
                   This is what the reference implementation does -- FlowSyn's
                   Discretizer.inverse_transform is
                   ``np.random.uniform(bins[k], bins[k+1])`` -- and it is the
                   assumption-free answer given only the bin.
    ``empirical``  resample a value from the **training** rows that fall in the
                   same bin.  DO NOT use this when continuous columns are scored in
                   the continuous space: it re-emits real training values, so the
                   marginal it produces is the training marginal and the
                   discretisation loss disappears from any binning-free metric.
                   Measured on ACS: at 12 quantile bins of HINCP the normalised
                   Wasserstein distance is 0.0024 under empirical decoding against
                   0.3249 under uniform -- a factor of 135 -- while the mutual
                   information retained against age is the same either way (93% vs
                   94%).  Decoding can rescue the marginal; it cannot rescue the
                   dependence the binning destroyed.  Re-emitting training values is
                   record reuse at the value level, which is the standard we hold the
                   persona prior to.
    ``midpoint``   deterministic bin representative (integer bins are exact;
                   binned columns get the bin midpoint, the >=t tail gets the
                   training mean of the tail).

    The choice is invisible to SRMSE and to block P/R -- both re-discretise with
    the same spec -- but it *is* visible to ``record_validity_rate`` (Age<3 and
    Age<15 rules cut inside the [0,5) and [15,20) bins) and to any downstream
    use of the synthetic population.  It is recorded in ``metrics.json``.
    """

    name: str
    kind: str
    mode: str
    pools: List[np.ndarray]          # per-code training values
    reps: np.ndarray                 # per-code fallback representative
    #: bin boundaries, needed by the "uniform" mode; None for non-binned kinds, where a
    #: code IS a value and all three modes coincide
    edges: Optional[np.ndarray]
    lo: float
    hi: float
    is_integer: bool
    #: the code of the spec's explicit NA cell, if it has one (BinningSpec.na_cell).
    #: Sampling it means the model chose "no value here", and the only faithful decode of
    #: that is NaN -- putting a number there would invent an in-universe record out of an
    #: out-of-universe one, which is exactly what the universe rules then score.
    na_code: Optional[int] = None

    def decode(self, codes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = np.empty(codes.shape[0], dtype=np.float64)
        for k in range(len(self.pools)):
            m = codes == k
            n = int(m.sum())
            if n == 0:
                continue
            pool = self.pools[k]
            if self.mode == "empirical" and pool.size:
                out[m] = pool[rng.integers(0, pool.size, size=n)]
            elif self.mode == "uniform" and self.edges is not None:
                e = self.edges
                lo_k, hi_k = float(e[k]), float(e[k + 1])
                out[m] = rng.uniform(lo_k, hi_k, size=n) if hi_k > lo_k else lo_k
            else:
                out[m] = self.reps[k]
        # F6: clip to the TRAINING support first, round afterwards
        out = np.clip(out, self.lo, self.hi)
        if self.is_integer:
            ilo = int(np.ceil(self.lo - 1e-6))
            ihi = int(np.floor(self.hi + 1e-6))
            out = np.clip(np.rint(out), ilo, ihi)
        if self.na_code is not None:
            out = np.where(codes == self.na_code, np.nan, out)
        return out

    # ---- plain-data (de)serialisation, so a checkpoint can be read without
    # importing this class ----
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "kind": self.kind, "mode": self.mode,
            "pools": [p.tolist() for p in self.pools],
            "reps": self.reps.tolist(),
            "lo": self.lo, "hi": self.hi, "is_integer": self.is_integer,
            "na_code": self.na_code,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NumericDecoder":
        return cls(name=d["name"], kind=d["kind"], mode=d["mode"],
                   pools=[np.asarray(p, dtype=np.float64) for p in d["pools"]],
                   reps=np.asarray(d["reps"], dtype=np.float64),
                   lo=float(d["lo"]), hi=float(d["hi"]),
                   is_integer=bool(d["is_integer"]),
                   na_code=(None if d.get("na_code") is None else int(d["na_code"])))


def build_numeric_decoder(name: str, spec, values: np.ndarray, codes: np.ndarray,
                          mode: str) -> NumericDecoder:
    n_cat = spec.n_categories
    # An na_cell spec puts the NA label last, so its code is n_cat-1 and it has no edge
    # pair; the bin loop below must not walk into it.
    na_code = (n_cat - 1) if bool(getattr(spec, "na_cell", False)) else None
    finite = values[np.isfinite(values)]
    lo, hi = float(finite.min()), float(finite.max())
    is_int = bool(np.allclose(finite, np.rint(finite)))
    pools: List[np.ndarray] = []
    reps = np.empty(n_cat, dtype=np.float64)
    for k in range(n_cat):
        if k == na_code:
            pools.append(np.empty(0, dtype=np.float64))
            reps[k] = np.nan
            continue
        v = values[codes == k]
        v = v[np.isfinite(v)]
        pools.append(np.ascontiguousarray(v, dtype=np.float64))
        if v.size:
            reps[k] = float(np.mean(v))
        elif spec.kind == "integer":
            reps[k] = float(int(spec.int_lo) + k)
        elif spec.kind == "binned":
            e = np.asarray(spec.edges, dtype=np.float64)
            reps[k] = 0.5 * (e[k] + e[k + 1])
        else:
            reps[k] = lo
    # "integer" columns map a code to one value, so uniform/empirical/midpoint coincide
    # and there is nothing to draw between; only "binned" columns carry real edges.
    edges = (np.asarray(spec.edges, dtype=np.float64)
             if spec.kind == "binned" and spec.edges is not None else None)
    if mode == "uniform" and edges is None:
        mode = "empirical" if spec.kind != "integer" else "midpoint"
    return NumericDecoder(name=name, kind=spec.kind, mode=mode, pools=pools,
                          reps=reps, edges=edges, lo=lo, hi=hi, is_integer=is_int,
                          na_code=na_code)


# --------------------------------------------------------------------------- #
# the network itself
# --------------------------------------------------------------------------- #
class DiscreteBN:
    """Discrete BN over integer-coded columns.

    ``cpts[node]`` has shape ``(q, r)`` with ``q`` the full Cartesian parent
    configuration count and ``r`` the child cardinality; rows sum to 1.  CPTs
    are smoothed with a Dirichlet prior::

        P(x_i = k | pa = j) = (N_ijk + alpha) / (N_ij + alpha * r_i)

    ``alpha = 1`` is Laplace; ``alpha = ess / (q * r)`` reproduces the BDeu
    posterior mean (use ``alpha_mode='bdeu'``).
    """

    def __init__(self, names: Sequence[str], cards: Sequence[int],
                 parents: Dict[str, Sequence[str]]):
        self.names = list(names)
        self.cards = [int(c) for c in cards]
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.parents = {n: list(parents.get(n, ())) for n in self.names}
        self.parent_idx = {n: [self.idx[p] for p in self.parents[n]] for n in self.names}
        self.order = topological_order(
            {self.idx[n]: [self.idx[p] for p in self.parents[n]] for n in self.names},
            len(self.names))
        self.cpts: Dict[str, np.ndarray] = {}
        self.alpha: Dict[str, float] = {}
        self.n_params = 0
        self.empty_config_share: Dict[str, float] = {}

    # ---------------- fit ---------------- #
    def fit(self, codes: np.ndarray, alpha: float = 1.0, alpha_mode: str = "fixed",
            weights: Optional[np.ndarray] = None, ess: float = 10.0) -> "DiscreteBN":
        self.n_params = 0
        for n in self.names:
            i = self.idx[n]
            pi = self.parent_idx[n]
            r = self.cards[i]
            j, q = _config_index(codes, self.cards, pi)
            key = j * r + codes[:, i].astype(np.int64)
            cnt = np.bincount(key, weights=weights, minlength=q * r
                              ).astype(np.float64).reshape(q, r)
            a = float(ess) / (q * r) if alpha_mode == "bdeu" else float(alpha)
            post = cnt + a
            self.cpts[n] = post / post.sum(axis=1, keepdims=True)
            self.alpha[n] = a
            self.n_params += q * (r - 1)
            n_ij = cnt.sum(axis=1)
            self.empty_config_share[n] = float(np.mean(n_ij == 0))
        return self

    # ---------------- sample ---------------- #
    def sample(self, n: int, rng: np.random.Generator,
               chunk: int = 200_000) -> np.ndarray:
        """Ancestral sampling in topological order -> ``(n, m)`` int32 codes."""
        m = len(self.names)
        out = np.empty((n, m), dtype=np.int32)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            b = e - s
            buf = np.zeros((b, m), dtype=np.int64)
            for i in self.order:
                name = self.names[i]
                pi = self.parent_idx[name]
                cpt = self.cpts[name]
                if pi:
                    j = np.zeros(b, dtype=np.int64)
                    for p in pi:
                        j = j * self.cards[p] + buf[:, p]
                else:
                    j = np.zeros(b, dtype=np.int64)
                cdf = np.cumsum(cpt[j], axis=1)
                u = rng.random(b)[:, None] * cdf[:, -1:]
                buf[:, i] = (u > cdf).sum(axis=1)
                np.clip(buf[:, i], 0, self.cards[i] - 1, out=buf[:, i])
            out[s:e] = buf.astype(np.int32)
        return out

    # ---------------- evidence sampling ---------------- #
    def sample_evidence(self, n: int, allowed: Dict[str, Sequence[int]],
                        rng: np.random.Generator, chunk: int = 200_000
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """Likelihood-weighted ancestral sampling under SET evidence.

        ``allowed[name]`` is the set of admissible integer codes for that column.
        This is the structured-conditioning baseline that a narrative interface
        can be compared against (Bayesian networks already offer structured control).

        Plain rejection sampling would also work but is a different method with a
        different cost curve; likelihood weighting is the standard exact-support
        answer and, unlike rejection, its price shows up as variance rather than
        as discarded draws.  At each constrained node the conditional is truncated
        to the admissible set, the sample is drawn from the renormalised tail, and
        the mass that was cut away multiplies an importance weight.  Every returned
        row therefore satisfies the evidence by construction (attainment == 1), and
        the honest cost is reported as effective sample size, not as yield.

        Returns ``(codes, weights)``; weights are unnormalised.
        """
        m = len(self.names)
        out = np.empty((n, m), dtype=np.int32)
        w_out = np.empty(n, dtype=np.float64)
        masks = {}
        for name, vals in allowed.items():
            i = self.idx[name]
            mk = np.zeros(self.cards[i], dtype=bool)
            mk[np.asarray(list(vals), dtype=np.int64)] = True
            if not mk.any():
                raise ValueError(f"evidence on {name!r} admits no state")
            masks[i] = mk
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            b = e - s
            buf = np.zeros((b, m), dtype=np.int64)
            w = np.ones(b, dtype=np.float64)
            for i in self.order:
                name = self.names[i]
                pi = self.parent_idx[name]
                j = np.zeros(b, dtype=np.int64)
                for p in pi:
                    j = j * self.cards[p] + buf[:, p]
                pr = self.cpts[name][j]                      # (b, r)
                if i in masks:
                    pr = pr * masks[i][None, :]
                    tot = pr.sum(axis=1)
                    w *= tot                                  # mass removed by the cut
                    pr = pr / np.maximum(tot, 1e-300)[:, None]
                cdf = np.cumsum(pr, axis=1)
                u = rng.random(b)[:, None] * cdf[:, -1:]
                buf[:, i] = np.clip((u > cdf).sum(axis=1), 0, self.cards[i] - 1)
            out[s:e] = buf.astype(np.int32)
            w_out[s:e] = w
        return out, w_out

    @staticmethod
    def ess(w: np.ndarray) -> float:
        """Kish effective sample size of importance weights."""
        w = np.asarray(w, dtype=np.float64)
        sw = w.sum()
        return float(sw * sw / np.maximum((w * w).sum(), 1e-300)) if sw > 0 else 0.0

    # ---------------- io ---------------- #
    def edges(self) -> List[Tuple[str, str]]:
        return sorted((p, n) for n in self.names for p in self.parents[n])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "names": list(self.names),
            "cards": list(self.cards),
            "parents": {k: list(v) for k, v in self.parents.items()},
            "cpts": {k: v.tolist() for k, v in self.cpts.items()},
            "alpha": dict(self.alpha),
            "n_params": int(self.n_params),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DiscreteBN":
        bn = cls(d["names"], d["cards"], d["parents"])
        bn.cpts = {k: np.asarray(v, dtype=np.float64) for k, v in d["cpts"].items()}
        bn.alpha = dict(d.get("alpha", {}))
        bn.n_params = int(d.get("n_params", 0))
        return bn
