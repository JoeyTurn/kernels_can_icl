"""Eigen-theory + KRR for the ICL-structured kernel

    k(z, z') = k_even(c, c') * (x_q . x_q') * QSCALE,
    k_even(c, c') = [exp(-|c - c'|^2 / 2w^2) + exp(-|c + c'|^2 / 2w^2)] / 2,

c = [X.flatten, y.flatten] the context (query excluded), QSCALE = 1/(d v) so the query factor has unit
mean diagonal (the kernel is then O(1) on the diagonal and a ridge means what it did for the plain
Gaussian kernel).

Why the eigenstructure is cheap (each fact is checked in test_structured_theory.py):
  1. x_q is independent of the context, so eigen(k) = eigen(k_even on contexts) x eigen(x_q . x_q'),
     the latter being d modes x_qb / sqrt(v) with eigenvalue v * QSCALE = 1/d.
  2. cosh keeps only even context levels: k_even = env(c) env(c') sum_{l even} (c.c')^l / (l! w^{2l}).
  3. Kernel and data are invariant under permuting positions and under flipping the sign of one
     (x_i, y_i) pair. The target E[y_q | Z] is too, so ALL target power (ID and OOD, both measures are
     invariant) lives in the invariant sector: sums over distinct positions of per-position monomials
     of even degree ("legs"). 21 features at level 2, 357 at level 4 for d=5.
  4. Everything else has zero target power and zero S cross terms, so it enters the learning curve
     only via (eigenvalue, multiplicity): level 2 exactly (two closed-form sectors), levels >= 4 as an
     independent-coordinate staircase.

The S-expansion (paper Def. 5) is used within levels: eigenvalue = pref * (1/d) * eig(W^1/2 G~ W^1/2),
G~ the level's invariant Gram Schur-complemented against all lower levels, W the kernel's weight on
each invariant feature (orbit counting, `feature_weight`), pref = E[env]^2.

Two moment backends:
  "exact" -- Gaussian in-context x: symbolic conditional moments (kicl_floor._tok_poly) + distinct-
             position combinatorics, reduced over the pool (ID) or the N(0, sb2 I) prior (OOD) exactly.
  "mc"    -- Monte Carlo over contexts; required for WHITEN_X (sum_i x_i x_i^T = c I exactly, which makes
             positions dependent), and used as an independent check of "exact".
"""
import itertools
import math
import os
import pickle
import time

import numpy as np
import torch

import kicl_floor as KF

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_HERE, "results", "structured_cache")


# ---------------------------------------------------------------------------
# features: the invariant sector
# ---------------------------------------------------------------------------
# A token/leg is a per-position monomial (px, py): prod_a x_a^px[a] * y^py.
# A feature is a tuple of legs placed at DISTINCT positions, summed over ordered position tuples
# (the kicl_floor convention, so E[F | beta] carries falling(L, #legs)).

def leg_monomials(d, deg):
    out = []
    for py in range(deg, -1, -1):
        for comb in itertools.combinations_with_replacement(range(d), deg - py):
            px = [0] * d
            for a in comb:
                px[a] += 1
            out.append((tuple(px), py))
    return out


def _even_partitions(n, maxpart=None):
    if n == 0:
        yield ()
        return
    maxpart = n if maxpart is None else maxpart
    for p in range(min(n, maxpart), 1, -1):
        if p % 2 == 0:
            for rest in _even_partitions(n - p, p):
                yield (p,) + rest


def invariant_features(d, level):
    """Every invariant feature of context degree `level` (a complete basis of the sector)."""
    if level == 0:
        return [()]
    feats = []
    for part in _even_partitions(level):
        groups = [(s, part.count(s)) for s in sorted(set(part), reverse=True)]
        per_group = [list(itertools.combinations_with_replacement(leg_monomials(d, s), c))
                     for s, c in groups]
        for choice in itertools.product(*per_group):
            feats.append(tuple(leg for grp in choice for leg in grp))
    return feats


def leg_label(leg):
    px, py = leg
    s = "" if py == 0 else ("y" if py == 1 else f"y^{py}")
    for a, p in enumerate(px):
        if p:
            s += f"x{a}" if p == 1 else f"x{a}^{p}"
    return s


def feature_label(f):
    return "1" if not f else "*".join("S[" + leg_label(l) + "]" for l in f)


def feature_level(f):
    return sum(sum(px) + py for px, py in f)


def feature_xy_degree(f):
    return sum(sum(px) for px, _ in f), sum(py for _, py in f)


def feature_weight(f, L, w):
    """Kernel weight of an invariant feature F in the S-expansion.

    Level-l term of cosh: (c.c')^l / (l! w^2l) = sum_mu n_mu mu(c) mu(c') / (l! w^2l), n_mu = l!/prod(exp!)
    ordered slot tuples per monomial. Restricted to invariant f it is sum_orbits (n/|orbit|) O O^T/(l! w^2l)
    with O the orbit sum; F = s O (s = automorphisms of identical legs), |orbit| = falling(L, m)/s, so
    weight(F) = 1 / (prod_legs legfact * falling(L, m) * s * w^2l).
    """
    if not f:
        return 1.0
    lf = 1.0
    for px, py in f:
        lf *= math.factorial(py)
        for p in px:
            lf *= math.factorial(p)
    s = 1.0
    for leg in set(f):
        s *= math.factorial(f.count(leg))
    lev = feature_level(f)
    return 1.0 / (lf * KF._falling(L, len(f)) * s * w ** (2 * lev))


def _merge(a, b):
    return (tuple(p + q for p, q in zip(a[0], b[0])), a[1] + b[1])


def _matchings(A, B):
    """Yield (tokens, n_positions) over all ways legs of A and B share positions (at most one each)."""
    p, q = len(A), len(B)
    for r in range(min(p, q) + 1):
        for S in itertools.combinations(range(q), r):
            for At in itertools.permutations(range(p), r):
                inj = dict(zip(At, S))
                toks = [(_merge(A[a], B[inj[a]]) if a in inj else A[a]) for a in range(p)]
                toks += [B[b] for b in range(q) if b not in S]
                yield tuple(sorted(toks)), len(toks)


# ---------------------------------------------------------------------------
# exact backend: symbolic conditional moments, two-stage flattening
# ---------------------------------------------------------------------------

class ExactAlgebra:
    """E[F_i F_j | beta] and E[F_i | beta] as polynomials in beta, reduced over any beta measure.

    Stage 1 (per unique token multiset, a "key"): the product of per-token conditional moment polys.
    Stage 2 (per Gram entry): sum over matchings of falling(L, n) * key.
    Reduction is linear, so a measure only needs the beta-moments of the registered multi-indices.
    """

    def __init__(self, d, v, e, L):
        self.d, self.v, self.e, self.L = d, float(v), float(e), int(L)
        self._mono = {}
        self.keys = {}          # key -> id
        self.key_polys = []     # id -> dict

    def mono(self, tok):
        p = self._mono.get(tok)
        if p is None:
            p = KF._tok_poly(tok[0], tok[1], self.v, self.e, self.d)
            self._mono[tok] = p
        return p

    def key_id(self, key):
        i = self.keys.get(key)
        if i is None:
            poly = {(0,) * self.d: 1.0}
            for tok in key:
                poly = KF._pmul(poly, self.mono(tok))
            i = len(self.key_polys)
            self.keys[key] = i
            self.key_polys.append(poly)
        return i

    def entries(self, A, B):
        return [(self.key_id(toks), KF._falling(self.L, n)) for toks, n in _matchings(A, B)]

    def finalize(self):
        """Flatten key polys into (key_seg, kidx, coef) over one multi-index table Kmat."""
        kset, seg, kidx, coef = {}, [], [], []
        for i, poly in enumerate(self.key_polys):
            for k, c in poly.items():
                seg.append(i)
                kidx.append(kset.setdefault(k, len(kset)))
                coef.append(c)
        self.Kmat = np.array(list(kset.keys()), dtype=np.intp).reshape(-1, self.d)
        self.kseg = np.array(seg, dtype=np.intp)
        self.kkidx = np.array(kidx, dtype=np.intp)
        self.kcoef = np.array(coef, dtype=float)
        self.maxpow = int(self.Kmat.max()) + 2 if self.Kmat.size else 2

    def key_values(self, measure, shift=None):
        """Value of every key under `measure`; shift=b multiplies the integrand by beta_b."""
        K = self.Kmat
        if shift is not None:
            K = K.copy()
            K[:, shift] += 1
        kind, payload = measure
        if kind == "pool":
            B = np.asarray(payload, float)
            bp = B[:, :, None] ** np.arange(self.maxpow)[None, None, :]
            acc = np.ones((B.shape[0], K.shape[0]))
            for c in range(self.d):
                acc *= bp[:, c, K[:, c]]
            M = acc.mean(0)
        else:
            M = KF._gauss_moments(K, float(payload))
        return np.bincount(self.kseg, weights=self.kcoef * M[self.kkidx], minlength=len(self.key_polys))


def _flat(entry_lists):
    seg, kid, fall = [], [], []
    for r, ents in enumerate(entry_lists):
        for k, f in ents:
            seg.append(r); kid.append(k); fall.append(f)
    return (np.array(seg, np.intp), np.array(kid, np.intp), np.array(fall, float), len(entry_lists))


def _apply(flat, kv):
    seg, kid, fall, n = flat
    return np.bincount(seg, weights=fall * kv[kid], minlength=n)


class ExactMoments:
    """Symbolic layer for a feature list (beta-independent; cached to disk)."""

    def __init__(self, d, v, e, L, feats, verbose=True):
        t0 = time.time()
        self.d, self.v, self.e, self.L, self.feats = d, v, e, L, feats
        alg = ExactAlgebra(d, v, e, L)
        P = len(feats)
        self.iu, self.ju = np.triu_indices(P)
        self.gram = _flat([alg.entries(feats[i], feats[j]) for i, j in zip(self.iu, self.ju)])
        self.mean = _flat([alg.entries(f, ()) for f in feats])
        # level-2 sector moments: per-position second moments of degree-2 legs and tokens
        self.legs2 = leg_monomials(d, 2)
        toks1 = [(tuple(int(k == a) for k in range(d)), 0) for a in range(d)] + [((0,) * d, 1)]   # x..., y
        self.toks1 = toks1
        m2 = len(self.legs2)
        self.same = _flat([[(alg.key_id((_merge(a, b),)), 1.0)] for a in self.legs2 for b in self.legs2])
        self.mean2 = _flat([[(alg.key_id((a,)), 1.0)] for a in self.legs2])
        self.cross = _flat([[(alg.key_id(tuple(sorted((a, b)))), 1.0)] for a in self.legs2 for b in self.legs2])
        # E_t[E[c_a c_c | beta] E[c_b c_d | beta]] for the two-position sector, index order (a, c, b, d)
        self.tt = _flat([[(alg.key_id(tuple(sorted((_merge(a, c), _merge(b, dd))))), 1.0)]
                         for a in toks1 for c in toks1 for b in toks1 for dd in toks1])
        self.m2, self.nt = m2, len(toks1)
        alg.finalize()
        self.alg = alg
        if verbose:
            print(f"  exact moments: {P} features, {P*(P+1)//2:,} pairs, {len(alg.key_polys):,} keys, "
                  f"{alg.Kmat.shape[0]:,} beta-monomials, {time.time()-t0:.1f}s")

    def reduce(self, measure):
        """Uncentered moments under `measure`: G = E[F F^T], V[b] = E[beta_b F], plus level-2 sectors."""
        kv = self.alg.key_values(measure)
        P = len(self.feats)
        G = np.zeros((P, P))
        g = _apply(self.gram, kv)
        G[self.iu, self.ju] = g
        G[self.ju, self.iu] = g
        V = np.stack([_apply(self.mean, self.alg.key_values(measure, shift=b)) for b in range(self.d)])
        m2, nt = self.m2, self.nt
        TT = _apply(self.tt, kv).reshape(nt, nt, nt, nt)
        return dict(G=G, V=V,
                    same=_apply(self.same, kv).reshape(m2, m2),
                    cross=_apply(self.cross, kv).reshape(m2, m2),
                    mean2=_apply(self.mean2, kv),
                    TT=TT.transpose(0, 2, 1, 3).reshape(nt * nt, nt * nt))


# ---------------------------------------------------------------------------
# data: one sampler shared by KRR, the MC backend and the oracles
# ---------------------------------------------------------------------------

def sample(cfg, n, rng, betas=None, whiten=False, wscale=None, task_idx=None):
    """Contexts from the pool (`betas`, uniform task) or from the N(0, sb2 I) prior (betas=None).

    whiten=True: X = sqrt(c) Q, Q Haar on the Stiefel manifold, so sum_i x_i x_i^T = c I exactly
    (c = wscale, default L v, which keeps each coordinate's marginal variance at v). Query ~ N(0, v I).
    """
    d, L, v, e = cfg.dim, cfg.L, cfg.v, cfg.e
    if betas is None:
        B = rng.normal(0.0, cfg.sigma_beta, (n, d))
    else:
        betas = np.asarray(betas, float)
        idx = rng.integers(0, len(betas), n) if task_idx is None else task_idx
        B = betas[idx]
    if whiten:
        c = L * v if wscale is None else float(wscale)
        Q, R = np.linalg.qr(rng.normal(size=(n, L, d)))
        Q = Q * np.sign(np.diagonal(R, axis1=1, axis2=2))[:, None, :]
        X = math.sqrt(c) * Q
    else:
        X = rng.normal(0.0, math.sqrt(v), (n, L, d))
    y = np.einsum("nld,nd->nl", X, B) + rng.normal(0.0, math.sqrt(e), (n, L))
    xq = rng.normal(0.0, math.sqrt(v), (n, d))
    yq = (xq * B).sum(1) + rng.normal(0.0, math.sqrt(e), n)
    return dict(X=X, y=y, xq=xq, yq=yq, beta=B)


def context_vec(s):
    """[X.flatten, y.flatten], as in structured_rf.context_of."""
    n = s["X"].shape[0]
    return np.concatenate([s["X"].reshape(n, -1), s["y"]], axis=1)


# ---------------------------------------------------------------------------
# MC backend
# ---------------------------------------------------------------------------

class FeatureEval:
    """Evaluate invariant features (<= 3 legs, i.e. levels <= 6) on sampled contexts."""

    def __init__(self, d, feats):
        self.d, self.feats = d, feats

    def tokens(self, s):
        return np.concatenate([s["X"], s["y"][:, :, None]], axis=2)       # (n, L, d+1), y last

    def evaluator(self, tok):
        """leg -> per-position values (n, L), built recursively so each monomial costs one multiply."""
        memo = {((0,) * self.d, 0): None}

        def val(leg):
            if leg in memo:
                return memo[leg]
            px, py = leg
            if py:
                prev, col = (px, py - 1), self.d
            else:
                a = next(k for k, p in enumerate(px) if p)
                prev, col = (tuple(p - (k == a) for k, p in enumerate(px)), 0), a
            base = val(prev)
            out = tok[..., col].copy() if base is None else base * tok[..., col]
            memo[leg] = out
            return out
        return val

    def __call__(self, s):
        tok = self.tokens(s)
        val = self.evaluator(tok)
        sums = {}

        def S(*legs):
            m = legs[0]
            for l in legs[1:]:
                m = _merge(m, l)
            if m not in sums:
                sums[m] = val(m).sum(1)
            return sums[m]

        n = tok.shape[0]
        out = np.empty((n, len(self.feats)))
        for c, f in enumerate(self.feats):
            m = len(f)
            if m == 0:
                out[:, c] = 1.0
            elif m == 1:
                out[:, c] = S(f[0])
            elif m == 2:
                a, b = f
                out[:, c] = S(a) * S(b) - S(a, b)
            elif m == 3:
                a, b, cc = f
                out[:, c] = (S(a) * S(b) * S(cc) - S(a, b) * S(cc) - S(a, cc) * S(b) - S(b, cc) * S(a)
                             + 2 * S(a, b, cc))
            else:
                raise NotImplementedError("FeatureEval supports up to 3 legs (level 6)")
        return out


def mc_reduce(cfg, feats, n, rng, betas=None, whiten=False, wscale=None, batch=10000):
    """MC estimate of the same moments ExactMoments.reduce returns (pool if betas given, else prior)."""
    d, L = cfg.dim, cfg.L
    fe = FeatureEval(d, feats)
    legs2, toks1 = leg_monomials(d, 2), leg_monomials(d, 1)
    P, m2, nt = len(feats), len(legs2), d + 1
    G = np.zeros((P, P)); V = np.zeros((d, P))
    same = np.zeros((m2, m2)); cross = np.zeros((m2, m2)); mean2 = np.zeros(m2)
    TT = np.zeros((nt, nt, nt, nt))
    done = 0
    while done < n:
        B = min(batch, n - done)
        tidx = None if betas is None else (np.arange(B) + done) % len(betas)   # exact task mixture
        s = sample(cfg, B, rng, betas, whiten, wscale, task_idx=tidx)
        done += B
        F = fe(s)
        G += F.T @ F
        V += s["beta"].T @ F
        tok = fe.tokens(s)
        val = fe.evaluator(tok)
        Pm = np.stack([val(l) for l in legs2], axis=2)                      # (B, L, m2)
        Sm = Pm.sum(1)
        Pf = Pm.reshape(-1, m2)
        within = Pf.T @ Pf
        same += within / L
        cross += (Sm.T @ Sm - within) / (L * (L - 1))
        mean2 += Sm.sum(0) / L
        O = (tok[:, :, :, None] * tok[:, :, None, :]).reshape(B, L, nt * nt)   # per-position outer products
        Os = O.sum(1)
        Of = O.reshape(-1, nt * nt)
        TT += ((Os.T @ Os - Of.T @ Of) / (L * (L - 1))).reshape(nt, nt, nt, nt)
    return dict(G=G / n, V=V / n, same=same / n, cross=cross / n, mean2=mean2 / n,
                TT=(TT / n).transpose(0, 2, 1, 3).reshape(nt * nt, nt * nt))


# ---------------------------------------------------------------------------
# kernel envelope prefactor
# ---------------------------------------------------------------------------

def envelope_pref(cfg, betas, width, whiten=False, wscale=None):
    """pref = E_pool[exp(-|c|^2 / 2w^2)]^2 in closed form (context only; the query factor is exact)."""
    d, L, v, e = cfg.dim, cfg.L, cfg.v, cfg.e
    t = 1.0 / (2.0 * width ** 2)
    N2 = (np.asarray(betas, float) ** 2).sum(1)
    if whiten:
        c = L * v if wscale is None else float(wscale)
        per = math.exp(-t * c * d) * (1 + 2 * t * e) ** (-L / 2) * np.exp(-t * c * N2 / (1 + 2 * t * e))
    else:
        g = 1 + 2 * t * e + 2 * t * v * N2 / (1 + 2 * t * v)
        per = (1 + 2 * t * v) ** (-d * L / 2) * g ** (-L / 2)
    return float(per.mean()) ** 2


# ---------------------------------------------------------------------------
# the eigensystem
# ---------------------------------------------------------------------------

def _leg2_n(leg):
    """Ordered slot tuples of a degree-2 monomial: 1 for a square, 2 for a cross term."""
    px, py = leg
    f = math.factorial(py)
    for p in px:
        f *= math.factorial(p)
    return 2.0 / f


def _half_crossing(ns, curve):
    """First n where a monotone learnability curve crosses 1/2, log-interpolated (nan if it never does)."""
    c = np.asarray(curve, float)
    k = np.where(c >= 0.5)[0]
    if not len(k) or not np.isfinite(c).any():
        return float("nan")
    k = int(k[0])
    if k == 0:
        return float(ns[0])
    x0, x1, y0, y1 = math.log(ns[k - 1]), math.log(ns[k]), c[k - 1], c[k]
    return float(math.exp(x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0))) if y1 > y0 else float(ns[k])


class StructuredTheory:
    """Eigensystem + learning curves of the structured kernel on one ICL setting.

    backend "exact" (default, Gaussian x) or "mc" (forced when whiten=True).
    """

    def __init__(self, cfg, width, max_level=4, backend="exact", whiten=False, wscale=None,
                 mc_n=2_000_000, mc_seed=0, bulk_max_level=80, cache=True, verbose=True):
        self.cfg, self.w, self.max_level = cfg, float(width), int(max_level)
        self.whiten = bool(whiten)
        self.wscale = (cfg.L * cfg.v if wscale is None else float(wscale)) if whiten else None
        self.backend = "mc" if whiten else backend
        self.mc_n, self.mc_seed, self.bulk_max_level, self.verbose = int(mc_n), mc_seed, bulk_max_level, verbose
        d, L = cfg.dim, cfg.L
        self.feats = [f for lev in range(0, self.max_level + 1, 2) for f in invariant_features(d, lev)]
        self.level = np.array([feature_level(f) for f in self.feats])
        self.weight = np.array([feature_weight(f, L, self.w) for f in self.feats])
        self.labels = [feature_label(f) for f in self.feats]
        self.xy = np.array([feature_xy_degree(f) for f in self.feats])
        self.qfac = 1.0 / d                    # query-linear factor eigenvalue with QSCALE = 1/(d v)
        self._em = self._load_exact(cache) if self.backend == "exact" else None

    # -- moments ---------------------------------------------------------------------------------
    def _load_exact(self, cache):
        c = self.cfg
        tag = f"exact_d{c.dim}_L{c.L}_v{c.v:.8g}_e{c.e:.8g}_lev{self.max_level}.pkl"
        path = os.path.join(CACHE_DIR, tag)
        if cache and os.path.exists(path):
            with open(path, "rb") as fh:
                em = pickle.load(fh)
            if self.verbose:
                print(f"  exact moments loaded from cache ({tag})")
            return em
        em = ExactMoments(c.dim, c.v, c.e, c.L, self.feats, verbose=self.verbose)
        if cache:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(path, "wb") as fh:
                pickle.dump(em, fh)
        return em

    def moments(self, betas):
        """(pool moments, OOD-prior moments)."""
        c = self.cfg
        if self.backend == "exact":
            return self._em.reduce(("pool", betas)), self._em.reduce(("gauss", c.sigma_beta2))
        t0 = time.time()
        rng = np.random.default_rng(self.mc_seed)
        mp = mc_reduce(c, self.feats, self.mc_n, rng, betas, self.whiten, self.wscale)
        mq = mc_reduce(c, self.feats, self.mc_n, rng, None, self.whiten, self.wscale)
        if self.verbose:
            print(f"  MC moments ({self.mc_n:,} contexts per measure): {time.time()-t0:.1f}s")
        return mp, mq

    # -- eigensystem ------------------------------------------------------------------------------
    def eigsys(self, betas, rel_tol=1e-10, moments=None):
        """Invariant-sector modes (all target power) + zero-power bulk, for one task pool."""
        c = self.cfg
        d, v = c.dim, c.v
        betas = np.asarray(betas, float)
        mp, mq = self.moments(betas) if moments is None else moments
        pref = envelope_pref(c, betas, self.w, self.whiten, self.wscale)
        G, Vp, Gq, Vq = mp["G"], mp["V"], mq["G"], mq["V"]
        P = len(self.feats)
        D = np.sqrt(np.maximum(np.diag(G), 1e-300))              # standardize: scales differ by v^level
        Gs = G / np.outer(D, D)
        basis = np.zeros((0, P))                                 # orthonormal (pool) modes, standardized coords
        lam, lev, dom = [], [], []
        for lv in range(0, self.max_level + 1, 2):
            idx = np.where(self.level == lv)[0]
            E = np.zeros((len(idx), P)); E[np.arange(len(idx)), idx] = 1.0
            R = E - (basis @ Gs[:, idx]).T @ basis if len(basis) else E      # Schur against lower levels
            Gt = R @ Gs @ R.T
            Gt = 0.5 * (Gt + Gt.T)
            om = np.sqrt(self.weight[idx]) * D[idx]                          # kernel weight, standardized
            Lam, U = np.linalg.eigh(om[:, None] * Gt * om[None, :])
            keep = Lam > rel_tol * max(Lam.max(), 1e-300)
            Lam, U = Lam[keep], U[:, keep]
            modes = ((U * om[:, None]) / np.sqrt(Lam)[None, :]).T @ R
            if len(basis):                                                   # round-off clean-up
                modes = modes - (modes @ Gs @ basis.T) @ basis
            basis = np.vstack([basis, modes])
            lam.extend(pref * self.qfac * Lam)
            lev.extend([lv] * len(Lam))
            dom.extend([self.labels[idx[k]] for k in np.argmax(np.abs(U), axis=0)])
        A = basis / D[None, :]                                   # modes in raw feature coordinates
        lam, lev = np.array(lam), np.array(lev)
        norm2, sv = c.norm2, math.sqrt(v)
        v_tr = sv * (A @ Vp.T) / math.sqrt(norm2)                # (K, d): mode k x query axis b
        a_q = sv * (A @ Vq.T) / math.sqrt(norm2)
        S = A @ Gq @ A.T
        S = 0.5 * (S + S.T)
        vx = self.wscale / c.L if self.whiten else v
        out = dict(lam=lam, level=lev, dom=dom, A=A, v_tr=v_tr, a_q=a_q, S=S, pref=pref,
                   sig_p=v * float((betas ** 2).sum(1).mean()) / norm2,   # query keeps variance v
                   sig_q=v * c.sigma_beta2 * d / norm2, e=c.e / norm2, norm2=norm2, vx=vx,
                   orth_err=float(np.abs(A @ G @ A.T - np.eye(len(lam))).max()),
                   mp=mp, mq=mq, betas=betas)
        out["bulk"] = self.bulk(out)
        return out

    # -- zero-power modes -------------------------------------------------------------------------
    def bulk(self, sy):
        """Zero-target-power modes as arrays (lam, mult, Sdiag, level, tag)."""
        c = self.cfg
        d, L, w = c.dim, c.L, self.w
        pref, qf = sy["pref"], self.qfac
        mp, mq = sy["mp"], sy["mq"]
        lam, mult, sd, lv, tag = [], [], [], [], []
        # level 2, standard sector of single-position legs: (1/2w^4) N^1/2 (same - cross) N^1/2, x (L-1)
        legs2 = leg_monomials(d, 2)
        Nh = np.sqrt([_leg2_n(l) for l in legs2])
        Mp = Nh[:, None] * (mp["same"] - mp["cross"]) * Nh[None, :]
        Mq = Nh[:, None] * (mq["same"] - mq["cross"]) * Nh[None, :]
        ev, U = np.linalg.eigh(0.5 * (Mp + Mp.T))
        for k in range(len(ev)):
            if ev[k] > 1e-12 * ev.max():
                lam.append(pref * qf * ev[k] / (2 * w ** 4)); mult.append(d * (L - 1))
                sd.append(float(U[:, k] @ Mq @ U[:, k] / ev[k])); lv.append(2); tag.append("L2-standard")
        # level 2, two-position (flip-odd) sector: (1/w^4) E[Sigma (x) Sigma], x L(L-1)/2
        ev, U = np.linalg.eigh(0.5 * (mp["TT"] + mp["TT"].T))
        for k in range(len(ev)):
            if ev[k] > 1e-12 * ev.max():
                lam.append(pref * qf * ev[k] / w ** 4); mult.append(d * L * (L - 1) / 2)
                sd.append(float(U[:, k] @ mq["TT"] @ U[:, k] / ev[k])); lv.append(2); tag.append("L2-pairs")
        # levels >= 4: independent-coordinate staircase, minus the invariant modes computed exactly
        vx = sy["vx"]
        vy_p = vx * float((sy["betas"] ** 2).sum(1).mean()) + c.e
        vy_q = vx * c.sigma_beta2 * d + c.e
        Dx, Dy = L * d, L
        inv_count = {}
        for (kx, ky), lev_ in zip(self.xy, self.level):
            if lev_ >= 4:
                inv_count[(kx, ky)] = inv_count.get((kx, ky), 0) + 1
        for lev_ in range(4, self.bulk_max_level + 1, 2):
            for ky in range(lev_ + 1):
                kx = lev_ - ky
                m = math.comb(Dx + kx - 1, kx) * math.comb(Dy + ky - 1, ky) - inv_count.get((kx, ky), 0)
                if m > 0:
                    lam.append(pref * qf * (vx / w ** 2) ** kx * (vy_p / w ** 2) ** ky); mult.append(d * float(m))
                    sd.append((vy_q / vy_p) ** ky); lv.append(lev_); tag.append(f"stair x{kx}y{ky}")
        return dict(lam=np.array(lam), mult=np.array(mult), Sdiag=np.array(sd), level=np.array(lv), tag=tag)

    # -- curves -----------------------------------------------------------------------------------
    @staticmethod
    def kappa(n, lam, mult, ridge):
        """Solve sum mult * lam/(lam+k) + ridge/k = n for each n (vectorized log-bisection)."""
        n = np.atleast_1d(np.asarray(n, float))
        lo = np.full(n.shape, 1e-80); hi = np.full(n.shape, 1e6)
        for _ in range(300):
            mid = np.sqrt(lo * hi)
            f = (mult[None, :] * lam[None, :] / (lam[None, :] + mid[:, None])).sum(1) + ridge / mid - n
            lo = np.where(f > 0, mid, lo); hi = np.where(f > 0, hi, mid)
        return np.sqrt(lo * hi)

    def curves(self, sy, ns, ridge):
        """Eigenlearning ID / OOD test MSE (normalized by norm2) at sample sizes ns."""
        ns = np.atleast_1d(np.asarray(ns, float))
        lam_p, V, S, a = sy["lam"], sy["v_tr"], sy["S"], sy["a_q"]
        vte = np.linalg.solve(S, a)
        b = sy["bulk"]
        lam_all = np.concatenate([lam_p, b["lam"]])
        mult_all = np.concatenate([np.full(len(lam_p), float(self.cfg.dim)), b["mult"]])
        sd_all = np.concatenate([np.diag(S), b["Sdiag"]])
        kap = self.kappa(ns, lam_all, mult_all, ridge)
        Lall = lam_all[None, :] / (lam_all[None, :] + kap[:, None])
        E0 = ns / (ns - (mult_all[None, :] * Lall ** 2).sum(1))
        Lp = Lall[:, :len(lam_p)]
        resid_tr = sy["sig_p"] - float((V ** 2).sum())
        resid_te = sy["sig_q"] - float((a * vte).sum())
        inner = ((1 - Lp) ** 2 * (V ** 2).sum(1)[None, :]).sum(1) + resid_tr + sy["e"]
        ID = E0 * inner
        Dl = vte[None, :, :] - Lp[:, :, None] * V[None, :, :]
        bias = np.einsum("nkb,kl,nlb->n", Dl, S, Dl)
        var = (mult_all[None, :] * sd_all[None, :] * Lall ** 2).sum(1) / ns * ID
        OOD = bias + var + resid_te + sy["e"]
        return dict(n=ns, ID=ID, OOD=OOD, kappa=kap, E0=E0, L_power=Lp)

    def learnability(self, sy, ns, ridge):
        """Theoretical eigenlearning learnabilities L_i(n) = lam_i/(lam_i + kappa(n)) of the modes that
        carry target power, plus their acquisition points.

        kappa(n) is the same global scalar the learning curve uses (the whole spectrum, bulk included).
        n_half_i, the n at which L_i = 1/2, needs no root finding: L = 1/2 means kappa = lam_i, so
        n_half_i = sum_j mult_j lam_j/(lam_j + lam_i) + ridge/lam_i.
        Per data degree p, the modes are aggregated by target power, L_p(n) = sum_i p_i L_i / sum_i p_i
        with p_i = sum_b v_{i,b}^2, so a degree is "learned" when its share of the target is.
        """
        ns = np.atleast_1d(np.asarray(ns, float))
        lam_p = sy["lam"]
        b = sy["bulk"]
        lam_all = np.concatenate([lam_p, b["lam"]])
        mult_all = np.concatenate([np.full(len(lam_p), float(self.cfg.dim)), b["mult"]])
        kap = self.kappa(ns, lam_all, mult_all, ridge)
        L = lam_p[None, :] / (lam_p[None, :] + kap[:, None])                 # (len(ns), modes)
        power = (sy["v_tr"] ** 2).sum(1)                                     # normalized target power
        n_half = (mult_all[None, :] * lam_all[None, :] / (lam_all[None, :] + lam_p[:, None])).sum(1)
        if ridge:
            n_half = n_half + ridge / lam_p
        deg = sy["level"] + 1
        deg_L, deg_n_half, deg_power = {}, {}, {}
        for p in sorted(set(deg.tolist())):
            m = deg == p
            tot = float(power[m].sum())
            deg_power[p] = tot
            curve = ((L[:, m] * power[m][None, :]).sum(1) / tot) if tot > 0 else np.full(len(ns), np.nan)
            deg_L[p] = curve
            deg_n_half[p] = _half_crossing(ns, curve)
        return dict(n=ns, kappa=kap, L=L, power=power, n_half=n_half, deg=deg, lam=lam_p,
                    deg_L=deg_L, deg_n_half=deg_n_half, deg_power=deg_power,
                    learned_power=(L * power[None, :]).sum(1), total_power=float(power.sum()))

    def floors(self, sy, cut):
        """(ID, OOD) as n -> infinity when every invariant mode of level <= cut is learned, nothing above."""
        m = sy["level"] <= cut
        V, S, a = sy["v_tr"][m], sy["S"][np.ix_(m, m)], sy["a_q"][m]
        vte = np.linalg.solve(S, a)
        Dl = vte - V
        ID = sy["sig_p"] - float((V ** 2).sum()) + sy["e"]
        OOD = float(np.einsum("kb,kl,lb->", Dl, S, Dl)) + sy["sig_q"] - float((a * vte).sum()) + sy["e"]
        return ID, OOD

    def spectrum(self, sy, top):
        """The `top` largest eigenvalues with multiplicity expanded (to compare with an empirical kernel)."""
        b = sy["bulk"]
        lam = np.concatenate([sy["lam"], b["lam"]])
        mult = np.concatenate([np.full(len(sy["lam"]), self.cfg.dim), b["mult"]])
        order = np.argsort(lam)[::-1]
        out, total = [], 0
        for k in order:
            m = int(min(mult[k], top - total))
            out.append(np.full(m, lam[k])); total += m
            if total >= top:
                break
        return np.concatenate(out)


# ---------------------------------------------------------------------------
# KRR with the structured kernel
# ---------------------------------------------------------------------------

class StructuredKRR:
    """k(z,z') = k_even(c,c') (x_q.x_q') qscale, fitted with an absolute ridge on the n x n Gram."""

    def __init__(self, width, ridge, qscale, device=None, dtype=torch.float64):
        self.w, self.ridge, self.qscale = float(width), float(ridge), float(qscale)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

    @torch.no_grad()
    def gram(self, C1, Q1, C2, Q2):
        """Peak memory ~3 n1 n2 entries (in-place exps)."""
        s = 2.0 * self.w ** 2
        A = C1 @ C2.T
        A.mul_(2.0 / s)
        B = (C1 * C1).sum(1)[:, None] + (C2 * C2).sum(1)[None, :]
        B.div_(-s)
        K = torch.add(B, A).exp_()
        B.sub_(A).exp_()
        del A
        K.add_(B)
        del B
        K.mul_(Q1 @ Q2.T).mul_(0.5 * self.qscale)
        return K

    def _t(self, a):
        return torch.as_tensor(np.asarray(a), dtype=self.dtype, device=self.device)

    @torch.no_grad()
    def fit(self, C, Q, y):
        self.C, self.Q = self._t(C), self._t(Q)
        K = self.gram(self.C, self.Q, self.C, self.Q)
        y = self._t(y)[:, None]
        scale = float(K.diagonal().mean())
        for jit in (0.0, 1e-10, 1e-8, 1e-6):
            K.diagonal().add_(self.ridge + jit * scale)
            Lc, info = torch.linalg.cholesky_ex(K)
            K.diagonal().sub_(self.ridge + jit * scale)
            if int(info) == 0:
                break
        del K
        self.alpha = torch.cholesky_solve(y, Lc)
        del Lc
        return self

    @staticmethod
    def max_n(device=None, frac=0.8):
        """Largest n whose Gram + Cholesky (~3 n^2 float64 at peak) fits in free memory."""
        if device is None or getattr(device, "type", str(device)) == "cuda":
            if torch.cuda.is_available():
                free = torch.cuda.mem_get_info()[0]
                return int(math.sqrt(frac * free / (3 * 8)))
        try:
            import psutil
            return int(math.sqrt(frac * psutil.virtual_memory().available / (3 * 8)))
        except ImportError:
            return 10000

    @torch.no_grad()
    def predict(self, C, Q, batch=4096):
        C, Q = self._t(C), self._t(Q)
        out = [self.gram(C[i:i + batch], Q[i:i + batch], self.C, self.Q) @ self.alpha
               for i in range(0, C.shape[0], batch)]
        return torch.cat(out)[:, 0].cpu().numpy()


def krr_learning_curve(cfg, betas, width, ridge, ns, trials, n_test=2048, seed=0, whiten=False, wscale=None,
                       device=None, verbose=True):
    """Empirical ID / OOD test MSE of structured-kernel KRR, (trials, len(ns)) arrays, normalized by norm2."""
    qscale = 1.0 / (cfg.dim * cfg.v)
    s2 = math.sqrt(cfg.norm2)
    ID = np.full((trials, len(ns)), np.nan); OOD = np.full((trials, len(ns)), np.nan)
    t0 = time.time()
    for t in range(trials):
        rng = np.random.default_rng(seed + 1000 * t)
        te_id = sample(cfg, n_test, rng, betas, whiten, wscale)
        te_ood = sample(cfg, n_test, rng, None, whiten, wscale)
        for j, n in enumerate(ns):
            tr = sample(cfg, int(n), rng, betas, whiten, wscale)
            m = StructuredKRR(width, ridge, qscale, device).fit(context_vec(tr), tr["xq"], tr["yq"] / s2)
            for te, arr in ((te_id, ID), (te_ood, OOD)):
                pred = m.predict(context_vec(te), te["xq"])
                arr[t, j] = float(np.mean((pred - te["yq"] / s2) ** 2))
            del m
        if verbose:
            print(f"  trial {t+1}/{trials} done ({time.time()-t0:.0f}s)")
    return ID, OOD


# ---------------------------------------------------------------------------
# oracles (same sampler, so WHITEN_X applies to them too)
# ---------------------------------------------------------------------------

def dmmse_weights(cfg, betas, X, y):
    logit = -((y[:, :, None] - np.einsum("nld,td->nlt", X, betas)) ** 2).sum(1) / (2 * cfg.e)
    logit -= logit.max(1, keepdims=True)
    p = np.exp(logit)
    return (p / p.sum(1, keepdims=True)) @ betas


def oracle_mse(cfg, betas, n_mc=20000, seed=0, whiten=False, wscale=None, chunk=2000):
    """Ridge (Bayes under the N(0, sb2 I) prior) and dMMSE (Bayes under the pool) test MSE, ID and OOD."""
    rng = np.random.default_rng(seed)
    betas = np.asarray(betas, float)
    lam = cfg.e / cfg.sigma_beta2
    acc = {k: 0.0 for k in ("ridge_id", "ridge_ood", "dmmse_id", "dmmse_ood")}
    done = 0
    while done < n_mc:
        B = min(chunk, n_mc - done); done += B
        for tag, pool in (("id", betas), ("ood", None)):
            s = sample(cfg, B, rng, pool, whiten, wscale)
            X, y, xq, yq = s["X"], s["y"], s["xq"], s["yq"]
            A = np.einsum("nld,nle->nde", X, X) + lam * np.eye(cfg.dim)
            wR = np.linalg.solve(A, np.einsum("nld,nl->nd", X, y)[:, :, None])[:, :, 0]
            r = (xq * wR).sum(1) - yq
            acc["ridge_" + tag] += float(r @ r)
            r = (xq * dmmse_weights(cfg, betas, X, y)).sum(1) - yq
            acc["dmmse_" + tag] += float(r @ r)
    return {k: val / n_mc / cfg.norm2 for k, val in acc.items()}


# ---------------------------------------------------------------------------
# corrected Table 2, by data degree
# ---------------------------------------------------------------------------
# Whitened coordinates: x~ = x/sqrt(v), beta~ = sqrt(v) beta, tau = |beta~|^2 + eps^2. Query axis b.
# Degree p: F_p = all invariant features of that degree, F_< all lower ones.
#   Gamma_p = Cov(F_p) - Cov(F_p,F_<) Cov(F_<)^-1 Cov(F_<,F_p)          (Schur against lower degrees)
#   V_p     = E[y* F_p] - Cov(F_p,F_<) Cov(F_<)^-1 E[y* F_<]
#   kernel metric W_p:  W^1/2 Gamma W^1/2 = U Lam U^T,  lambda_k = pref (1/d) Lam_k,
#   v_k = U_k^T W^1/2 V_p / sqrt(Lam_k).   Kernel-free: P_p = V_p^T Gamma_p^-1 V_p = sum_k v_k^2.

def pool_stats(cfg, betas):
    """Whitened pool moments / cumulants used by the closed forms."""
    v, e = cfg.v, cfg.e
    Bt = math.sqrt(v) * np.asarray(betas, float)
    mu = Bt.mean(0)
    M2 = Bt.T @ Bt / len(Bt)
    Sig = M2 - np.outer(mu, mu)
    tau = (Bt ** 2).sum(1) + e
    Z = Bt - mu
    k3 = np.einsum("ta,tb,tc->abc", Z, Z, Z) / len(Bt)
    m4 = np.einsum("ta,tb,tc,td->abcd", Z, Z, Z, Z) / len(Bt)
    k4 = m4 - (np.einsum("ab,cd->abcd", Sig, Sig) + np.einsum("ac,bd->abcd", Sig, Sig)
               + np.einsum("ad,bc->abcd", Sig, Sig))
    return dict(Bt=Bt, mu=mu, M2=M2, Sig=Sig, tau=tau, taubar=tau.mean(), k3=k3, k4=k4,
                Tr1k3=np.einsum("aac->c", k3), Tr2k4=np.einsum("aabb->", k4),
                Etau2=float((tau ** 2).mean()), Vartau=float(tau.var()),
                Etaub=(tau[:, None] * Bt).mean(0), Covtaub=(tau[:, None] * Bt).mean(0) - tau.mean() * mu)


def closed_form_deg3(cfg, betas, width):
    """Exact (any L, any pool) degree-3 block of the structured kernel in closed form.

    Features A = sum y_i^2, B_a = sum y_i x~_ia, C_ac = sum x~_ia x~_ic (a <= c). Returns the Gram,
    the target overlaps, the kernel-free captured power P3[b], and the per-mode (lam, v) the kernel's
    metric picks -- to be compared with StructuredTheory's level-2 modes.
    """
    d, L, v = cfg.dim, cfg.L, cfg.v
    st = pool_stats(cfg, betas)
    mu, M2, Sig, taubar = st["mu"], st["M2"], st["Sig"], st["taubar"]
    pairs = [(a, c) for a in range(d) for c in range(a, d)]
    nC = len(pairs)
    n = 1 + d + nC
    G = np.zeros((n, n))
    iA, iB, iC = 0, slice(1, 1 + d), slice(1 + d, n)
    G[iA, iA] = 2 * L * st["Etau2"] + L ** 2 * st["Vartau"]
    G[iA, iB] = G[iB, iA] = 2 * L * st["Etaub"] + L ** 2 * st["Covtaub"]
    G[iB, iB] = L * (M2 + taubar * np.eye(d)) + L ** 2 * Sig
    dl = np.eye(d)
    for r, (a, c) in enumerate(pairs):
        G[iA, 1 + d + r] = G[1 + d + r, iA] = 2 * L * M2[a, c]
        for ee in range(d):
            G[1 + ee, 1 + d + r] = G[1 + d + r, 1 + ee] = L * (mu[a] * dl[c, ee] + mu[c] * dl[a, ee])
        for s, (a2, c2) in enumerate(pairs):
            G[1 + d + r, 1 + d + s] = L * (dl[a, a2] * dl[c, c2] + dl[a, c2] * dl[c, a2])
    V = np.zeros((d, n))
    V[:, iA] = L * st["Covtaub"]
    V[:, iB] = L * Sig
    P3 = np.array([V[b] @ np.linalg.solve(G, V[b]) for b in range(d)])
    # kernel metric: raw feature weights (feature_weight) times the whitening scale^2 of each feature
    om = np.zeros(n)
    om[iA] = 1 / (2 * L * width ** 4)
    om[iB] = v / (L * width ** 4)
    for r, (a, c) in enumerate(pairs):
        om[1 + d + r] = v ** 2 / ((2 if a == c else 1) * L * width ** 4)
    oh = np.sqrt(om)
    Lam, U = np.linalg.eigh(oh[:, None] * G * oh[None, :])
    keep = Lam > 1e-12 * Lam.max()
    Lam, U = Lam[keep], U[:, keep]
    coef = ((U * oh[:, None]).T @ V.T) / np.sqrt(Lam)[:, None]        # (modes, d)
    pref = envelope_pref(cfg, betas, width)
    order = np.argsort(Lam)[::-1]
    return dict(G=G, V=V, P3=P3, lam=pref * Lam[order] / d, coef=coef[order], stats=st,
                labels=["A=S[y^2]"] + [f"B{a}=S[yx{a}]" for a in range(d)] + [f"C{a}{c}" for a, c in pairs])


def table2_pdf_values(cfg, betas):
    """The PDF's Table-2 degree-3 rows, as printed, for comparison.

    (2,0): s~_a / sqrt((1 + 1/L) s~_a + taubar/L) on the eigenvalues s~ of Sigma~.
    (3,1): Tr1[k3]_b / sqrt(G31 + 2 E[tau^2]/L), with G31 = Tr2[k4] + 2 Tr(Sigma~^2) - Tr1[k3]^T Sigma~^-1 Tr1[k3]
           ("with projection", as printed) and without the last term ("no projection").
    """
    L = cfg.L
    st = pool_stats(cfg, betas)
    s = np.linalg.eigvalsh(st["Sig"])[::-1]
    v20 = s / np.sqrt((1 + 1 / L) * s + st["taubar"] / L)
    t = st["Tr1k3"]
    base = st["Tr2k4"] + 2 * np.trace(st["Sig"] @ st["Sig"])
    proj = float(t @ np.linalg.solve(st["Sig"], t))
    v31_with = t / math.sqrt(max(base - proj + 2 * st["Etau2"] / L, 1e-300))
    v31_without = t / math.sqrt(base + 2 * st["Etau2"] / L)
    return dict(v20=v20, v31_with_projection=v31_with, v31_no_projection=v31_without, s=s)


def mc_check_coeffs(th, sy, betas, n=200_000, seed=11, batch=20000, max_level=None):
    """Monte Carlo ground truth for the eigenfunctions: orthonormality, E[y psi], E[f_dMMSE psi] (Lemma 2).

    max_level restricts the check to modes of level <= max_level (their eigenfunctions only involve
    features of that level or below, so the high-degree features need not be evaluated)."""
    cfg = th.cfg
    lev_f = np.array([2 * f[0] for f in th.feats]) if hasattr(th, "feature_matrix") else th.level
    fmask = np.ones(len(lev_f), bool) if max_level is None else lev_f <= max_level
    mmask = np.ones(len(sy["lam"]), bool) if max_level is None else sy["level"] <= max_level
    if hasattr(th, "feature_matrix"):
        sub = [f for f, k in zip(th.feats, fmask) if k]
        fe = lambda s_: th.feature_matrix(s_, sub)
    else:
        fe = FeatureEval(cfg.dim, [f for f, k in zip(th.feats, fmask) if k])
    A = sy["A"][np.ix_(mmask, fmask)]
    v_ref = sy["v_tr"][mmask]
    rng = np.random.default_rng(seed)
    K, d = int(mmask.sum()), cfg.dim
    PP = np.zeros((K, K)); cy = np.zeros((K, d)); cy2 = np.zeros((K, d)); cm = np.zeros((K, d))
    done = 0
    s2 = math.sqrt(cfg.norm2 * cfg.v)
    while done < n:
        B = min(batch, n - done); done += B
        s = sample(cfg, B, rng, betas, th.whiten, th.wscale)
        psi = fe(s) @ A.T
        PP += psi.T @ psi
        ty = (s["yq"][:, None] * s["xq"]) / s2                                # target x query mode
        tm = (dmmse_predict(cfg, betas, s)[:, None] * s["xq"]) / s2
        cy += psi.T @ ty; cy2 += (psi ** 2).T @ (ty ** 2); cm += psi.T @ tm
    PP /= n; cy /= n; cm /= n
    se = np.sqrt(np.maximum(cy2 / n - cy ** 2, 0) / n)
    return dict(orth_dev=float(np.abs(PP - np.eye(K)).max()), coef_y=cy, coef_dmmse=cm, se=se,
                z_y=float(np.max(np.abs(cy - v_ref) / np.maximum(se, 1e-12))),
                z_dmmse=float(np.max(np.abs(cm - v_ref) / np.maximum(se, 1e-12))))


def ridge_predict(cfg, betas, s):
    """Ridge oracle (Bayes under the N(0, sb2 I) prior), normalized like the training target."""
    X, y = s["X"], s["y"]
    A = np.einsum("nld,nle->nde", X, X) + (cfg.e / cfg.sigma_beta2) * np.eye(cfg.dim)
    w = np.linalg.solve(A, np.einsum("nld,nl->nd", X, y)[:, :, None])[:, :, 0]
    return (s["xq"] * w).sum(1) / math.sqrt(cfg.norm2)


def oracle_projections(th, sy, betas, n_mc=200_000, seed=5, batch=20_000, max_level=None):
    """Mode coefficients and norms of the Ridge / dMMSE oracles, under the ID and the OOD measure.

    c_{i,b} = E[f_o psi_i x_qb / sqrt(v)] for the modes of data degree <= max_level+1, plus the oracle
    norms E[f_o^2]. Feeds overlap_curves; the measured overlaps (krr_overlap_curve) need none of this.
    """
    cfg = th.cfg
    fe, A, mmask = _mode_eval(th, sy, max_level)
    V = sy["v_tr"][mmask]
    sv = math.sqrt(cfg.v)
    out = dict(mask=mmask, deg=sy["level"][mmask] + 1)
    for tag, pool in (("ID", betas), ("OOD", None)):
        rng = np.random.default_rng(seed + (0 if pool is not None else 1))
        cR = np.zeros_like(V); cM = np.zeros_like(V)
        nR = nM = nRM = 0.0
        done = 0
        while done < n_mc:
            B = min(batch, n_mc - done); done += B
            s = sample(cfg, B, rng, pool, th.whiten, th.wscale)
            fR, fM = ridge_predict(cfg, betas, s), dmmse_predict(cfg, betas, s) / math.sqrt(cfg.norm2)
            psi = fe(s) @ A.T
            cR += psi.T @ (fR[:, None] * s["xq"] / sv)
            cM += psi.T @ (fM[:, None] * s["xq"] / sv)
            nR += float(fR @ fR); nM += float(fM @ fM); nRM += float(fR @ fM)
        out[tag] = dict(cR=cR / n_mc, cM=cM / n_mc, nR=nR / n_mc, nM=nM / n_mc, nRM=nRM / n_mc,
                        cos_RM=(nRM / n_mc) / math.sqrt((nR / n_mc) * (nM / n_mc)))
    return out


def overlap_curves(th, sy, proj, ns, ridge):
    """Theory cosine between the learned function and each oracle, under both measures.

    The learned function has mean coefficients L_i v_{i,b} on the (ID-orthonormal) modes, so
        <f_hat, f_o>_m = sum_{i,b} L_i v_{i,b} c^m_{i,b},
        |f_hat|^2_m    = (LV)^T G_m (LV) + (ID error / n) sum_k mult_k G_m,kk L_k^2,
    with G_m = I under the pool measure and S under the prior. The second term is the noise the fit
    picks up across the whole spectrum; it is what makes the cosines small at low n.

    The norm runs over EVERY mode. The numerators use proj's Monte Carlo coefficients, zero-padded
    above its max_level, except that under ID the dMMSE coefficients are known exactly (it is the ID
    conditional mean, so they are v_tr) and the padding uses those. Under OOD there is no such
    identity -- E_q[f_dMMSE psi] also picks up dMMSE's power at degrees the engine does not carry,
    and that cross term partly cancels the rest -- so run proj at the engine's full max_level
    whenever you can (210 features when whitened, 12,650 at degree 9 otherwise).
    """
    ns = np.atleast_1d(np.asarray(ns, float))
    cv = th.curves(sy, ns, ridge)
    m = proj["mask"]
    V, lam, S = sy["v_tr"], sy["lam"], sy["S"]
    b = sy["bulk"]
    lam_all = np.concatenate([lam, b["lam"]])
    mult_all = np.concatenate([np.full(len(lam), float(th.cfg.dim)), b["mult"]])
    sd_all = np.concatenate([np.diag(S), b["Sdiag"]])
    Lall = lam_all[None, :] / (lam_all[None, :] + cv["kappa"][:, None])
    LV = (lam[None, :] / (lam[None, :] + cv["kappa"][:, None]))[:, :, None] * V[None]
    out = dict(n=ns)
    for tag, G, sdiag in (("ID", np.eye(len(V)), np.ones(len(lam_all))), ("OOD", S, sd_all)):
        p = proj[tag]
        cR = np.zeros_like(V); cR[m] = p["cR"]                 # MC, degrees <= proj's max_level
        cM = np.zeros_like(V); cM[m] = p["cM"]
        if tag == "ID":
            cM[~m] = V[~m]                                     # exact for the modes proj skipped
        sig2 = np.einsum("nib,ij,njb->n", LV, G, LV)
        noise = cv["ID"] / ns * (mult_all[None, :] * sdiag[None, :] * Lall ** 2).sum(1)
        nrm = np.sqrt(sig2 + noise)
        out[tag] = dict(cos_ridge=np.einsum("nib,ib->n", LV, cR) / nrm / math.sqrt(p["nR"]),
                        cos_dmmse=np.einsum("nib,ib->n", LV, cM) / nrm / math.sqrt(p["nM"]),
                        norm2=sig2 + noise, signal=sig2, noise=noise, cos_RM=p["cos_RM"])
    return out


def mc_model_overlaps(cfg, betas, predict, n_mc=100_000, seed=31, batch=20_000, whiten=False, wscale=None):
    """Measured cosine between a trained predictor and the two oracles, under both measures.

    Everything is evaluated on the same samples, so this needs no eigensystem: cos = E[f_hat f_o] /
    sqrt(E[f_hat^2] E[f_o^2]). `predict(s)` returns predictions in the training units.
    """
    out = {}
    for tag, pool in (("ID", betas), ("OOD", None)):
        rng = np.random.default_rng(seed + (0 if pool is not None else 1))
        acc = dict(ff=0.0, fR=0.0, fM=0.0, RR=0.0, MM=0.0)
        done = 0
        while done < n_mc:
            B = min(batch, n_mc - done); done += B
            s = sample(cfg, B, rng, pool, whiten, wscale)
            f = predict(s)
            fR, fM = ridge_predict(cfg, betas, s), dmmse_predict(cfg, betas, s) / math.sqrt(cfg.norm2)
            acc["ff"] += float(f @ f); acc["RR"] += float(fR @ fR); acc["MM"] += float(fM @ fM)
            acc["fR"] += float(f @ fR); acc["fM"] += float(f @ fM)
        a = {k: val / n_mc for k, val in acc.items()}
        out[tag] = dict(cos_ridge=a["fR"] / math.sqrt(a["ff"] * a["RR"]),
                        cos_dmmse=a["fM"] / math.sqrt(a["ff"] * a["MM"]), norm2=a["ff"])
    return out


def krr_overlap_curve(cfg, betas, width, ridge, ns, trials=1, n_mc=100_000, seed=200, whiten=False,
                      wscale=None, device=None, verbose=True):
    """Fit the structured kernel at each n and measure its cosine with the Ridge / dMMSE oracles."""
    qscale = 1.0 / (cfg.dim * cfg.v)
    s2 = math.sqrt(cfg.norm2)
    keys = [(side, o) for side in ("ID", "OOD") for o in ("cos_ridge", "cos_dmmse")]
    out = {k: np.full((trials, len(ns)), np.nan) for k in keys}
    t0 = time.time()
    for t in range(trials):
        rng = np.random.default_rng(seed + 1000 * t)
        for j, n in enumerate(ns):
            tr = sample(cfg, int(n), rng, betas, whiten, wscale)
            mdl = StructuredKRR(width, ridge, qscale, device).fit(context_vec(tr), tr["xq"], tr["yq"] / s2)
            ov = mc_model_overlaps(cfg, betas, lambda s_: mdl.predict(context_vec(s_), s_["xq"]),
                                   n_mc=n_mc, seed=seed + 7 * t, whiten=whiten, wscale=wscale)
            for side, o in keys:
                out[(side, o)][t, j] = ov[side][o]
            del mdl
        if verbose:
            print(f"  overlap trial {t+1}/{trials} done ({time.time()-t0:.0f}s)")
    return out


def _mode_eval(th, sy, max_level):
    """(feature evaluator, mode matrix A, mode mask) restricted to modes of level <= max_level."""
    lev_f = np.array([2 * f[0] for f in th.feats]) if hasattr(th, "feature_matrix") else th.level
    fmask = np.ones(len(lev_f), bool) if max_level is None else lev_f <= max_level
    mmask = np.ones(len(sy["lam"]), bool) if max_level is None else sy["level"] <= max_level
    sub = [f for f, k in zip(th.feats, fmask) if k]
    fe = (lambda s_: th.feature_matrix(s_, sub)) if hasattr(th, "feature_matrix") else FeatureEval(th.cfg.dim, sub)
    return fe, sy["A"][np.ix_(mmask, fmask)], mmask


def mc_learnability(th, sy, betas, predict, n=100_000, seed=21, batch=10000, max_level=4):
    """Measured learnability of a trained model, mode by mode.

    `predict(s)` returns predictions on a sample dict in the training units (target y_q / sqrt(norm2)).
    The model's coefficient on mode (i, b) is c_{i,b} = E[f_hat psi_i x_qb / sqrt(v)] under the ID
    (pool) measure, so L_i = sum_b c_{i,b} v_{i,b} / sum_b v_{i,b}^2 is the eigenlearning learnability
    of that mode, to be compared with lam_i / (lam_i + kappa(n)).
    """
    cfg = th.cfg
    fe, A, mmask = _mode_eval(th, sy, max_level)
    v_ref = sy["v_tr"][mmask]
    rng = np.random.default_rng(seed)
    K, d = A.shape[0], cfg.dim
    c = np.zeros((K, d)); c2 = np.zeros((K, d))
    s2 = math.sqrt(cfg.v)
    done = 0
    while done < n:
        B = min(batch, n - done); done += B
        s = sample(cfg, B, rng, betas, th.whiten, th.wscale)
        psi = fe(s) @ A.T
        tg = predict(s)[:, None] * s["xq"] / s2
        c += psi.T @ tg; c2 += (psi ** 2).T @ (tg ** 2)
    c /= n
    se = np.sqrt(np.maximum(c2 / n - c ** 2, 0) / n)
    pw = (v_ref ** 2).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        Lmode = np.where(pw > 0, (c * v_ref).sum(1) / np.maximum(pw, 1e-300), np.nan)
        Lse = np.where(pw > 0, np.sqrt(((se * v_ref) ** 2).sum(1)) / np.maximum(pw, 1e-300), np.nan)
    deg = sy["level"][mmask] + 1
    deg_L = {}
    for p in sorted(set(deg.tolist())):
        m = deg == p
        tot = float(pw[m].sum())
        deg_L[p] = float((c[m] * v_ref[m]).sum() / tot) if tot > 0 else float("nan")
    return dict(coef=c, se=se, L=Lmode, L_se=Lse, power=pw, deg=deg, deg_L=deg_L, mask=mmask)


def dmmse_predict(cfg, betas, s):
    return (s["xq"] * dmmse_weights(cfg, betas, s["X"], s["y"])).sum(1)
