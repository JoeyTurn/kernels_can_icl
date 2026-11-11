"""O(L)-invariant eigen-theory for the structured kernel -- reaches the degree-9 floor.

Why it is exact. Write the context as the L x (d+1) matrix Z = [X y]. Then
  * the kernel only sees c.c' = tr(Z^T Z') and |c|^2 = tr(Z^T Z), so it is invariant under Z -> QZ,
    Z' -> QZ' for any Q in O(L) (not only position permutations and sign flips);
  * the data law is O(L)-invariant too: given beta the rows are iid Gaussian, so the density depends on
    Z only through M = Z^T Z; with WHITEN_X, X is Haar on the Stiefel manifold and y = X beta + eta;
  * the target E[y_q | Z] depends on Z only through M (sufficient statistic), for both measures.
So every eigenfunction that carries target power, ID or OOD, is an O(L)-invariant function, i.e. a
polynomial in the (d+1)(d+2)/2 entries of M (first fundamental theorem of invariant theory); the
level-2k ones are polynomials of degree k. Other O(L) isotypic components carry no power and do not
couple through S.

Pieces:
  * basis: monomials in the entries of M (variables = unordered coordinate pairs, y is coordinate d);
  * Gram / target overlaps: moments of M under the pool and the prior.
      Gaussian x: M | beta ~ Wishart(L, Sigma_beta); all moments by the Stein recursion
        E[W_ij f] = L S_ij E[f] + sum_{(a,b) in f} e_ab (S_ia E[f W_jb / W_ab] + S_ib E[f W_ja / W_ab]),
      averaged over the pool (ID) or tensor Gauss-Hermite nodes of N(0, sb2 I) (OOD, exact).
      WHITEN_X: W = cI, S = X^T y ~ N(c beta, c eps^2 I), A = |S|^2/c + eps^2 chi^2_{L-d}: closed form.
  * kernel: the level-2k term (c.c')^2k / ((2k)! w^4k), averaged over O(L), is
        kappa_2k(M, M') = sum_{kappa |- k} C_kappa(M M') / (k! 4^k w^4k (L/2)_kappa)
    with C_kappa the zonal polynomials (Muirhead Thm 7.4.1: int_O(n) etr(AH) dH = 0F1(n/2; AA^T/4)).
"""
import itertools
import math
import os
import pickle
import time
from fractions import Fraction

import numpy as np
import torch

import kicl_floor as KF
import structured_theory as ST

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_HERE, "results", "ol_cache")


# ---------------------------------------------------------------------------
# zonal polynomials in the power-sum basis (exact rationals, k <= 6)
# ---------------------------------------------------------------------------

def partitions(k, maxpart=None):
    if k == 0:
        yield ()
        return
    maxpart = k if maxpart is None else maxpart
    for p in range(min(k, maxpart), 0, -1):
        for rest in partitions(k - p, p):
            yield (p,) + rest


def _z(lam):
    out = 1
    for i in set(lam):
        m = lam.count(i)
        out *= i ** m * math.factorial(m)
    return out


def _p_to_m(k):
    """Matrix P[lam][mu] with p_lam = sum_mu P m_mu (monomial symmetric functions), exact integers."""
    parts = list(partitions(k))
    idx = {mu: i for i, mu in enumerate(parts)}
    P = [[0] * len(parts) for _ in parts]
    for li, lam in enumerate(parts):
        # expand prod_i (sum_j x_j^{lam_i}) in k variables; coefficient of x^mu (mu sorted, padded)
        counts = {}
        for assign in itertools.product(range(k), repeat=len(lam)):
            e = [0] * k
            for part, var in zip(lam, assign):
                e[var] += part
            key = tuple(sorted([x for x in e if x], reverse=True))
            counts[(key, tuple(e))] = counts.get((key, tuple(e)), 0) + 1
        for (mu, e), c in counts.items():
            if list(e) == sorted(e, reverse=True):          # read the coefficient at the sorted exponent
                P[li][idx[mu]] += c
    return parts, P


def zonal_in_power_sums(k):
    """{kappa: {lam: Fraction}} with C_kappa = sum_lam coef p_lam, normalized so sum_kappa C_kappa = p_1^k."""
    parts, P = _p_to_m(k)
    n = len(parts)
    import sympy
    Pm = sympy.Matrix(P)
    M2P = Pm.inv()                                  # m_mu = sum_lam M2P[mu, lam] p_lam
    ip = lambda a, b: sum(a[i] * b[i] * _z(parts[i]) * 2 ** len(parts[i]) for i in range(n))
    order = sorted(range(n), key=lambda i: parts[i][0] if False else _dom_key(parts[i]))
    J = {}
    done = []
    for i in order:                                 # increasing dominance: orthogonalize m_kappa against lower J
        vec = [sympy.Rational(M2P[i, j]) for j in range(n)]
        for jj in done:
            coef = ip(vec, J[jj]) / ip(J[jj], J[jj])
            vec = [vec[t] - coef * J[jj][t] for t in range(n)]
        J[i] = vec
        done.append(i)
    # normalize: sum_kappa s_kappa J_kappa = p_1^k
    target = [sympy.Integer(1) if parts[t] == (1,) * k else sympy.Integer(0) for t in range(n)]
    A = sympy.Matrix([[J[i][t] for i in range(n)] for t in range(n)])
    s = A.solve(sympy.Matrix(target))
    return {parts[i]: {parts[t]: Fraction(str(sympy.nsimplify(s[i] * J[i][t]))) for t in range(n)}
            for i in range(n)}


def _dom_key(p):
    """A linear extension of dominance order (partial sums, lexicographic) -- total for k <= 5."""
    return tuple(itertools.accumulate(p))


def gen_pochhammer(a, kappa):
    """(a)_kappa for real zonal polynomials: prod_i (a - (i-1)/2)_{kappa_i}."""
    out = 1.0
    for i, ki in enumerate(kappa):
        b = a - i / 2.0
        for r in range(ki):
            out *= b + r
    return out


def level_kernel_power_sums(k, L, w):
    """kappa_2k(M, M') = sum_lam b_lam p_lam(M M')   ->   {lam: b_lam} (floats)."""
    Z = zonal_in_power_sums(k)
    out = {}
    for kap, poly in Z.items():
        pre = 1.0 / (math.factorial(k) * 4 ** k * w ** (4 * k) * gen_pochhammer(L / 2.0, kap))
        for lam, c in poly.items():
            out[lam] = out.get(lam, 0.0) + pre * float(c)
    return out


# ---------------------------------------------------------------------------
# monomials in the entries of M
# ---------------------------------------------------------------------------

class Monomials:
    """Monomials of degree <= D in nv variables, ranked by the combinadic of their sorted variable list.

    A degree-m monomial is a sorted list v_1 <= ... <= v_m; c_i = v_i + i - 1 is strictly increasing and
    rank = sum_i C(c_i, i). Ranks are contiguous 0 .. C(nv+m-1, m)-1 within each degree.
    """

    def __init__(self, nv, D):
        self.nv, self.D = nv, D
        self.B = np.array([[math.comb(n, r) for r in range(D + 2)] for n in range(nv + D + 2)], dtype=np.int64)
        self.lists = []
        for m in range(D + 1):
            if m == 0:
                self.lists.append(np.zeros((1, 0), dtype=np.int64))
                continue
            arr = np.array(list(itertools.combinations_with_replacement(range(nv), m)), dtype=np.int64)
            r = self.rank(arr)
            out = np.empty_like(arr)
            out[r] = arr
            self.lists.append(out)
        self.count = [len(l) for l in self.lists]

    def rank(self, arr):
        """arr (n, m) sorted rows -> rank within degree m."""
        if arr.shape[1] == 0:
            return np.zeros(len(arr), dtype=np.int64)
        c = arr + np.arange(arr.shape[1])[None, :]
        return self.B[c, np.arange(1, arr.shape[1] + 1)[None, :]].sum(1)

    def exps(self, m):
        e = np.zeros((self.count[m], self.nv), dtype=np.int64)
        for j in range(m):
            np.add.at(e, (np.arange(self.count[m]), self.lists[m][:, j]), 1)
        return e


def pair_vars(d):
    """Variables of M: unordered coordinate pairs (a <= b), coordinates 0..d-1 = x, d = y."""
    pv = [(a, b) for a in range(d + 1) for b in range(a, d + 1)]
    vid = {}
    for i, (a, b) in enumerate(pv):
        vid[(a, b)] = vid[(b, a)] = i
    return pv, vid


# ---------------------------------------------------------------------------
# Wishart moments by the Stein recursion (vectorized over nodes, torch)
# ---------------------------------------------------------------------------

class WishartRecursion:
    """Index tables for E[monomial(W)] under Wishart(L, Sigma), all degrees <= D (beta-independent)."""

    def __init__(self, d, D, verbose=True):
        t0 = time.time()
        self.d, self.D = d, D
        self.pv, self.vid = pair_vars(d)
        nv = len(self.pv)
        self.mono = Monomials(nv, D)
        nc = d + 1
        vid = np.full((nc, nc), -1, dtype=np.int64)
        for (a, b), i in self.vid.items():
            vid[a, b] = i
        pa = np.array([p[0] for p in self.pv]); pb = np.array([p[1] for p in self.pv])
        self.tables = [None]
        for m in range(1, D + 1):
            V = self.mono.lists[m]                       # (n, m) sorted var ids
            first = V[:, 0]
            i_, j_ = pa[first], pb[first]
            rest = V[:, 1:]                              # f = h / W_ij (still sorted)
            parent = self.mono.rank(rest)
            lead_sig = i_ * nc + j_                      # L * Sigma_ij * E[f]
            slots = []
            for s in range(m - 1):
                ab = rest[:, s]
                a, b = pa[ab], pb[ab]
                others = np.delete(rest, s, axis=1)
                for (sig_r, sig_c, new_a, new_b) in ((i_, a, j_, b), (i_, b, j_, a)):
                    nvv = vid[new_a, new_b]
                    lst = np.sort(np.concatenate([others, nvv[:, None]], axis=1), axis=1)
                    slots.append((sig_r * nc + sig_c, self.mono.rank(lst)))
            self.tables.append(dict(parent=parent, lead=lead_sig, slots=slots))
        if verbose:
            print(f"  Wishart recursion tables: d={d}, degree <= {D}, "
                  f"{sum(self.mono.count):,} monomials, {time.time()-t0:.1f}s")

    def moments(self, Sig, L, weights, extra=None, chunk=256, device=None, dtype=torch.float64):
        """sum_nodes weights * E[mono(W) | Sigma_node] for every monomial, per degree.

        Sig (N, nc, nc); weights (N,); extra (N, q): also return sum_nodes weights*extra[:, j]*E[...] for
        degrees <= D (used for the target overlaps E[beta_b g]).
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        N, nc = Sig.shape[0], self.d + 1
        tabs = [None] + [dict(parent=torch.as_tensor(t["parent"], device=device),
                              lead=torch.as_tensor(t["lead"], device=device),
                              slots=[(torch.as_tensor(a, device=device), torch.as_tensor(b, device=device))
                                     for a, b in t["slots"]])
                         for t in self.tables[1:]]
        acc = [torch.zeros(c, dtype=dtype, device=device) for c in self.mono.count]
        accx = None if extra is None else [torch.zeros(c, extra.shape[1], dtype=dtype, device=device)
                                            for c in self.mono.count]
        for s0 in range(0, N, chunk):
            S = torch.as_tensor(Sig[s0:s0 + chunk].reshape(-1, nc * nc), dtype=dtype, device=device).T  # (nc^2, n)
            wv = torch.as_tensor(weights[s0:s0 + chunk], dtype=dtype, device=device)
            ex = None if extra is None else torch.as_tensor(extra[s0:s0 + chunk], dtype=dtype, device=device)
            prev = torch.ones(1, S.shape[1], dtype=dtype, device=device)
            acc[0] += prev @ wv
            if ex is not None:
                accx[0] += (prev * wv) @ ex
            for m in range(1, self.D + 1):
                t = tabs[m]
                cur = L * S[t["lead"]] * prev[t["parent"]]
                for sig, tgt in t["slots"]:
                    cur += S[sig] * prev[tgt]
                acc[m] += cur @ wv
                if ex is not None:
                    accx[m] += (cur * wv) @ ex
                prev = cur
        return [a.cpu().numpy() for a in acc], (None if extra is None else [a.cpu().numpy() for a in accx])


# ---------------------------------------------------------------------------
# whitened moments (closed form): W = cI, S ~ N(m, s2 I) per node, A = |S|^2/c + eps^2 chi^2_nu
# ---------------------------------------------------------------------------

def _sa_moments(d, D, c, eps2, nu, means, s2):
    """E[S^alpha A^j | node] for every (S, A)-monomial of degree <= D; variables S_0..S_{d-1}, A (= index d).

    S ~ N(means, s2 I) per node, A = |S|^2/c + eps2 chi^2_nu, chi independent.
    Returns (Monomials over d+1 vars, list per degree of (n_m, N) arrays).
    """
    sa = Monomials(d + 1, D)
    N = means.shape[0]
    maxp = 3 * D + 2
    mu = [np.ones((N, d)), means.copy()]
    for p in range(2, maxp + 1):
        mu.append(means * mu[p - 1] + (p - 1) * s2 * mu[p - 2])
    mu = np.stack(mu)                                            # (maxp+1, N, d)
    chi = [1.0]
    for p in range(1, D + 1):
        chi.append(chi[-1] * (nu + 2 * (p - 1)))
    out = []
    for m in range(D + 1):
        E = sa.exps(m) if m else np.zeros((1, d + 1), dtype=np.int64)
        vals = np.zeros((len(E), N))
        for r, e in enumerate(E):
            alpha, j = e[:d], int(e[d])
            val = np.zeros(N)
            for i in range(j + 1):                               # A^j = sum_i C(j,i) (|S|^2/c)^i (eps2 chi)^(j-i)
                dp = np.zeros((i + 1, N)); dp[0] = 1.0            # E[S^alpha |S|^{2i}] by a DP over coordinates
                for a in range(d):
                    nd = np.zeros_like(dp)
                    for q in range(i + 1):
                        for qa in range(i - q + 1):
                            nd[q + qa] += dp[q] * mu[alpha[a] + 2 * qa, :, a] / math.factorial(qa)
                    dp = nd
                val += math.comb(j, i) * c ** (-i) * eps2 ** (j - i) * chi[j - i] * math.factorial(i) * dp[i]
            vals[r] = val
        out.append(vals)
    return sa, out


def whitened_moments(mono, d, D, c, eps2, nu, means, s2, weights, extra=None):
    """Moments of every M-monomial (degree <= D) under the whitened law, same format as
    WishartRecursion.moments. W = cI: a monomial with an off-diagonal W entry vanishes, each diagonal W
    entry contributes a factor c, and the rest is an (S, A)-monomial."""
    pv, vid = pair_vars(d)
    sa, samom = _sa_moments(d, D, c, eps2, nu, means, s2)
    red = [samom[m] @ weights for m in range(D + 1)]
    redx = None if extra is None else [samom[m] @ (weights[:, None] * extra) for m in range(D + 1)]
    kind = np.zeros(len(pv), dtype=np.int64)                     # 0 diag W, 1 offdiag W, 2 S_a, 3 A
    samap = np.full(len(pv), -1, dtype=np.int64)
    for i, (a, b) in enumerate(pv):
        if b < d:
            kind[i] = 0 if a == b else 1
        elif a < d:
            kind[i] = 2; samap[i] = a
        else:
            kind[i] = 3; samap[i] = d
    acc, accx = [], []
    for m in range(D + 1):
        V = mono.lists[m]
        if m == 0:
            acc.append(red[0].copy()); accx.append(None if extra is None else redx[0].copy())
            continue
        kk = kind[V]
        zero = (kk == 1).any(1)
        wdeg = (kk == 0).sum(1)
        sav = np.where(kk >= 2, samap[V], np.iinfo(np.int64).max)
        sav.sort(1)
        sdeg = (kk >= 2).sum(1)
        out = np.zeros(len(V)); outx = None if extra is None else np.zeros((len(V), extra.shape[1]))
        for q in range(m + 1):
            rows = np.where((sdeg == q) & ~zero)[0]
            if not len(rows):
                continue
            r = sa.rank(sav[rows, :q]) if q else np.zeros(len(rows), dtype=np.int64)
            fac = float(c) ** wdeg[rows]
            out[rows] = fac * red[q][r]
            if extra is not None:
                outx[rows] = fac[:, None] * redx[q][r]
        acc.append(out); accx.append(outx)
    return acc, (accx if extra is not None else None)


# ---------------------------------------------------------------------------
# Gauss-Hermite nodes for N(0, sb2 I_d)
# ---------------------------------------------------------------------------

def gh_nodes(d, q, sb2):
    x, w = np.polynomial.hermite_e.hermegauss(q)                 # probabilists': weight exp(-x^2/2)
    w = w / w.sum()
    grids = np.meshgrid(*([x] * d), indexing="ij")
    nodes = np.stack([g.ravel() for g in grids], 1) * math.sqrt(sb2)
    wg = np.meshgrid(*([w] * d), indexing="ij")
    weights = np.prod(np.stack([g.ravel() for g in wg], 1), 1)
    return nodes, weights


def sigma_nodes(betas, v, e):
    """Per-node (x, y) covariance [[v I, v beta], [v beta^T, v|beta|^2 + e]], y last."""
    N, d = betas.shape
    S = np.zeros((N, d + 1, d + 1))
    S[:, :d, :d] = v * np.eye(d)
    S[:, :d, d] = S[:, d, :d] = v * betas
    S[:, d, d] = v * (betas ** 2).sum(1) + e
    return S


# ---------------------------------------------------------------------------
# the kernel on M-monomials
# ---------------------------------------------------------------------------

def level_omega(mono, d, k, L, w):
    """Dense Omega_k over degree-k monomials: kappa_2k(M,M') = sum_{a,b} g_a(M) Omega[a,b] g_b(M')."""
    pv, vid = pair_vars(d)
    nc = d + 1
    vidm = np.full((nc, nc), -1, dtype=np.int64)
    for (a, b), i in vid.items():
        vidm[a, b] = i
    coef = level_kernel_power_sums(k, L, w)
    n = mono.count[k]
    Om = np.zeros(n * n)
    for lam, blam in coef.items():
        if blam == 0.0:
            continue
        # p_lam = prod_parts tr((M M')^j): indices i_1..i_{2j} per part, M between (2t, 2t+1), M' between (2t+1, 2t+2)
        nidx = 2 * sum(lam)
        for idx in _index_blocks(nc, nidx):
            Mv, Mpv, off = [], [], 0
            for j in lam:
                cyc = idx[:, off:off + 2 * j]
                for t in range(j):
                    Mv.append(vidm[cyc[:, 2 * t], cyc[:, 2 * t + 1]])
                    Mpv.append(vidm[cyc[:, 2 * t + 1], cyc[:, (2 * t + 2) % (2 * j)]])
                off += 2 * j
            ra = mono.rank(np.sort(np.stack(Mv, 1), 1))
            rb = mono.rank(np.sort(np.stack(Mpv, 1), 1))
            Om += blam * np.bincount(ra * n + rb, minlength=n * n)
    return Om.reshape(n, n)


def _index_blocks(nc, nidx, block=2_000_000):
    total = nc ** nidx
    for s in range(0, total, block):
        r = np.arange(s, min(total, s + block), dtype=np.int64)
        out = np.empty((len(r), nidx), dtype=np.int64)
        for p in range(nidx - 1, -1, -1):
            out[:, p] = r % nc
            r //= nc
        yield out


# ---------------------------------------------------------------------------
# the eigensystem on the O(L)-invariant sector
# ---------------------------------------------------------------------------

class OLTheory(ST.StructuredTheory):
    """Drop-in replacement for StructuredTheory whose power-carrying basis is polynomials in M = Z^T Z.

    max_level = 2K: invariant functions through degree K in M, i.e. data degree 2K + 1 (8 -> degree 9).
    """

    def __init__(self, cfg, width, max_level=8, whiten=False, wscale=None, gh_q=None, bulk_max_level=80,
                 chunk=128, device=None, cache=True, verbose=True):
        self.cfg, self.w, self.max_level = cfg, float(width), int(max_level)
        self.K = self.max_level // 2
        self.whiten = bool(whiten)
        self.wscale = (cfg.L * cfg.v if wscale is None else float(wscale)) if whiten else None
        self.bulk_max_level, self.verbose, self.chunk = bulk_max_level, verbose, chunk
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gh_q = gh_q or (2 * self.K + 1)
        d, L = cfg.dim, cfg.L
        self.qfac = 1.0 / d
        self.pv, self.vid = pair_vars(d)
        self.use_cache = cache
        t0 = time.time()
        if self.whiten:                                # moments are closed form; only the index is needed
            self.rec = None
            self.mono = self._cached(f"mono_d{d}_D{2 * self.K}.pkl", lambda: Monomials(len(self.pv), 2 * self.K))
        else:
            self.rec = self._cached(f"wish_d{d}_D{2 * self.K}.pkl",
                                    lambda: WishartRecursion(d, 2 * self.K, verbose))
            self.mono = self.rec.mono
        self.all_feats = [(m, r) for m in range(self.K + 1) for r in range(self.mono.count[m])]
        if self.whiten:
            # W = cI: a monomial containing any W entry is c^w times a lower-degree one, so the Schur step
            # against lower levels removes it exactly -- keep the (S, A)-monomials only.
            isW = np.array([b_ < d for (_, b_) in self.pv])
            self.all_feats = [(m, r) for m, r in self.all_feats if not isW[self.mono.lists[m][r]].any()]
        self.omega = {}
        for k in range(1, self.K + 1):
            key = f"omega_d{d}_k{k}_L{L}_w{self.w:.8g}.npy"
            self.omega[k] = self._cached(key, lambda k=k: level_omega(self.mono, d, k, L, self.w), npy=True)
        if verbose:
            print(f"  O(L) engine ready: degree <= {self.K} in M ({sum(self.mono.count[:self.K + 1]):,} functions, "
                  f"data degree {2 * self.K + 1}), {time.time() - t0:.1f}s")

    def _cached(self, name, fn, npy=False):
        path = os.path.join(CACHE_DIR, name)
        if self.use_cache and os.path.exists(path):
            if npy:
                return np.load(path)
            with open(path, "rb") as fh:
                return pickle.load(fh)
        obj = fn()
        if self.use_cache:
            os.makedirs(CACHE_DIR, exist_ok=True)
            if npy:
                np.save(path, obj)
            else:
                with open(path, "wb") as fh:
                    pickle.dump(obj, fh)
        return obj

    # -- moments -----------------------------------------------------------------------------------
    def _moment_tables(self, betas):
        """(pool, prior) tables: mom[m] (n_m,) and momx[m] (n_m, d) = E[beta_b * monomial]."""
        c, d, L = self.cfg, self.cfg.dim, self.cfg.L
        D = 2 * self.K
        t0 = time.time()
        if not self.whiten:
            wp = np.full(len(betas), 1.0 / len(betas))
            mp = self.rec.moments(sigma_nodes(betas, c.v, c.e), L, wp, extra=betas, chunk=self.chunk,
                                  device=self.device)
            nodes, wq = gh_nodes(d, self.gh_q, c.sigma_beta2)
            mq = self.rec.moments(sigma_nodes(nodes, c.v, c.e), L, wq, extra=nodes, chunk=self.chunk,
                                  device=self.device)
            if self.verbose:
                print(f"  Wishart moments: pool {len(betas)} tasks + prior {len(nodes):,} Gauss-Hermite nodes, "
                      f"{time.time() - t0:.1f}s")
            return mp, mq
        cw, nu = self.wscale, L - d
        wp = np.full(len(betas), 1.0 / len(betas))
        mp = whitened_moments(self.mono, d, D, cw, c.e, nu, cw * betas, cw * c.e, wp, extra=betas)
        # prior: S ~ N(0, (c^2 sb2 + c eps2) I) and E[beta | S] = c sb2 S / (c^2 sb2 + c eps2)
        s2q = cw * cw * c.sigma_beta2 + cw * c.e
        mq0, _ = whitened_moments(self.mono, d, D, cw, c.e, nu, np.zeros((1, d)), s2q, np.ones(1))
        k_s = cw * c.sigma_beta2 / s2q
        mqx = []
        for m in range(D):
            V = self.mono.lists[m]
            out = np.zeros((len(V), d))
            for b in range(d):
                lst = np.sort(np.concatenate([V, np.full((len(V), 1), self.vid[(b, d)])], 1), 1)
                out[:, b] = k_s * mq0[m + 1][self.mono.rank(lst)]
            mqx.append(out)
        mqx.append(np.zeros((self.mono.count[D], d)))
        if self.verbose:
            print(f"  whitened moments (closed form): {time.time() - t0:.1f}s")
        return mp, (mq0, mqx)

    def _gram(self, mom, momx, feats):
        """G[i,j] = E[g_i g_j] and V[b, i] = E[beta_b g_i] for a list of (degree, rank) features."""
        P = len(feats)
        degs = np.array([m for m, _ in feats]); ranks = np.array([r for _, r in feats])
        G = np.zeros((P, P))
        for mi in range(self.K + 1):
            for mj in range(mi, self.K + 1):
                I = np.where(degs == mi)[0]; J = np.where(degs == mj)[0]
                if not len(I) or not len(J):
                    continue
                A = self.mono.lists[mi][ranks[I]]; B = self.mono.lists[mj][ranks[J]]
                step = max(1, 4_000_000 // max(len(J), 1))
                for s in range(0, len(I), step):
                    a = A[s:s + step]
                    merged = np.concatenate([np.repeat(a, len(J), 0), np.tile(B, (len(a), 1))], 1)
                    merged.sort(1)
                    val = mom[mi + mj][self.mono.rank(merged)].reshape(len(a), len(J))
                    G[np.ix_(I[s:s + step], J)] = val
                    G[np.ix_(J, I[s:s + step])] = val.T
        V = np.zeros((self.cfg.dim, P))
        for i, (m, r) in enumerate(feats):
            V[:, i] = momx[m][r]
        return G, V

    # -- eigensystem ---------------------------------------------------------------------------------
    def eigsys(self, betas, rel_tol=1e-11):
        c = self.cfg
        d, v = c.dim, c.v
        betas = np.asarray(betas, float)
        (mom_p, momx_p), (mom_q, momx_q) = self._moment_tables(betas)
        t0 = time.time()
        feats = self.all_feats
        G, Vp = self._gram(mom_p, momx_p, feats)
        keepf = np.diag(G) > 1e-300                          # whitened: monomials with off-diagonal W vanish
        feats = [f for f, k in zip(feats, keepf) if k]
        G, Vp = G[np.ix_(keepf, keepf)], Vp[:, keepf]
        Gq, Vq = self._gram(mom_q, momx_q, feats)
        lev = np.array([2 * m for m, _ in feats])
        pref = ST.envelope_pref(c, betas, self.w, self.whiten, self.wscale)
        P = len(feats)
        dev = self.device
        T_ = lambda a: torch.as_tensor(a, dtype=torch.float64, device=dev)
        D = np.sqrt(np.diag(G))
        Gs = T_(G / np.outer(D, D))
        basis = torch.zeros((0, P), dtype=torch.float64, device=dev)
        lam, levs, dom = [], [], []
        for k in range(0, self.K + 1):
            idx = np.where(lev == 2 * k)[0]
            it = torch.as_tensor(idx, device=dev)
            if len(basis):
                R = -(basis @ Gs[:, it]).T @ basis
                R[torch.arange(len(idx), device=dev), it] += 1.0
            else:
                R = torch.zeros((len(idx), P), dtype=torch.float64, device=dev)
                R[torch.arange(len(idx), device=dev), it] = 1.0
            Gt = R @ (Gs @ R.T)
            Gt = 0.5 * (Gt + Gt.T)
            if k == 0:
                Om = np.ones((1, 1))
            else:
                ranks = np.array([feats[i][1] for i in idx])
                Om = self.omega[k][np.ix_(ranks, ranks)]
            Oms = T_(D[idx][:, None] * Om * D[idx][None, :])
            ev, Uo = torch.linalg.eigh(0.5 * (Oms + Oms.T))
            Oh = (Uo * torch.sqrt(torch.clamp(ev, min=0))) @ Uo.T
            Lam, U = torch.linalg.eigh(Oh @ Gt @ Oh)
            keep = Lam > rel_tol * max(float(Lam.max()), 1e-300)
            Lam, U = Lam[keep], U[:, keep]
            OhU = Oh @ U
            modes = (OhU / torch.sqrt(Lam)[None, :]).T @ R
            if len(basis):
                modes = modes - (modes @ Gs @ basis.T) @ basis
            basis = torch.cat([basis, modes])
            Lnp = Lam.cpu().numpy()
            lam.extend(pref * self.qfac * Lnp)
            levs.extend([2 * k] * len(Lnp))
            dom.extend([self._label(feats[idx[j]]) for j in torch.argmax(torch.abs(OhU), dim=0).cpu().numpy()])
            del R, Gt, Oms, Oh, OhU, U
        basis = basis.cpu().numpy()
        A = basis / D[None, :]
        lam, levs = np.array(lam), np.array(levs)
        norm2, sv = c.norm2, math.sqrt(v)
        v_tr = sv * (A @ Vp.T) / math.sqrt(norm2)
        a_q = sv * (A @ Vq.T) / math.sqrt(norm2)
        At = T_(A)
        S = (At @ T_(Gq) @ At.T).cpu().numpy()
        S = 0.5 * (S + S.T)
        orth = float((At @ T_(G) @ At.T - torch.eye(len(lam), dtype=torch.float64, device=dev)).abs().max())
        del At
        self.feats = feats
        vx = self.wscale / c.L if self.whiten else v
        out = dict(lam=lam, level=levs, dom=dom, A=A, v_tr=v_tr, a_q=a_q, S=S, pref=pref,
                   sig_p=v * float((betas ** 2).sum(1).mean()) / norm2, sig_q=v * c.sigma_beta2 * d / norm2,
                   e=c.e / norm2, norm2=norm2, vx=vx,
                   orth_err=orth, betas=betas, feats=feats)
        out["bulk"] = self.bulk(out)
        if self.verbose:
            print(f"  O(L) eigensystem: {len(lam)} modes, orthonormality err {out['orth_err']:.1e}, "
                  f"{time.time() - t0:.1f}s")
        return out

    def _label(self, f):
        m, r = f
        names = []
        for vv in self.mono.lists[m][r]:
            a, b = self.pv[vv]
            na = "y" if a == self.cfg.dim else f"x{a}"
            nb = "y" if b == self.cfg.dim else f"x{b}"
            names.append(f"M[{na}{nb}]")
        return "*".join(names) or "1"

    def _feat_xy(self, f):
        m, r = f
        kx = ky = 0
        for vv in self.mono.lists[m][r]:
            a, b = self.pv[vv]
            kx += int(a < self.cfg.dim) + int(b < self.cfg.dim)
            ky += int(a == self.cfg.dim) + int(b == self.cfg.dim)
        return kx, ky

    def _leg_to_pair(self, leg):
        px, py = leg
        idx = [a for a, p in enumerate(px) for _ in range(p)] + [self.cfg.dim] * py
        return tuple(sorted(idx))

    # -- zero-power modes -----------------------------------------------------------------------------
    def bulk(self, sy):
        """Level 2 exact (standard + pair sectors); levels >= 4 staircase minus the exact O(L)-invariant set."""
        c = self.cfg
        d, L, w = c.dim, c.L, self.w
        pref, qf = sy["pref"], self.qfac
        lam, mult, sd, lv, tag = [], [], [], [], []
        pairs = [(a, b) for a in range(d + 1) for b in range(a, d + 1)]
        nh = np.sqrt([1.0 if a == b else 2.0 for a, b in pairs])
        nc = d + 1
        if not self.whiten:
            def sectors(Sig, wts):
                pa = np.array([p[0] for p in pairs]); pb = np.array([p[1] for p in pairs])
                s_ab = Sig[:, pa, pb]                                          # (N, np)
                cross = np.einsum("n,ni,nj->ij", wts, s_ab, s_ab)
                same = cross + np.einsum("n,nij->ij", wts,
                                         Sig[:, pa[:, None], pa[None, :]] * Sig[:, pb[:, None], pb[None, :]]
                                         + Sig[:, pa[:, None], pb[None, :]] * Sig[:, pb[:, None], pa[None, :]])
                TT = np.einsum("n,nac,nbd->abcd", wts, Sig, Sig).reshape(nc * nc, nc * nc)
                return same, cross, TT
            betas = sy["betas"]
            sp = sectors(sigma_nodes(betas, c.v, c.e), np.full(len(betas), 1.0 / len(betas)))
            nodes, wq = gh_nodes(d, 5, c.sigma_beta2)                           # beta-degree 4: exact
            sq = sectors(sigma_nodes(nodes, c.v, c.e), wq)
        else:
            rng = np.random.default_rng(0)
            mpp = ST.mc_reduce(c, [()], 200_000, rng, sy["betas"], True, self.wscale)
            mqq = ST.mc_reduce(c, [()], 200_000, rng, None, True, self.wscale)
            order = [self._leg_to_pair(l) for l in ST.leg_monomials(d, 2)]
            perm = np.array([order.index(p) for p in pairs])                  # pair i <- leg perm[i]
            sp = (mpp["same"][np.ix_(perm, perm)], mpp["cross"][np.ix_(perm, perm)], mpp["TT"])
            sq = (mqq["same"][np.ix_(perm, perm)], mqq["cross"][np.ix_(perm, perm)], mqq["TT"])
        Mp = nh[:, None] * (sp[0] - sp[1]) * nh[None, :]
        Mq = nh[:, None] * (sq[0] - sq[1]) * nh[None, :]
        ev, U = np.linalg.eigh(0.5 * (Mp + Mp.T))
        for k in range(len(ev)):
            if ev[k] > 1e-12 * ev.max():
                lam.append(pref * qf * ev[k] / (2 * w ** 4)); mult.append(d * (L - 1))
                sd.append(float(U[:, k] @ Mq @ U[:, k] / ev[k])); lv.append(2); tag.append("L2-standard")
        ev, U = np.linalg.eigh(0.5 * (sp[2] + sp[2].T))
        for k in range(len(ev)):
            if ev[k] > 1e-12 * ev.max():
                lam.append(pref * qf * ev[k] / w ** 4); mult.append(d * L * (L - 1) / 2)
                sd.append(float(U[:, k] @ sq[2] @ U[:, k] / ev[k])); lv.append(2); tag.append("L2-pairs")
        vx = sy["vx"]
        vy_p = vx * float((sy["betas"] ** 2).sum(1).mean()) + c.e
        vy_q = vx * c.sigma_beta2 * d + c.e
        Dx, Dy = L * d, L
        inv_count = {}
        for f in sy["feats"]:
            if f[0] >= 2:
                key = self._feat_xy(f)
                inv_count[key] = inv_count.get(key, 0) + 1
        for lev_ in range(4, self.bulk_max_level + 1, 2):
            for ky in range(lev_ + 1):
                kx = lev_ - ky
                m = math.comb(Dx + kx - 1, kx) * math.comb(Dy + ky - 1, ky) - inv_count.get((kx, ky), 0)
                if m > 0:
                    lam.append(pref * qf * (vx / w ** 2) ** kx * (vy_p / w ** 2) ** ky); mult.append(d * float(m))
                    sd.append((vy_q / vy_p) ** ky); lv.append(lev_); tag.append(f"stair x{kx}y{ky}")
        return dict(lam=np.array(lam), mult=np.array(mult), Sdiag=np.array(sd), level=np.array(lv), tag=tag)


def _ol_feature_matrix(self, s, feats=None):
    """Evaluate M-monomial features on sampled contexts (for Monte Carlo checks)."""
    feats = self.feats if feats is None else feats
    Z = np.concatenate([s["X"], s["y"][:, :, None]], axis=2)
    Mm = np.einsum("nla,nlb->nab", Z, Z)
    pa = np.array([p[0] for p in self.pv]); pb = np.array([p[1] for p in self.pv])
    vals = Mm[:, pa, pb]                                               # (n, nv)
    out = np.ones((Z.shape[0], len(feats)))
    for i, (m, r) in enumerate(feats):
        for vv in self.mono.lists[m][r]:
            out[:, i] *= vals[:, vv]
    return out


OLTheory.feature_matrix = _ol_feature_matrix
