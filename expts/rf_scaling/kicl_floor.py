"""ICL kernel-floor + oracle engine, extracted from

    expts/eigenlearning/run_eigenlearning_OOD_floor_vs_taskdiversity.ipynb

so that it can be imported rather than pasted a third time. The algebra layer
(`_tok_poly` ... `_prod_distinct_poly`), the feature enumeration (`build_features`),
`ood_inputs`, `precompute_ood`/`ood_floors_fast`, `make_tasks`, the numpy oracle ports
and `oracle_mse_one_betas` are all **verbatim** copies of that notebook's cells; only the
notebook globals (dim, L, v, e, CUTOFFS, DEVICE, KIND_TRAIN) have been lifted into
`KICLConfig` and explicit arguments. `test_parity()` asserts agreement with the
notebook's own numbers at its Figure-4 config.

Conventions (matching kernels_can_icl.task_distrs):
  L_cfg  -- context length INCLUDING the query, so there are L = L_cfg - 1 in-context pairs
  v      -- sigma_x2, the per-coordinate in-context input variance (also used for the query)
  e      -- sigma_sq, the label-noise variance (on every label, query included)
  norm2  -- E[y_query^2] = v*d*sigma_beta2 + e; every risk returned here is divided by it
"""
import math
import itertools
import re
import time
from dataclasses import dataclass, field

import numpy as np
import torch



# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KICLConfig:
    """One ICL-regression problem setting.

    Construct with `L_cfg` (context length incl. query). `sigma_x2=None` reproduces the
    kernels_can_icl package default 1/(d*(L_cfg-1)); the configs used here pass it explicitly.
    """
    dim: int
    L_cfg: int
    sigma_sq: float
    sigma_x2: float = None
    sigma_beta2: float = 1.0
    name: str = ""

    @property
    def L(self):
        """Number of in-context (x, y) pairs = the 'falling' L passed to the floor engine."""
        return self.L_cfg - 1

    @property
    def v(self):
        if self.sigma_x2 is not None:
            return float(self.sigma_x2)
        return 1.0 / (self.dim * (self.L_cfg - 1))

    @property
    def e(self):
        return float(self.sigma_sq)

    @property
    def sigma_beta(self):
        return self.sigma_beta2 ** 0.5

    @property
    def norm2(self):
        """E[y_query^2]; every risk in this module is normalized by it."""
        return self.v * self.dim * self.sigma_beta2 + self.e

    @property
    def D(self):
        """Width of the flattened context Z = [X.flatten, y.flatten, Xq.flatten]."""
        return self.L * (self.dim + 1) + self.dim

    def describe(self):
        return (f"{self.name or 'config'}: d={self.dim} L_cfg={self.L_cfg} (L={self.L} pairs) "
                f"e={self.e} v={self.v:.5f} sigma_beta2={self.sigma_beta2} "
                f"D={self.D} norm2={self.norm2:.5f}")


# The three settings in play, plus the paper's Figure-4 setting.
#
#   A  d=7,  8 pairs (L_cfg=9)   eps^2=0.25   the widest OOD gap; the default
#   B  d=6,  9 pairs (L_cfg=10)  eps^2=0.25   highest learnability per feature
#   C  d=4,  9 pairs (L_cfg=10)  eps^2=0.18   cheap, fits a 40 GB card; the validation run
#
# L_cfg is the context length INCLUDING the query, so L_cfg=9 means 8 in-context (x, y) pairs --
# which is what the task-diversity plots label "L=8".
#
# WHICH AXIS COSTS FEATURES. The degree-3 ICL channel Xyq sits in a degenerate family of size
#
#     C(d(L+1) + 1, 2) * L  ~  d^2 (L+1)^2 L / 2,
#
# and the whole family has to be resolved before that channel is learned. So the context length
# enters cubically and the dimension only quadratically: d=6, L=9 needs a block of 16,470 where
# d=4, L=14 needs 25,620. More dimensions with a shorter context is the cheaper trade.
#
# The counterweight, which has to be checked per setting rather than assumed: the degree-3 kernel
# is the L -> infinity linearization of ridge (sum_i x_i x_i^T -> L Sigma_x), so its truncation
# penalty falls like 1/L. Shortening the context raises the floor itself. Whether the OOD gap
# survives is a separate question from whether the channel is learnable -- check it with
# build_floor_engine + run_floor_sweep + gap_summary before committing GPU time.
CONFIG_A = KICLConfig(dim=7, L_cfg=9, sigma_sq=0.25, sigma_x2=1.0 / 7, name="Config A")
CONFIG_B = KICLConfig(dim=6, L_cfg=10, sigma_sq=0.25, sigma_x2=1.0 / 6, name="Config B")
CONFIG_C = KICLConfig(dim=4, L_cfg=10, sigma_sq=0.18, sigma_x2=1.0 / 4, name="Config C")
CONFIG_PAPER = KICLConfig(dim=8, L_cfg=16, sigma_sq=0.25, sigma_x2=1.0 / (8 * 4), name="paper Fig 4")

# The transience settings: d=4, eps^2=0.05, sigma_x2=1/d, trained on a whitened-covariance,
# finite-mean pool (make_tasks kind "whiten_keepmean"). 4 pairs is the default because the
# closed-form RFF solve can still reach the degree-5 channels that carry the rise; 6 pairs is the
# longer run (the closed form then shows the dip but not the rise -- only SGD gets there).
CONFIG_D4P4 = KICLConfig(dim=4, L_cfg=5, sigma_sq=0.05, sigma_x2=1.0 / 4, name="d4 4pairs")
CONFIG_D4P6 = KICLConfig(dim=4, L_cfg=7, sigma_sq=0.05, sigma_x2=1.0 / 4, name="d4 6pairs")


def make_config(dim, pairs, eps=0.05):
    """sigma_x2 = 1/d, `pairs` in-context pairs (L_cfg = pairs + 1, the query counted), noise eps."""
    return KICLConfig(dim=int(dim), L_cfg=int(pairs) + 1, sigma_sq=float(eps),
                      sigma_x2=1.0 / int(dim), name=f"d{int(dim)} {int(pairs)}pairs eps{float(eps):g}")


def config_d4(pairs):
    """d=4, eps^2=0.05, sigma_x2=1/4 with `pairs` in-context pairs (L_cfg = pairs + 1)."""
    return {4: CONFIG_D4P4, 6: CONFIG_D4P6}.get(int(pairs), make_config(4, pairs, 0.05))



# -------------------------------------------------------------------------
# verbatim from the notebook: algebra helpers
# -------------------------------------------------------------------------


def _df(n):
    r = 1.0
    while n > 0: r *= n; n -= 2
    return r
def _gm(p, sb2):                                   # E[Z^p], Z ~ N(0, sb2)
    return 0.0 if p % 2 else _df(p - 1) * sb2 ** (p // 2)

def _tok_poly(px, py, v, e, d):
    def momX(p): return 0.0 if p % 2 else _df(p - 1) * v ** (p // 2)
    def comps(m, dd):
        if dd == 1: yield (m,); return
        for i in range(m + 1):
            for rest in comps(m - i, dd - 1): yield (i,) + rest
    poly = {}
    for j in range(py + 1):
        Exi = 0.0 if j % 2 else _df(j - 1) * e ** (j // 2)
        if Exi == 0.0: continue
        cj = math.comb(py, j); m = py - j
        for k in comps(m, d):
            mult = math.factorial(m)
            for kc in k: mult //= math.factorial(kc)
            xm = 1.0
            for a in range(d):
                xm *= momX(px[a] + k[a])
                if xm == 0.0: break
            if xm == 0.0: continue
            poly[k] = poly.get(k, 0.0) + cj * Exi * mult * xm
    return poly

def _padd(p1, p2, s=1.0):
    out = dict(p1)
    for k, c in p2.items(): out[k] = out.get(k, 0.0) + s * c
    return out
def _pmul(p1, p2):
    out = {}
    for k1, c1 in p1.items():
        for k2, c2 in p2.items():
            k = tuple(a + b for a, b in zip(k1, k2))
            out[k] = out.get(k, 0.0) + c1 * c2
    return out
def _gaussmean(p, sb2):
    s = 0.0
    for k, c in p.items():
        t = c
        for kc in k:
            t *= _gm(kc, sb2)
            if t == 0.0: break
        s += t
    return s
def _empmean(p, betas):
    s = np.zeros(betas.shape[0])
    for k, c in p.items():
        t = c * np.ones(betas.shape[0])
        for ci, kc in enumerate(k):
            if kc: t = t * betas[:, ci] ** kc
        s = s + t
    return float(np.mean(s))

def _combine(l1, l2, d):
    return [(tuple(px1[t] + px2[t] for t in range(d)), py1 + py2, c1 * c2)
            for px1, py1, c1 in l1 for px2, py2, c2 in l2]
def _legpoly(leg, v, e, d):
    out = {}
    for px, py, c in leg:
        for k, val in _tok_poly(px, py, v, e, d).items():
            out[k] = out.get(k, 0.0) + c * val
    return out
def _falling(L, k):
    r = 1.0
    for t in range(k): r *= (L - t)
    return r
def _prod_distinct_poly(legsA, legsB, v, e, L, d):
    p, q = len(legsA), len(legsB); total = {}
    for r in range(min(p, q) + 1):
        for S in itertools.combinations(range(q), r):
            for At in itertools.permutations(range(p), r):
                inj = {S[i]: At[i] for i in range(r)}; usedA = set(inj.values()); tokens = []
                for a in range(p):
                    if a in usedA:
                        s = next(ss for ss in S if inj[ss] == a)
                        tokens.append(_combine(legsA[a], legsB[s], d))
                    else: tokens.append(legsA[a])
                for b in range(q):
                    if b not in S: tokens.append(legsB[b])
                acc = {(0,) * d: _falling(L, len(tokens))}
                for lg in tokens: acc = _pmul(acc, _legpoly(lg, v, e, d))
                total = _padd(total, acc)
    return total



# -------------------------------------------------------------------------
# verbatim from the notebook: feature enumeration (ood_features_highdeg.py)
# -------------------------------------------------------------------------


def _xmono(d, axes, py):
    px = [0] * d
    for a in axes:
        px[a] += 1
    return (tuple(px), py, 1.0)

def build_features(d, max_degree=7, kinds=None):
    # `kinds` (not in the source notebook) keeps only those feature families, e.g. {'q', 'Xy'};
    # None keeps everything, which is the notebook's behaviour.
    F = []
    tok = lambda coords, ypow: [_xmono(d, coords, ypow)]
    add = lambda legs, b, label: F.append(dict(terms=[(tuple(legs), 1.0)], b=b, label=label))
    for b in range(d):
        add([], b, f"q{b}")
        if max_degree >= 3:
            add([tok([], 2)], b, f"y2q{b}")
            for a in range(d):
                add([tok([a], 1)], b, f"X{a}yq{b}")
            for a in range(d):
                for c in range(a, d):
                    add([tok([a, c], 0)], b, f"X{a}X{c}q{b}")
        if max_degree >= 5:
            add([tok([], 4)], b, f"y4q{b}")
            for a in range(d):
                add([tok([a], 3)], b, f"y3X{a}q{b}")
            for a in range(d):
                for c in range(a, d):
                    add([tok([a, c], 2)], b, f"y2X{a}X{c}q{b}")
            for a in range(d):
                for c in range(a, d):
                    add([tok([a], 1), tok([c], 1)], b, f"yX{a}_yX{c}q{b}_cross")
            for a in range(d):
                add([tok([a, a], 0), tok([a], 1)], b, f"yXdotX{a}q{b}_cross")
        if max_degree >= 7:
            add([tok([], 6)], b, f"y6q{b}")
            for a in range(d):
                add([tok([a], 5)], b, f"y5X{a}q{b}")
            for a in range(d):
                for c in range(a, d):
                    add([tok([a, c], 4)], b, f"y4X{a}X{c}q{b}")
            for a in range(d):
                for c in range(a, d):
                    for e in range(c, d):
                        add([tok([a, c, e], 3)], b, f"y3X{a}X{c}X{e}q{b}")
            for a in range(d):
                for c in range(d):
                    add([tok([a], 3), tok([c], 1)], b, f"y3X{a}_yX{c}q{b}_cross")
    if kinds is not None:
        F = [f for f in F if _feat_coords(f['label'])[0] in kinds]
    return F

_FEAT_PATS = [
    (r'y3X(\d+)_yX(\d+)q(\d+)_cross', 'y3X_yX'),
    (r'yX(\d+)_yX(\d+)q(\d+)_cross',  'yX_yX'),
    (r'yXdotX(\d+)q(\d+)_cross',      'yXdotX'),
    (r'y6q(\d+)',                     'y6'),
    (r'y5X(\d+)q(\d+)',               'y5X'),
    (r'y4X(\d+)X(\d+)q(\d+)',         'y4XX'),
    (r'y4q(\d+)',                     'y4'),
    (r'y3X(\d+)X(\d+)X(\d+)q(\d+)',   'y3XXX'),
    (r'y3X(\d+)q(\d+)',               'y3X'),
    (r'y2X(\d+)X(\d+)q(\d+)',         'y2XX'),
    (r'y2q(\d+)',                     'y2'),
    (r'X(\d+)yq(\d+)',                'Xy'),
    (r'X(\d+)X(\d+)q(\d+)',           'XX'),
    (r'q(\d+)',                       'q'),
]
_FEAT_DEG = {'q': 1, 'Xy': 3, 'XX': 3, 'y2': 3,
             'y4': 5, 'y3X': 5, 'y2XX': 5, 'yX_yX': 5, 'yXdotX': 5,
             'y6': 7, 'y5X': 7, 'y4XX': 7, 'y3XXX': 7, 'y3X_yX': 7}

def _feat_coords(label):
    for pat, kind in _FEAT_PATS:
        m = re.fullmatch(pat, label)
        if m:
            g = list(map(int, m.groups()))
            return kind, tuple(g[:-1]), g[-1]
    raise ValueError(f"unrecognized feature label {label!r}; extend _FEAT_PATS")

def _feat_degree(label):
    return _FEAT_DEG[_feat_coords(label)[0]]

def _canon_label(label):
    kind, inp, b = _feat_coords(label)
    sw = lambda c: 0 if c == b else (b if c == 0 else c)
    i0 = tuple(sorted(sw(c) for c in inp))
    if kind == 'q':  return 'q0'
    if kind == 'y2': return 'y2q0'
    if kind == 'Xy': return f'X{i0[0]}yq0'
    if kind == 'XX': return f'X{i0[0]}X{i0[1]}q0'
    raise ValueError(f"_canon_label only supports degree<=3; got {label!r}")



# -------------------------------------------------------------------------
# verbatim from the notebook: reference engine (ood_inputs)
# -------------------------------------------------------------------------

def _poolmean(p, mom):
    s = 0.0
    for k, c in p.items():
        t = c
        for kc in k:
            t *= mom[kc]
        s += t
    return s

def _gram2(Fsel, v, e, L, d, red_p, red_q):
    P = len(Fsel); Gp = np.zeros((P, P)); Gq = np.zeros((P, P))
    for i in range(P):
        for j in range(i, P):
            if Fsel[i]['b'] != Fsel[j]['b']:
                continue
            poly = {}
            for li, ci in Fsel[i]['terms']:
                for lj, cj in Fsel[j]['terms']:
                    poly = _padd(poly, _prod_distinct_poly(li, lj, v, e, L, d), ci * cj)
            gp = v * red_p(poly); gq = v * red_q(poly)
            Gp[i, j] = Gp[j, i] = gp; Gq[i, j] = Gq[j, i] = gq
    return Gp, Gq

def _Vvec2(Fsel, v, e, L, d, red_p, red_q):
    Vp = np.zeros(len(Fsel)); Vq = np.zeros(len(Fsel))
    for i, F in enumerate(Fsel):
        b = F['b']; poly = {}
        for legs, c in F['terms']:
            poly = _padd(poly, _prod_distinct_poly(legs, (), v, e, L, d), c)
        poly_b = {tuple(k[t] + (1 if t == b else 0) for t in range(d)): cc for k, cc in poly.items()}
        Vp[i] = v * red_p(poly_b); Vq[i] = v * red_q(poly_b)
    return Vp, Vq

def ood_inputs(betas_train, test_prior, v, e, L, d, marked, tol=1e-10, kmax=12,
               exact=False, max_degree=7, pooled=False):
    # pooled=False (default): PER-ROW empirical task moments, so the degree-1 eigcoeffs equal the
    # actual task mean Mb[k]=E_tasks[beta_k] (= the single task itself at T=1). pooled=True uses the
    # iid-coordinate pooled marginal (fast/axis-symmetric) but is WRONG at small T (it can't represent
    # a specific task vector). See notebook markdown.
    betas_train = np.asarray(betas_train, float)
    mom = np.array([np.mean(betas_train.ravel() ** k) for k in range(kmax + 1)])
    p_red = (lambda p: _poolmean(p, mom)) if pooled else (lambda p: _empmean(p, betas_train))
    signal_p = (d * mom[2] * v) if pooled else float(np.mean(np.sum(betas_train ** 2, 1))) * v

    kind, payload = test_prior
    if kind == 'gauss':
        sb2 = float(payload); q_red = lambda p: _gaussmean(p, sb2); Eb2 = sb2
    else:
        bt = np.asarray(payload, float)
        if pooled:
            momq = np.array([np.mean(bt.ravel() ** k) for k in range(kmax + 1)])
            q_red = lambda p: _poolmean(p, momq); Eb2 = float(momq[2])
        else:
            q_red = lambda p: _empmean(p, bt); Eb2 = float(np.mean(np.sum(bt ** 2, 1)) / d)

    F = build_features(d, max_degree)
    lab2 = {f['label']: i for i, f in enumerate(F)}
    for b in range(d):
        lab2.setdefault(f"Xyq{b}",  lab2.get(f"X{b}yq{b}"))
        lab2.setdefault(f"X2q{b}",  lab2.get(f"X{b}X{b}q{b}"))
        lab2.setdefault(f"y3Xq{b}", lab2.get(f"y3X{b}q{b}"))
        lab2.setdefault(f"y5Xq{b}", lab2.get(f"y5X{b}q{b}"))
    by_axis = {}
    for i, f in enumerate(F):
        by_axis.setdefault(f['b'], []).append(i)

    real = [(p, F[lab2[m]]['b'], F[lab2[m]]['label'])
            for p, m in enumerate(marked) if lab2.get(m) is not None]

    def _diag_deg3(li):
        _, inp, bb = _feat_coords(li)
        return _feat_degree(li) <= 3 and all(c == bb for c in inp)
    fast = pooled and (not exact) and all(_diag_deg3(li) for _, _, li in real)

    Pm = len(marked)
    v_tr = np.zeros(Pm); v_te = np.zeros(Pm); a = np.zeros(Pm); S = np.eye(Pm)

    def _block(idxs):
        Fb = [F[i] for i in idxs]
        Gp, Gq = _gram2(Fb, v, e, L, d, p_red, q_red)
        Vp, Vq = _Vvec2(Fb, v, e, L, d, p_red, q_red)
        Lc = np.linalg.cholesky(Gp + tol * np.mean(np.diag(Gp)) * np.eye(len(idxs)))
        Li = np.linalg.solve(Lc, np.eye(len(idxs)))
        Sb = Li @ Gq @ Li.T; vtr = Li @ Vp; ab = Li @ Vq
        return Sb, vtr, ab, np.linalg.solve(Sb, ab), {F[i]['label']: t for t, i in enumerate(idxs)}

    if fast:
        Sb, vtr, ab, vte, pos = _block(by_axis[0])
        by_b = {}
        for p, b, li in real:
            by_b.setdefault(b, []).append((p, li))
        for members in by_b.values():
            for p, li in members:
                t = pos[_canon_label(li)]
                v_tr[p] = vtr[t]; a[p] = ab[t]; v_te[p] = vte[t]
                for pj, lj in members:
                    S[p, pj] = Sb[t, pos[_canon_label(lj)]]
    else:
        for b in sorted({bb for _, bb, _ in real}):
            Sb, vtr, ab, vte, pos = _block(by_axis[b])
            for pi, bi, li in real:
                if bi != b:
                    continue
                v_tr[pi] = vtr[pos[li]]; a[pi] = ab[pos[li]]; v_te[pi] = vte[pos[li]]
                for pj, bj, lj in real:
                    if bj == b:
                        S[pi, pj] = Sb[pos[li], pos[lj]]

    signal_q = Eb2 * d * v
    residual = max(signal_q - float(a @ v_te), 0.0)
    norm2 = signal_q + e
    delta = v_te - v_tr
    resid_p = max(signal_p - float(v_tr @ v_tr), 0.0)
    s = np.sqrt(norm2)
    return dict(eigcoeffs_train=v_tr / s,
                eigcoeffs_test=v_te / s,
                test_gram_OOD=S,
                noise_var_eff=(residual + e) / norm2,
                noise_var_train_eff=(resid_p + e) / norm2,
                floor=float(delta @ S @ delta + residual + e) / norm2)



# -------------------------------------------------------------------------
# verbatim from the notebook: fast cached engine (precompute_ood / ood_floors_fast)
# -------------------------------------------------------------------------

# ---- canonical context key (for the axis-symmetry permutation) -------------
def _ctx_key(f, swap):
    toks = []
    for leg in f['terms'][0][0]:
        for (px, py, _c) in leg:
            toks.append((tuple(px[swap[i]] for i in range(len(px))), py))
    return tuple(sorted(toks))

def _flatten(polys_ix_cs, nrows):
    """polys_ix_cs: list of (coeffs, kidx) -> flat (seg, kidx, coef) for np.bincount assembly."""
    seg, kidx, coef = [], [], []
    for r, (cs, ix) in enumerate(polys_ix_cs):
        seg.append(np.full(len(cs), r, dtype=np.intp)); kidx.append(ix); coef.append(cs)
    z = (np.zeros(0, np.intp), np.zeros(0, np.intp), np.zeros(0))
    return (np.concatenate(seg) if seg else z[0],
            np.concatenate(kidx) if kidx else z[1],
            np.concatenate(coef) if coef else z[2], nrows)

# ---- precompute: reference-block polynomial layer -> distinct multi-indices Kmat + flat
#      (segment, kidx, coeff) arrays so per-sample assembly is a vectorized np.bincount.
def precompute_ood(v, e, L, d, marked, max_degree=7, tol=1e-10, axis_symmetric=True, kinds=None):
    F = build_features(d, max_degree, kinds)
    lab2 = {f['label']: i for i, f in enumerate(F)}
    for b in range(d):
        lab2.setdefault(f"Xyq{b}",  lab2.get(f"X{b}yq{b}"))
        lab2.setdefault(f"X2q{b}",  lab2.get(f"X{b}X{b}q{b}"))
        lab2.setdefault(f"y3Xq{b}", lab2.get(f"y3X{b}q{b}"))
        lab2.setdefault(f"y5Xq{b}", lab2.get(f"y5X{b}q{b}"))
    by_axis = {}
    for i, f in enumerate(F):
        by_axis.setdefault(f['b'], []).append(i)
    real = [(p, F[lab2[m]]['b'], F[lab2[m]]['label'])
            for p, m in enumerate(marked) if lab2.get(m) is not None]
    axes = sorted({bb for _, bb, _ in real})
    b0 = axes[0] if axes else 0

    F0 = [F[i] for i in by_axis[b0]]; P = len(F0)
    Kset = {}
    def _reg(poly):
        cs = np.empty(len(poly)); ix = np.empty(len(poly), dtype=np.intp)
        for t, (k, c) in enumerate(poly.items()):
            ix[t] = Kset.setdefault(k, len(Kset)); cs[t] = c
        return cs, ix
    iu, ju = np.triu_indices(P)
    gram_pairs = []
    for i, j in zip(iu, ju):
        poly = {}
        for li, ci in F0[i]['terms']:
            for lj, cj in F0[j]['terms']:
                poly = _padd(poly, _prod_distinct_poly(li, lj, v, e, L, d), ci * cj)
        gram_pairs.append(_reg(poly))
    vpoly = []
    for f in F0:
        bb = f['b']; poly = {}
        for legs, c in f['terms']:
            poly = _padd(poly, _prod_distinct_poly(legs, (), v, e, L, d), c)
        vpoly.append(_reg({tuple(k[t] + (1 if t == bb else 0) for t in range(d)): cc
                           for k, cc in poly.items()}))
    Kmat = np.array(list(Kset.keys()), dtype=np.intp) if Kset else np.zeros((0, d), np.intp)
    key0 = {_ctx_key(F0[t], list(range(d))): t for t in range(P)}
    block0 = dict(P=P, Kmat=Kmat, iu=iu, ju=ju,
                  gram=_flatten(gram_pairs, len(iu)), vvec=_flatten(vpoly, P),
                  pos={F0[t]['label']: t for t in range(P)})

    blocks = {b0: block0}; perm = {b0: np.arange(P)}
    for b in axes:
        if b == b0:
            continue
        Fb = [F[i] for i in by_axis[b]]
        blocks[b] = dict(P=len(Fb), pos={Fb[t]['label']: t for t in range(len(Fb))})
        swap = list(range(d)); swap[b0], swap[b] = b, b0
        perm[b] = np.array([key0[_ctx_key(Fb[t], swap)] for t in range(len(Fb))])
    return dict(real=real, axes=axes, b0=b0, blocks=blocks, perm=perm,
                axis_symmetric=True, Pm=len(marked), d=d, v=v, e=e, tol=tol)

def _gauss_moments(Kmat, sb2):
    if Kmat.size == 0:
        return np.ones(0)
    gmtab = np.array([_gm(p, sb2) for p in range(int(Kmat.max()) + 1)])
    M = np.ones(Kmat.shape[0])
    for c in range(Kmat.shape[1]):
        M *= gmtab[Kmat[:, c]]
    return M

def _assemble(block, M, v):
    seg, kidx, coef, nP = block['gram']
    gu = np.bincount(seg, weights=coef * M[kidx], minlength=nP) * v
    P = block['P']; G = np.zeros((P, P)); iu, ju = block['iu'], block['ju']
    G[iu, ju] = gu; G[ju, iu] = gu
    vseg, vkidx, vcoef, nV = block['vvec']
    V = np.bincount(vseg, weights=vcoef * M[vkidx], minlength=nV) * v
    return G, V

def ood_floors_fast(pre, betas, sb2):
    """PER-ROW empirical floor. Train reduce = empirical mean over the T task rows (degree-1 eigcoeffs
    = task mean = the single task at T=1). Beta powers are gathered once; each Gram is assembled by a
    vectorized bincount. The isotropic gauss test Gram is reduced once on the reference block; axis b's
    train Gram is that block with betas' columns swapped 0<->b, mapped back by the precomputed perm.
    Matches ood_inputs(..., pooled=False)."""
    betas = np.asarray(betas, float); T = betas.shape[0]
    d, v, e, tol, Pm = pre['d'], pre['v'], pre['e'], pre['tol'], pre['Pm']
    v_tr = np.zeros(Pm); v_te = np.zeros(Pm); a = np.zeros(Pm); S = np.eye(Pm)
    b0 = pre['b0']; blk0 = pre['blocks'][b0]; Kmat = blk0['Kmat']
    maxp = int(Kmat.max()) + 1 if Kmat.size else 1
    bp = betas[:, :, None] ** np.arange(maxp)[None, None, :]                # (T,d,maxp) exponentiate once
    Gq, Vq = _assemble(blk0, _gauss_moments(Kmat, sb2), v)                 # isotropic -> once
    solves = {}
    for b in pre['axes']:
        swap = list(range(d)); swap[b0], swap[b] = b, b0
        acc = np.ones((T, Kmat.shape[0]))
        for c in range(d):
            acc *= bp[:, swap[c], Kmat[:, c]]                              # gather, no re-exponentiation
        Gp, Vp = _assemble(blk0, acc.mean(0), v)
        Lc = np.linalg.cholesky(Gp + tol * np.mean(np.diag(Gp)) * np.eye(blk0['P']))
        Li = np.linalg.solve(Lc, np.eye(blk0['P']))
        Sb = Li @ Gq @ Li.T; vtr = Li @ Vp; ab = Li @ Vq
        solves[b] = (Sb, vtr, ab, np.linalg.solve(Sb, ab))
    for pi, bi, li in pre['real']:
        Sb, vtr, ab, vte = solves[bi]
        ti = pre['perm'][bi][pre['blocks'][bi]['pos'][li]]
        v_tr[pi] = vtr[ti]; a[pi] = ab[ti]; v_te[pi] = vte[ti]
        for pj, bj, lj in pre['real']:
            if bj == bi:
                tj = pre['perm'][bi][pre['blocks'][bi]['pos'][lj]]
                S[pi, pj] = Sb[ti, tj]

    signal_q = sb2 * d * v
    residual = max(signal_q - float(a @ v_te), 0.0)
    norm2 = signal_q + e
    delta = v_te - v_tr
    signal_p = float(np.mean(np.sum(betas ** 2, 1))) * v
    resid_p = max(signal_p - float(v_tr @ v_tr), 0.0)
    return (resid_p + e) / norm2, float(delta @ S @ delta + residual + e) / norm2



# -------------------------------------------------------------------------
# verbatim from the notebook: task-pool shapes (make_tasks)
# -------------------------------------------------------------------------

def _zca(hc, T):
    """Symmetric whitening of centred rows: cov -> I (or the projector onto the span if rank-limited)."""
    evals, U = torch.linalg.eigh((hc.T @ hc) / T)
    keep = evals > 1e-10 * evals.max().clamp_min(1e-30)
    scale = torch.where(keep, evals.clamp_min(1e-30).rsqrt(), torch.zeros_like(evals))
    return hc @ (U * scale) @ U.T


def _chol_white(w, T):
    """Cholesky whitening: the ZCA pool up to a rotation, so rotation-invariant moment norms agree."""
    wc = w - w.mean(0, keepdim=True)
    L = torch.linalg.cholesky(wc.T @ wc / T)
    return torch.linalg.solve_triangular(L, wc.T, upper=False).T


def make_tasks(T, d, kind, device):
    if kind == "shifted":
        # ~ N(mean, I): nonzero first moment (||mean||=r), cov=I, zero skew.
        r = 2
        u = torch.randn(d, device=device); u = u / u.norm()
        return torch.randn(T, d, device=device) + r * u

    if kind == "antipodal":
        assert T % 2 == 0, "antipodal needs even T"
        h = torch.randn(T // 2, d, device=device)
        return torch.cat([h, -h], dim=0)[torch.randperm(T, device=device)]

    if kind == "ondiags":
        h = torch.randn(T, d, device=device)
        h = h - h.mean(0, keepdim=True)            # beta_bar = 0 exactly
        C = (h.T @ h) / T
        _, U = torch.linalg.eigh(C)
        return h @ U                               # decorrelated: cov = diag(evals)

    if kind == "offdiags":
        # anisotropic prior rotated OFF the coordinate axes -> sizeable off-diagonal cov. beta_bar=0.
        aniso = 8.0
        h = torch.randn(T, d, device=device)
        h = h - h.mean(0, keepdim=True)
        evals, U = torch.linalg.eigh((h.T @ h) / T)
        Zw = (h @ U) / evals.clamp_min(1e-12).sqrt()   # whiten: cov = I
        s = torch.linspace(1.0, aniso, d, device=device)
        s = s / s.pow(2).mean().sqrt()
        Q, R = torch.linalg.qr(torch.randn(d, d, device=device))
        Q = Q * torch.sign(torch.diagonal(R))          # Haar rotation
        return (Zw * s) @ Q

    elif kind == "isogauss":
        # cov=I exactly, higher moments isotropic AND small (excess kurtosis/6th -> 0).
        if T <= d:
            raise ValueError("isogauss needs T > d for full-rank whitening.")
        def whiten(x):
            xc = x - x.mean(0, keepdim=True)
            C = (xc.T @ xc) / T
            evals, U = torch.linalg.eigh(C)
            return xc @ (U * evals.clamp_min(1e-12).rsqrt()) @ U.T
        n_dir = min(4096, max(256, 6 * d * d))
        Gdir  = torch.randn(n_dir, d, device=device)
        U_dir = torch.cat([torch.eye(d, device=device),
                           Gdir / Gdir.norm(dim=1, keepdim=True)], 0)
        W_ISO, W_SMALL = 10.0, 1.0
        w4, w6 = 1.0, 1.0
        w = torch.randn(T, d, device=device, requires_grad=True)
        optimizer = torch.optim.LBFGS([w], max_iter=250, line_search_fn="strong_wolfe")
        def closure():
            optimizer.zero_grad()
            z  = whiten(w)
            p  = z @ U_dir.T
            m4 = (p ** 4).mean(0)
            m6 = (p ** 6).mean(0)
            iso   = w4 * (m4 / 3.0).var(unbiased=False) + w6 * (m6 / 15.0).var(unbiased=False)
            small = w4 * ((m4.mean() - 3.0) / 3.0) ** 2 + w6 * ((m6.mean() - 15.0) / 15.0) ** 2
            loss = W_ISO * iso + W_SMALL * small
            loss.backward()
            return loss
        optimizer.step(closure)
        with torch.no_grad():
            return whiten(w)

    elif kind == "skewnormal":
        # i.i.d. N(0,I) draws, ACCEPTED with prob Phi(alpha<x,u>) -> multivariate skew-normal.
        alpha  = 6.0
        do_whiten = True
        u = torch.randn(d, device=device); u = u / u.norm()
        kept, need = [], T
        while need > 0:
            g   = torch.randn(max(3 * need, 1024), d, device=device)
            acc = torch.rand(g.shape[0], device=device) < torch.special.ndtr(alpha * (g @ u))
            kept.append(g[acc]); need = T - sum(k.shape[0] for k in kept)
        h = torch.cat(kept, 0)[:T]
        h = h                                          # (beta_bar = 0 NOT subtracted; see original)
        if do_whiten:
            C = (h.T @ h) / T
            evals, U = torch.linalg.eigh(C)
            h = h @ (U * evals.clamp_min(1e-12).rsqrt()) @ U.T
        return h

    elif kind in ("unlucky_wide", "unlucky_clustered"):
        pool_mult = 64 if kind == "unlucky_wide" else 320
        M = pool_mult * T
        pool = torch.randn(M, d, device=device)
        if kind == "unlucky_wide":
            d2  = ((pool - pool.mean(0, keepdim=True)) ** 2).sum(-1)
            idx = torch.argsort(d2)[-T:]
        else:
            ctr = pool[torch.randint(M, (1,)).item()]
            d2  = ((pool - ctr) ** 2).sum(-1)
            idx = torch.argsort(d2)[:T]
        return pool[idx]

    if kind == "whiten_keepmean":
        # Covariance whitened, sample mean KEPT: beta = mu_hat + ZCA(h - mu_hat). Sigma_hat = I and
        # mu_hat is the natural mean of T draws (||mu_hat||^2 ~ d/T), so the degree-1 channel still
        # has something pool-specific to learn. Whitening the raw second moment instead would leave
        # Cov = I - mu mu^T, which is not white.
        #
        # Computed in float64 so Sigma_hat = I holds to ~1e-12; the draw itself stays float32 so the
        # random stream matches the other kinds. T=1 returns the raw draw (nothing to whiten), and
        # T <= d is rank-limited: the covariance is the projector onto the span of the centered draws.
        h = torch.randn(T, d, device=device).double()
        if T == 1:
            return h
        mu = h.mean(0, keepdim=True)
        hc = h - mu
        evals, U = torch.linalg.eigh((hc.T @ hc) / T)
        keep = evals > 1e-10 * evals.max().clamp_min(1e-30)
        scale = torch.where(keep, evals.clamp_min(1e-30).rsqrt(), torch.zeros_like(evals))
        return mu + hc @ (U * scale) @ U.T

    if kind == "whiten":
        # Centered ZCA: mu_hat = 0 and Sigma_hat = I exactly, higher moments left as drawn.
        # (isogauss additionally flattens the 4th and 6th moments by LBFGS; this does not.)
        #
        # Centering costs one rank, so Sigma_hat = I is only attainable for T > d. Below that the
        # covariance is rank-limited and the closest thing to I in Frobenius norm is the projector
        # onto the span -- every nonzero eigenvalue set to 1, the orthogonal complement left empty.
        # That is what this returns, so small T degrades gracefully instead of failing.
        h = torch.randn(T, d, device=device)
        if T > 1:
            h = h - h.mean(0, keepdim=True)   # a single task cannot be centered: it would vanish
        C = (h.T @ h) / T
        evals, U = torch.linalg.eigh(C)
        keep = evals > 1e-10 * evals.max().clamp_min(1e-30)
        scale = torch.where(keep, evals.clamp_min(1e-30).rsqrt(), torch.zeros_like(evals))
        return h @ (U * scale) @ U.T

    if kind == "whiten_zeromean_tr1":
        # Sigma_hat = I, mu_hat = 0 AND Tr_1[kappa_3] = 0 (the vector sum_a kappa_3[a,a,c]).
        #
        # Those three are exactly the pool statistics the degree<=3 channels can see: with all of them
        # zero the degree-3 floor equals its Gaussian-prior value (optimally scaled one-step GD) at
        # EVERY T, instead of sitting above it by an O(1/T) pool artifact. The q and y2q channels then
        # carry no target power at all, so the whole degree-3 solution is the Xyq channel.
        #
        # For T <= d+1 the centred whitened draw is a regular simplex: every |z_t|^2 is equal and the
        # z_t sum to zero, so Tr_1 kappa_3 vanishes already and no projection is needed.
        h = torch.randn(T, d, device=device).double()
        z = _zca(h - h.mean(0, keepdim=True), T)
        if T <= d + 1:
            return z
        w = z.clone().requires_grad_(True)
        opt = torch.optim.LBFGS([w], max_iter=500, tolerance_grad=1e-14, tolerance_change=1e-16,
                                history_size=50, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            zz = _chol_white(w, T)          # same pool as ZCA up to a rotation; stabler gradients
            loss = (((zz * zz).sum(1, keepdim=True) * zz).mean(0) ** 2).sum()
            loss.backward()
            return loss

        for _ in range(4):
            opt.step(closure)
        with torch.no_grad():
            return _zca(w - w.mean(0, keepdim=True), T)

    return torch.randn(T, d, device=device)          # "normal": isotropic N(0, I)



# -------------------------------------------------------------------------
# verbatim from the notebook: numpy oracle ports + vectorized MC evaluator
# -------------------------------------------------------------------------

# --- numpy ports of kernels_can_icl/oracle_models.py (per-context predictors) ---
class Ridge:
    """Ridge oracle, ridge strength = noise_var  (posterior mean under N(0, I) prior)."""
    def __init__(self, noise_var): self.ridge = float(noise_var)
    def predict(self, X, y, query):
        A = X.T @ X + self.ridge * np.eye(X.shape[1])
        w = np.linalg.lstsq(A, X.T @ y, rcond=None)[0]
        return query @ w

class dMMSE:
    """Discrete MMSE over the finite candidate set `betas` (posterior mean over the T pretraining tasks)."""
    def __init__(self, betas, noise_var): self.betas = betas; self.sigma_sq = float(noise_var)
    def predict(self, X, y, query):
        sq = (y - X @ self.betas.T) ** 2
        logits = (-1.0 / (2.0 * self.sigma_sq)) * sq.sum(0)
        logits -= logits.max()
        sc = np.exp(logits); sc /= sc.sum()
        return query @ (sc @ self.betas).reshape(-1, 1)

def oracle_mse_one_betas(betas, dim, L_cfg, v, sigma_sq, sigma_beta, norm2, n_mc, rng, chunk=256):
    """Vectorized MC test-MSE of Ridge & dMMSE for ONE shared task set `betas`. ID query task = a row of
    `betas`; OOD query task ~ N(0, sigma_beta^2 I). MSE vs the noisy query label, normalized by norm2."""
    T = betas.shape[0]; L1 = L_cfg - 1
    sig, sx, lam, eye = np.sqrt(sigma_sq), np.sqrt(v), sigma_sq, np.eye(dim)
    acc = dict(ridge_id=0.0, ridge_ood=0.0, dmmse_id=0.0, dmmse_ood=0.0); done = 0
    while done < n_mc:
        B = min(chunk, n_mc - done); done += B
        Xc = rng.normal(0.0, sx, size=(B, L1, dim)); Xq = rng.normal(0.0, sx, size=(B, dim))
        bID = betas[rng.integers(0, T, size=B)]; bOOD = rng.normal(0.0, sigma_beta, size=(B, dim))
        A = np.einsum('bld,ble->bde', Xc, Xc) + lam * eye
        for tag, bb in [('id', bID), ('ood', bOOD)]:
            yc = np.einsum('bld,bd->bl', Xc, bb) + rng.normal(0, sig, size=(B, L1))
            yq = np.einsum('bd,bd->b', Xq, bb) + rng.normal(0, sig, size=B)
            wR = np.linalg.solve(A, np.einsum('bld,bl->bd', Xc, yc)[:, :, None])[:, :, 0]
            acc['ridge_'+tag] += np.sum((np.einsum('bd,bd->b', Xq, wR) - yq) ** 2)
            preds = np.einsum('bld,td->blt', Xc, betas)
            logit = -((yc[:, :, None] - preds) ** 2).sum(1) / (2 * sigma_sq); logit -= logit.max(1, keepdims=True)
            sc = np.exp(logit); sc /= sc.sum(1, keepdims=True); wD = sc @ betas
            acc['dmmse_'+tag] += np.sum((np.einsum('bd,bd->b', Xq, wD) - yq) ** 2)
    return {k: acc[k] / n_mc / norm2 for k in acc}



# ---------------------------------------------------------------------------
# public API: task pools, floor engines, sweeps
# ---------------------------------------------------------------------------

def default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def draw_betas(T, cfg, kind="normal", device=None):
    """T pretraining tasks as a (T, dim) float64 numpy array.

    The SAME draw must feed the floor engine and the oracles, otherwise the kernel and
    dMMSE curves are computed against different task pools and the comparison is meaningless.
    """
    device = default_device() if device is None else device
    W = make_tasks(int(T), int(cfg.dim), kind, device)
    W = W * cfg.sigma_beta            # make_tasks draws unit-variance; scale to sigma_beta2
    return W.detach().cpu().numpy().astype(float)


def build_floor_engine(cfg, cutoffs=(1, 3, 5)):
    """Precompute the (beta-independent) polynomial layer once per degree cutoff.

    Returns {deg: pre}, where `pre` is consumed by `ood_floors_fast(pre, betas, sigma_beta2)`.
    This is the expensive part; it is independent of T and of the task draw.
    """
    out = {}
    for deg in cutoffs:
        marked = [f["label"] for f in build_features(cfg.dim, deg)]
        out[deg] = precompute_ood(cfg.v, cfg.e, cfg.L, cfg.dim, marked, deg)
    return out


# ---------------------------------------------------------------------------
# channel ladder: floors as the channel families switch on, in kernel-eigenvalue order
# ---------------------------------------------------------------------------

# (feature kind, y-degree, x-degree). A family's kernel eigenvalue is ~ rb^a * ra^b with
# rb = (d v + eps^2)/w^2 > ra = v/w^2, so within a degree more y's come first, and degree 3 precedes
# degree 5 whenever ra > rb^2 (w^2 > (dv+eps^2)^2/v, i.e. w >~ 0.5x median for the d=4 configs; at
# exactly 0.5x median XX and y4 are near-degenerate, which barely matters since y4 carries ~1% of
# the rise). The rise itself is almost all y2XX + yX_yX = y_i y_j x_i x_j x_q, the kappa_3 term.
LADDER = (("q", 0, 1), ("y2", 2, 1), ("Xy", 1, 2), ("XX", 0, 3),
          ("y4", 4, 1), ("y3X", 3, 2), ("y2XX", 2, 3), ("yX_yX", 2, 3), ("yXdotX", 1, 4))
LADDER_LABELS = {"q": "q", "y2": "+y2q", "Xy": "+Xyq", "XX": "+XXq (deg<=3)", "y4": "+y4q",
                 "y3X": "+y3Xq", "y2XX": "+y2XXq", "yX_yX": "+yX_yX", "yXdotX": "+yXdotX (deg<=5)"}


def build_ladder_engine(cfg, ladder=LADDER, max_degree=5):
    """[(kind, pre)]: entry k is the engine for a kernel that has learned ladder[:k+1] and nothing else.

    Each prefix gets its OWN engine, whose feature block is exactly that prefix. Do not build one
    deg<=5 engine and drop entries from pre['real']: ood_floors_fast regresses the OOD target on the
    whole block, so a partial marked set gives the wrong residual (it put the q-only floor at 1.02
    where the closed form 1 + v*||mu_hat||^2/norm2 gives 1.13).
    """
    out, kinds = [], set()
    for kind, _, _ in ladder:
        kinds = kinds | {kind}
        labels = [f["label"] for f in build_features(cfg.dim, max_degree, kinds)]
        out.append((kind, precompute_ood(cfg.v, cfg.e, cfg.L, cfg.dim, labels, max_degree,
                                         kinds=kinds)))
    return out


def ladder_floors(engine, betas, sb2):
    """{kind: (ID floor, OOD floor)}, cumulative in ladder order, for one task pool."""
    out = {}
    for kind, pre in engine:
        try:
            out[kind] = ood_floors_fast(pre, betas, sb2)
        except np.linalg.LinAlgError:
            out[kind] = (np.nan, np.nan)
    return out


def ladder_pieces(pre, betas, sb2):
    """What ood_floors_fast computes internally, returned instead of reduced to a floor, so that a
    PARTIALLY learned predictor can be scored. `pre` should be the full ladder engine (last entry of
    build_ladder_engine), whose orthonormalization order is the eigenvalue order.

    Returns dict with, in the training-orthonormal basis of the marked features:
      v_tr  -- the training target's coefficients (what a fully learned kernel converges to)
      v_te  -- the OOD-optimal coefficients;  S -- the OOD Gram of the basis
      resid_te / resid_tr -- target power outside the block (OOD / ID);  fam -- family of each feature
    A predictor with per-feature learnability L has
      OOD = ((v_te - L v_tr)^T S (v_te - L v_tr) + resid_te + e) / norm2
      ID  = (||(1 - L) v_tr||^2 + resid_tr + e) / norm2
    and L = 1 on the whole block reproduces ood_floors_fast exactly.
    """
    betas = np.asarray(betas, float); T = betas.shape[0]
    d, v, e, tol, Pm = pre['d'], pre['v'], pre['e'], pre['tol'], pre['Pm']
    v_tr = np.zeros(Pm); v_te = np.zeros(Pm); a = np.zeros(Pm); S = np.eye(Pm)
    b0 = pre['b0']; blk0 = pre['blocks'][b0]; Kmat = blk0['Kmat']
    bp = betas[:, :, None] ** np.arange(int(Kmat.max()) + 1)[None, None, :]
    Gq, Vq = _assemble(blk0, _gauss_moments(Kmat, sb2), v)
    solves = {}
    for b in pre['axes']:
        swap = list(range(d)); swap[b0], swap[b] = b, b0
        acc = np.ones((T, Kmat.shape[0]))
        for c in range(d):
            acc *= bp[:, swap[c], Kmat[:, c]]
        Gp, Vp = _assemble(blk0, acc.mean(0), v)
        Lc = np.linalg.cholesky(Gp + tol * np.mean(np.diag(Gp)) * np.eye(blk0['P']))
        Li = np.linalg.solve(Lc, np.eye(blk0['P']))
        Sb = Li @ Gq @ Li.T; vtr = Li @ Vp; ab = Li @ Vq
        solves[b] = (Sb, vtr, ab, np.linalg.solve(Sb, ab))
    fam = []
    for pi, bi, li in pre['real']:
        Sb, vtr, ab, vte = solves[bi]
        ti = pre['perm'][bi][pre['blocks'][bi]['pos'][li]]
        v_tr[pi] = vtr[ti]; a[pi] = ab[ti]; v_te[pi] = vte[ti]
        fam.append(_feat_coords(li)[0])
        for pj, bj, lj in pre['real']:
            if bj == bi:
                S[pi, pj] = Sb[ti, pre['perm'][bi][pre['blocks'][bi]['pos'][lj]]]
    signal_q = sb2 * d * v
    return dict(v_tr=v_tr, v_te=v_te, S=S, fam=np.array(fam), e=e, norm2=signal_q + e,
                resid_te=max(signal_q - float(a @ v_te), 0.0),
                resid_tr=max(float(np.mean(np.sum(betas ** 2, 1))) * v - float(v_tr @ v_tr), 0.0))


def score_learnability(pieces, L):
    """(ID, OOD) normalized error of the predictor with per-feature learnability L (array or scalar)."""
    p = pieces
    dl = p["v_te"] - L * p["v_tr"]
    r = (1.0 - L) * p["v_tr"]
    return ((float(r @ r) + p["resid_tr"] + p["e"]) / p["norm2"],
            (float(dl @ p["S"] @ dl) + p["resid_te"] + p["e"]) / p["norm2"])


class ChannelSpectrum:
    """Enumerable spectrum of the Gaussian kernel on flattened ICL contexts.

    Z has two coordinate-variance classes -- x (context + query, multiplicity L*d + d, variance v) and
    y (multiplicity L, variance d*v*sigma_beta2 + eps^2 unless `y_var` is given) -- and lambda depends
    only on the degree in each: lambda(a, b) = rb^a ra^b / tot, a = y-degree, b = x-degree, with trace 1
    (K(z, z) = 1). The coordinates are treated as independent Gaussians, which ignores the
    beta-induced correlation between tokens: a sizing tool, not a prediction of the curve.

    form="mehler" (default) uses the EXACT per-coordinate ratio of the Gaussian kernel on a Gaussian
    coordinate, r = b/(a+b+c) with a = 1/(4 var), b = 1/(2 w^2), c = sqrt(a^2 + 2ab) (Rasmussen &
    Williams 4.3.1). form="hea" uses the dot-product-expansion ratio var/w^2, which it approaches only
    for wide kernels: at 0.5x median the y ratio is 0.29 (hea) vs 0.19 (mehler), and the empirical
    spectrum of the kernel on real ICL contexts follows mehler to ~10-20% over the top ~400 modes where
    hea is off by up to 2x -- so hea understated the SGD step count for the kappa_3 family ~3x. At 1.5x
    median the two agree to a few percent. tot = prod (1 - r)^-m keeps the trace at 1 in both forms.

    Ridge enters in covariance units throughout: an absolute ridge r_abs on Phi^T Phi (or on the n x n
    kernel matrix) at sample count n is `ridge_rel = r_abs / n`.
    """

    def __init__(self, cfg, width, maxdeg=40, ladder=LADDER, form="mehler", y_var=None):
        from math import comb
        self.cfg, self.width, self.form = cfg, float(width), form
        mA, mB = cfg.L * cfg.dim + cfg.dim, cfg.L
        vy = cfg.dim * cfg.v * cfg.sigma_beta2 + cfg.e if y_var is None else float(y_var)

        def ratio(var):
            if form == "hea":
                return var / width ** 2
            a, b = 1.0 / (4.0 * var), 1.0 / (2.0 * width ** 2)
            return b / (a + b + math.sqrt(a * a + 2.0 * a * b))

        self.ra, self.rb = ratio(cfg.v), ratio(vy)
        tot = (1.0 - self.ra) ** (-mA) * (1.0 - self.rb) ** (-mB)
        lam, mult = [], []
        for deg in range(maxdeg + 1):
            for a in range(deg + 1):
                b = deg - a
                lam.append(self.ra ** b * self.rb ** a / tot)
                mult.append(float(comb(mA + b - 1, b) * comb(mB + a - 1, a)))
        self.lam, self.mult = np.array(lam), np.array(mult)
        self.lam1 = 1.0 / tot                                # the constant mode
        self.tail = max(0.0, 1.0 - float(self.mult @ self.lam))   # trace above maxdeg
        self.family = {k: self.ra ** b * self.rb ** a / tot for k, a, b in ladder}
        self.ladder = ladder

    def d_eff(self, kappa):
        """sum_i lambda_i / (lambda_i + kappa): the mode count a kernel at this kappa resolves."""
        return float(self.mult @ (self.lam / (self.lam + kappa))) + self.tail / kappa

    def n_at(self, kappa, ridge_rel=0.0):
        """Sample count at which the conservation law gives this kappa (inf if unreachable)."""
        if ridge_rel >= kappa:
            return float("inf")
        return self.d_eff(kappa) / (1.0 - ridge_rel / kappa)

    def kappa(self, n, ridge_rel=0.0):
        """Log-bracketed solve of sum lambda/(lambda+k) + ridge_rel*n/k = n."""
        f = lambda k: self.d_eff(k) + ridge_rel * n / k - n
        lo, hi = 1e-60, 1e3
        for _ in range(300):
            mid = math.sqrt(lo * hi)
            lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
        return math.sqrt(lo * hi)

    def learnability(self, kind, n, ridge_rel=0.0):
        lf = self.family[kind]
        return lf / (lf + self.kappa(n, ridge_rel))

    def n_for(self, kind, L=0.5, ridge_rel=0.0):
        """Samples for a family to reach learnability L (kernel ridge, sample-limited)."""
        lf = self.family[kind]
        return self.n_at(lf * (1.0 - L) / L, ridge_rel)

    def P_for(self, kind, L=0.5):
        """Random features whose span holds the family at learnability ~L (d_eff at lambda/3 slack)."""
        lf = self.family[kind]
        return self.d_eff(lf * (1.0 - L) / L / 3.0)

    def L_ceiling(self, kind, P):
        """Highest learnability a P-feature model can give this family (inverse of P_for)."""
        lo, hi = 1e-9, 1.0 - 1e-12
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if self.P_for(kind, mid) < P else (lo, mid)
        return lo

    def steps_for(self, kind, eta, L=0.5):
        """Gradient-flow steps for a family to reach learnability L: ln(1/(1-L)) / (eta*lambda)."""
        return math.log(1.0 / (1.0 - L)) / (eta * self.family[kind])


def run_floor_sweep(cfg, T_list, pre_by_deg, n_samples=5, n_mc=5000, seed=0,
                    kind="normal", device=None, verbose=True):
    """Task-diversity sweep: kernel floors at each degree cutoff + dMMSE/Ridge oracles.

    One `betas` draw per (T, sample) is shared by every cutoff AND by both oracles.
    Returns a dict of (n_samples, len(T_list)) arrays, all normalized by cfg.norm2.
    """
    cutoffs = sorted(pre_by_deg)
    R = dict(floor_id={deg: np.full((n_samples, len(T_list)), np.nan) for deg in cutoffs},
             floor_ood={deg: np.full((n_samples, len(T_list)), np.nan) for deg in cutoffs})
    for k in ("dmmse_id", "dmmse_ood", "ridge_id", "ridge_ood"):
        R[k] = np.full((n_samples, len(T_list)), np.nan)
    R["T_list"] = list(T_list)
    R["cutoffs"] = cutoffs
    R["config"] = cfg

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    t_all = time.time()
    if verbose:
        print(f"floor sweep: {cfg.describe()}")
        print(f"  kind={kind}  cutoffs={cutoffs}  n_samples={n_samples}  n_mc={n_mc}")
    for ti, T in enumerate(T_list):
        t0 = time.time()
        for s in range(n_samples):
            try:
                betas = draw_betas(T, cfg, kind, device)
            except ValueError as ex:
                # the only legitimate skip: e.g. isogauss requires T > d for full-rank whitening.
                # Anything else is a bug and must not be silently turned into NaNs.
                if s == 0 and verbose:
                    print(f"  make_tasks('{kind}') skipped T={T}: {ex}")
                continue
            for deg in cutoffs:
                try:
                    fid, food = ood_floors_fast(pre_by_deg[deg], betas, cfg.sigma_beta2)
                except np.linalg.LinAlgError:
                    fid, food = np.nan, np.nan
                R["floor_id"][deg][s, ti] = fid
                R["floor_ood"][deg][s, ti] = food
            om = oracle_mse_one_betas(betas, cfg.dim, cfg.L_cfg, cfg.v, cfg.e,
                                      cfg.sigma_beta, cfg.norm2, n_mc, rng)
            for k in ("dmmse_id", "dmmse_ood", "ridge_id", "ridge_ood"):
                R[k][s, ti] = om[k]
        if verbose:
            top = max(cutoffs)
            print(f"  T={T:5d}: floor(deg<={top}) ID/OOD="
                  f"{np.nanmean(R['floor_id'][top][:, ti]):.3f}/{np.nanmean(R['floor_ood'][top][:, ti]):.3f}"
                  f"  dMMSE ID/OOD={np.nanmean(R['dmmse_id'][:, ti]):.3f}/"
                  f"{np.nanmean(R['dmmse_ood'][:, ti]):.3f}  ({time.time() - t0:.1f}s)")
    if verbose:
        print(f"total {time.time() - t_all:.1f}s")
    return R


def gap_summary(R, deg=3, thresh=0.05):
    """Where does the deg-cutoff kernel floor sit below the dMMSE OOD oracle?

    Returns (gap_per_T, best_T, best_gap, visible_T_window). This is the quantity that
    decides whether a config is worth training on at all: training can only approach the
    floor from above, so a config with no gap here cannot produce one empirically.
    """
    T_list = np.asarray(R["T_list"], float)
    gap = np.nanmean(R["dmmse_ood"], 0) - np.nanmean(R["floor_ood"][deg], 0)
    if np.all(np.isnan(gap)):
        return gap, None, np.nan, []
    i = int(np.nanargmax(gap))
    window = [int(t) for t, g in zip(T_list, gap) if np.isfinite(g) and g >= thresh]
    return gap, int(T_list[i]), float(gap[i]), window


def plot_side(R, side, cutoffs=None, title=None, ax=None, measured=None):
    """One panel (side='id' or 'ood') of the Figure-4 style task-diversity plot.

    `measured` is an optional list of (label, n_or_None, mean, sem, color) tuples for
    trained-model curves to overlay on the theory floors.
    """
    import matplotlib.pyplot as plt
    T_list = R["T_list"]
    cutoffs = R["cutoffs"] if cutoffs is None else cutoffs
    n_samples = R["floor_ood"][cutoffs[0]].shape[0]
    if ax is None:
        _, ax = plt.subplots(figsize=(7.4, 5.0))
    kcolors = plt.cm.Blues(np.linspace(0.55, 0.95, len(cutoffs)))
    for ci, deg in enumerate(cutoffs):
        m = np.nanmean(R[f"floor_{side}"][deg], 0)
        sd = np.nanstd(R[f"floor_{side}"][deg], 0)
        lab = f"kernel floor  deg<={deg}"
        if n_samples > 1:
            ax.errorbar(T_list, m, yerr=sd, marker="o", ms=4, lw=1.5,
                        color=kcolors[ci], capsize=2, label=lab)
        else:
            ax.plot(T_list, m, marker="o", ms=4, lw=1.5, color=kcolors[ci], label=lab)
    for k, lab, col, mk in [(f"dmmse_{side}", "dMMSE oracle", "crimson", "s"),
                            (f"ridge_{side}", "Ridge oracle", "black", "^")]:
        m = np.nanmean(R[k], 0)
        sd = np.nanstd(R[k], 0)
        if n_samples > 1:
            ax.errorbar(T_list, m, yerr=sd, marker=mk, ms=5, lw=1.8, color=col, capsize=2, label=lab)
        else:
            ax.plot(T_list, m, marker=mk, ms=5, lw=1.8, color=col, label=lab)
    for lab, mean, sem, col in (measured or []):
        ax.errorbar(T_list, mean, yerr=sem, marker="D", ms=4, lw=1.8, ls="--",
                    color=col, capsize=2, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("task diversity  T   (# pretraining tasks)")
    ax.set_ylabel(f"{side.upper()} test MSE  (normalized)")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    return ax



# ---------------------------------------------------------------------------
# parity with the source notebook
# ---------------------------------------------------------------------------

def test_parity(cfg=None, degs=(1, 3), Ts=(1, 8, 64), seed=0, rtol=1e-6, verbose=True):
    """Assert this module reproduces run_eigenlearning_OOD_floor_vs_taskdiversity.ipynb.

    Three checks:
      1. the fast path `ood_floors_fast` equals the reference `ood_inputs(..., pooled=False)`
      2. at T=1 the deg<=3 kernel floor equals the dMMSE oracle on BOTH ID and OOD
         (the kernel learns the single task exactly, so there is nothing left to generalize)
      3. the OOD-minus-ID gap shrinks monotonically as T grows
    """
    cfg = CONFIG_PAPER if cfg is None else cfg
    rng = np.random.default_rng(seed)
    ok = True

    # --- 1. fast path == reference path ---
    for deg in degs:
        marked = [f["label"] for f in build_features(cfg.dim, deg)]
        pre = precompute_ood(cfg.v, cfg.e, cfg.L, cfg.dim, marked, deg)
        for T in Ts:
            betas = rng.normal(0.0, cfg.sigma_beta, size=(T, cfg.dim))
            fid, food = ood_floors_fast(pre, betas, cfg.sigma_beta2)
            ref = ood_inputs(betas, ("gauss", cfg.sigma_beta2), cfg.v, cfg.e, cfg.L,
                             cfg.dim, marked, max_degree=deg, pooled=False)
            same = (np.allclose(fid, ref["noise_var_train_eff"], rtol=rtol, atol=1e-12)
                    and np.allclose(food, ref["floor"], rtol=rtol, atol=1e-12))
            ok &= same
            if verbose:
                print(f"  [fast==ref] deg<={deg} T={T:>4}: "
                      f"ID {fid:.6f} vs {ref['noise_var_train_eff']:.6f} | "
                      f"OOD {food:.6f} vs {ref['floor']:.6f}   {'ok' if same else 'MISMATCH'}")
            assert same, f"ood_floors_fast != ood_inputs at deg<={deg}, T={T}"

    # --- 2. at T=1 the kernel floor IS dMMSE ---
    marked3 = [f["label"] for f in build_features(cfg.dim, 3)]
    pre3 = precompute_ood(cfg.v, cfg.e, cfg.L, cfg.dim, marked3, 3)
    betas1 = rng.normal(0.0, cfg.sigma_beta, size=(1, cfg.dim))
    fid1, food1 = ood_floors_fast(pre3, betas1, cfg.sigma_beta2)
    om1 = oracle_mse_one_betas(betas1, cfg.dim, cfg.L_cfg, cfg.v, cfg.e,
                               cfg.sigma_beta, cfg.norm2, 20000, np.random.default_rng(seed + 1))
    close_id = abs(fid1 - om1["dmmse_id"]) < 0.02 * max(1.0, abs(fid1))
    close_ood = abs(food1 - om1["dmmse_ood"]) < 0.02 * max(1.0, abs(food1))
    ok &= close_id and close_ood
    if verbose:
        print(f"  [T=1 floor==dMMSE] ID {fid1:.4f} vs {om1['dmmse_id']:.4f} | "
              f"OOD {food1:.4f} vs {om1['dmmse_ood']:.4f}   "
              f"{'ok' if (close_id and close_ood) else 'MISMATCH'}")
    assert close_id and close_ood, "at T=1 the deg<=3 floor should equal the dMMSE oracle"

    # --- 3. OOD - ID gap shrinks with T ---
    gaps = []
    for T in (1, 4, 16, 64, 256):
        betas = rng.normal(0.0, cfg.sigma_beta, size=(T, cfg.dim))
        fid, food = ood_floors_fast(pre3, betas, cfg.sigma_beta2)
        gaps.append(food - fid)
    mono = all(gaps[i] >= gaps[i + 1] - 1e-9 for i in range(len(gaps) - 1))
    ok &= mono
    if verbose:
        print(f"  [OOD-ID gap decreasing] {['%.4f' % g for g in gaps]}   "
              f"{'ok' if mono else 'NOT MONOTONE'}")
    assert mono, f"OOD-ID gap should shrink with T, got {gaps}"

    if verbose:
        print("test_parity: PASS")
    return ok


if __name__ == "__main__":
    print(CONFIG_PAPER.describe())
    test_parity()
    for c in (CONFIG_A, CONFIG_B, CONFIG_C):
        print(c.describe())
