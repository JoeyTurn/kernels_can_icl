"""Shared pieces of the two transience notebooks (closed-form RFF ridge, and SGD):

  * the seeded task pool, injected into a RegressionTaskDistribution
  * the dMMSE / Ridge oracles on shared test contexts, and the normalized MSE
  * the reference numbers for one pool (channel-ladder floors + oracles)
  * the two figures -- OOD/ID error against n at a few T, and against T at a few n -- and a
    dip/rise summary table

Every record the notebooks produce has the same shape, so these work for both:

    dict(T=..., trial=..., ladder={kind: (ID floor, OOD floor)}, oracles={dmmse_id, dmmse_ood,
         ridge_id, ridge_ood}, curve=[{"n": ..., "id": ..., "ood": ..., ...}, ...])

`n` is n_train for the closed form and n_seen = step * batch for SGD.
"""
import glob
import math
import os
import pickle
import re

import numpy as np
import torch

import kicl_floor as KF

# The ladder floors drawn on the n-curves. The full ladder is in the records; these are the rungs
# where something visible happens (deg-1 mean, the Xyq dip, deg<=3, the kappa_3 rise, deg<=5).
LADDER_SHOWN = ("q", "Xy", "XX", "y3X", "yX_yX", "yXdotX")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def make_pool_distr(cfg, T, seed, pool_kind, device):
    """(td, betas): a RegressionTaskDistribution whose pretraining pool is a seeded `pool_kind` draw.

    The pool is drawn on the CPU so it is the same on Colab and locally. `betas` is float64 numpy of
    the exact float32 pool the model trains on, so floors, oracles and model all see one pool.
    """
    from kernels_can_icl.task_distrs import RegressionTaskDistribution
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 31))
    W = KF.draw_betas(T, cfg, pool_kind, torch.device("cpu"))
    td = RegressionTaskDistribution(
        dim=cfg.dim, num_weights=int(T), noise_variance=cfg.e, sample_beta="normal",
        sigma_beta2=cfg.sigma_beta2,
        sigma_x2=cfg.v,                    # NOT the package default 1/(d(L_cfg-1))
        device=device)
    td.pretrain_W = torch.as_tensor(W, dtype=torch.float32, device=device)
    return td, td.pretrain_W.double().cpu().numpy()


def gen_raw(td, cfg, n, distribution):
    """(X_flat, y_query, X_ctx, y_ctx, X_q). 'pretrain' = the pool, 'true' = fresh beta ~ N(0, I)."""
    Xc, yc, Xq, yq = td.generate_data(context_len=cfg.L_cfg, batch_sz=int(n),
                                      distribution=distribution,
                                      uniform_beta_sampling=(distribution == "pretrain"))
    return td._get_input(Xc, yc, Xq), yq.squeeze(-1).squeeze(-1), Xc, yc, Xq


def make_tests(td, cfg, sizes, seed):
    """{name: dict(X, y, Xc, yc, Xq)} for sizes = {name: (n, distribution)}.

    Drawn from a dedicated seed with the RNG streams restored afterwards, so the test sets are
    identical across a resume and the training stream is unaffected.
    """
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed + 987_654)
    out = {}
    for name, (n, dist) in sizes.items():
        X, y, Xc, yc, Xq = gen_raw(td, cfg, n, dist)
        out[name] = dict(X=X, y=y, Xc=Xc, yc=yc, Xq=Xq)
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    return out


# ---------------------------------------------------------------------------
# oracles and references
# ---------------------------------------------------------------------------

@torch.no_grad()
def oracle_preds(Xc, yc, Xq, betas, cfg, chunk=2048):
    """Ridge (posterior mean under N(0, sigma_beta2 I)) and dMMSE (posterior mean over the pool)
    predictions on given contexts. Xc (B,L,d), yc (B,L,1), Xq (B,1,d), betas (T,d)."""
    B = Xc.shape[0]
    lam = cfg.e / cfg.sigma_beta2
    eye = torch.eye(cfg.dim, device=Xc.device, dtype=torch.float64)
    out_r = torch.empty(B, dtype=torch.float64, device=Xc.device)
    out_d = torch.empty(B, dtype=torch.float64, device=Xc.device)
    bet = torch.as_tensor(betas, dtype=torch.float64, device=Xc.device)
    for i in range(0, B, chunk):
        xc = Xc[i:i + chunk].double()
        yv = yc[i:i + chunk].double().squeeze(-1)
        xq = Xq[i:i + chunk].double().squeeze(1)
        A = torch.einsum("bld,ble->bde", xc, xc) + lam * eye
        wR = torch.linalg.solve(A, torch.einsum("bld,bl->bd", xc, yv).unsqueeze(-1)).squeeze(-1)
        out_r[i:i + chunk] = torch.einsum("bd,bd->b", xq, wR)
        preds = torch.einsum("bld,td->blt", xc, bet)
        logit = -((yv.unsqueeze(-1) - preds) ** 2).sum(1) / (2.0 * cfg.e)
        sc = torch.softmax(logit - logit.max(1, keepdim=True).values, dim=1)
        out_d[i:i + chunk] = torch.einsum("bd,bd->b", xq, sc @ bet)
    return out_r, out_d


def nmse(pred, y, cfg):
    """Test MSE against the noisy query label, divided by norm2 = E[y_q^2] (the theory's axis)."""
    return float(((pred.double().squeeze() - y.double().squeeze()) ** 2).mean()) / cfg.norm2


def nmse_se(pred, y, cfg):
    """(normalized MSE, its standard error over the test contexts)."""
    sq = (pred.double().squeeze() - y.double().squeeze()) ** 2
    return float(sq.mean()) / cfg.norm2, float(sq.std() / math.sqrt(sq.numel())) / cfg.norm2


def predict_sgd(pieces, spec, eta, steps, P, batch):
    """Bias-only theory curve for one-pass SGD: family f is learned to L_f(t) = 1 - exp(-eta lambda_f t),
    capped at the learnability a P-feature span can hold. No SGD noise, no finite-P variance, and the
    i.i.d.-token spectrum: a prediction of where the curve should go, not a fit."""
    fams = pieces["fam"]
    caps = {f: spec.L_ceiling(f, P) for f in set(fams)}
    lam = np.array([spec.family[f] for f in fams])
    cap = np.array([caps[f] for f in fams])
    out = dict(n=[int(t) * batch for t in steps], id=[], ood=[])
    for t in steps:
        i, o = KF.score_learnability(pieces, np.minimum(1.0 - np.exp(-eta * lam * t), cap))
        out["id"].append(i)
        out["ood"].append(o)
    return out


def references(tests, betas, cfg, ladder_engine, id_key="id", ood_key="ood"):
    """Ladder floors for this pool, and both oracles on the SAME test contexts as the model."""
    ref = dict(ladder=KF.ladder_floors(ladder_engine, betas, cfg.sigma_beta2), oracles={})
    for side, key in (("id", id_key), ("ood", ood_key)):
        t = tests[key]
        pr, pd_ = oracle_preds(t["Xc"], t["yc"], t["Xq"], betas, cfg)
        ref["oracles"][f"ridge_{side}"] = nmse(pr, t["y"], cfg)
        ref["oracles"][f"dmmse_{side}"] = nmse(pd_, t["y"], cfg)
    return ref


# ---------------------------------------------------------------------------
# channel probe: which channels does the trained model actually implement?
# ---------------------------------------------------------------------------

K3_KINDS = ("y2XX", "yX_yX")                              # y_i y_j x_i x_j x_q: the kappa_3 family
DEG5_KINDS = ("y4", "y3X", "y2XX", "yX_yX", "yXdotX")     # every degree-5 family
KNOCKOUTS = {"k3": K3_KINDS, "deg5": DEG5_KINDS}
# Groups drawn in the implemented-fraction panel. Families with ~no target power are left out: their
# fraction is a ratio of two tiny numbers.
PROBE_GROUPS = (("mean  (q)", ("q",)),
                ("ICL  (x y q)", ("Xy",)),
                ("kappa_3  (y^2 x^3 q)", K3_KINDS),
                ("other degree 5", ("y4", "y3X", "yXdotX")))


def _leg_value(leg, Xc, yc):
    (px, py, c), = leg                             # one token-level monomial per leg
    v = torch.full(Xc.shape[:2], float(c), dtype=torch.float64, device=Xc.device)
    for a, pw in enumerate(px):
        if pw:
            v = v * Xc[:, :, a] ** pw
    return v * yc[:, :, 0] ** py if py else v      # (n, L)


@torch.no_grad()
def icl_features(feats, Xc, yc, Xq):
    """The ICL feature columns (n, F) of KF.build_features on contexts Xc (n,L,d), yc (n,L,1), Xq (n,1,d)."""
    Xc, yc, Xq = Xc.double(), yc.double(), Xq.double()
    cols = []
    for f in feats:
        (legs, coef), = f["terms"]
        xq = Xq[:, 0, f["b"]]
        if len(legs) == 0:
            s = torch.ones_like(xq)
        elif len(legs) == 1:
            s = _leg_value(legs[0], Xc, yc).sum(1)
        else:                                      # distinct tokens: sum_i sum_{j != i}
            a, c = _leg_value(legs[0], Xc, yc), _leg_value(legs[1], Xc, yc)
            s = a.sum(1) * c.sum(1) - (a * c).sum(1)
        cols.append(coef * s * xq)
    return torch.stack(cols, 1)


def ladder_ordered_features(dim, max_degree=5):
    """KF.build_features sorted into ladder (eigenvalue) order, query axes interleaved within a family.

    The probe runs ONE Gram-Schmidt over all of them in this order. Query axes are orthogonal in
    expectation; orthonormalizing them jointly also removes their O(1/sqrt N) sample overlap, which
    per-axis orthonormalization leaks between families (~0.01-0.02 at N = 400k).
    """
    rank = {k: i for i, (k, _, _) in enumerate(KF.LADDER)}
    kind = lambda f: KF._feat_coords(f["label"])[0]
    return sorted(KF.build_features(dim, max_degree), key=lambda f: rank.get(kind(f), len(rank)))


class ChannelProbe:
    """Reads off from a trained model's function how much of each channel family it implements.

    The degree<=5 ICL features are orthonormalized under the pretraining distribution, per query axis,
    in ladder (eigenvalue) order, so a family's coordinates are the part of it not already carried by
    the families before it. With a_t the coordinates of the Bayes (dMMSE) target -- which an
    infinite-data kernel converges to -- and a_f the model's, a family's implemented fraction is
    <a_f, a_t> / <a_t, a_t> over its coordinates: 0 before it is learned, 1 once it is.

    A knockout subtracts a family's coordinates from the model's own predictions and re-scores them,
    which attributes an error change to that family directly rather than through a theory curve.

    For a model linear in its weights, f = Phi w, the projection is precomputed once as
    Q = L^-1 F^T Phi / N (F x P), so each new w costs one matvec. `feature_map` is Phi (X_flat -> (c, P)).
    """

    @torch.no_grad()
    def __init__(self, td, cfg, betas, n_probe, seed, feature_map, tests, chunk=4096, max_degree=5,
                 sides=("id", "ood")):
        self.cfg, self.sides = cfg, tuple(sides)
        self.feats = ladder_ordered_features(cfg.dim, max_degree)
        self.kinds = [KF._feat_coords(f["label"])[0] for f in self.feats]

        pr = make_tests(td, cfg, {"probe": (n_probe, "pretrain")}, seed + 424_242)["probe"]
        dev = pr["X"].device
        Fp = torch.cat([icl_features(self.feats, pr["Xc"][i:i + 65536], pr["yc"][i:i + 65536],
                                     pr["Xq"][i:i + 65536]) for i in range(0, n_probe, 65536)])
        t = oracle_preds(pr["Xc"], pr["yc"], pr["Xq"], betas, cfg)[1]            # Bayes target, noise-free
        M = None
        for i in range(0, n_probe, chunk):
            Ph = feature_map(pr["X"][i:i + chunk]).double()
            M = Fp[i:i + chunk].T @ Ph if M is None else M.addmm_(Fp[i:i + chunk].T, Ph)
            del Ph
        M /= n_probe
        G = Fp.T @ Fp / n_probe
        bt = Fp.T @ t / n_probe
        del Fp, pr, t

        eye = torch.eye(G.shape[0], device=dev, dtype=G.dtype)
        Lc = torch.linalg.cholesky(G + 1e-12 * G.diagonal().mean() * eye)
        Li = torch.linalg.solve_triangular(Lc, eye, upper=False)
        self.Q = Li @ M                             # model weights -> coordinates in the orthonormal basis
        self.a_t = Li @ bt
        self.Linv_T = Li.T                          # raw features -> orthonormal basis (right-multiply)
        del M, Lc, Li

        self.Psi, self.y = {}, {}
        for s in self.sides:
            self.Psi[s] = icl_features(self.feats, tests[s]["Xc"], tests[s]["yc"], tests[s]["Xq"]) @ self.Linv_T
            self.y[s] = tests[s]["y"].double().squeeze()
        self._idx = {}

        # sanity: the Bayes target's degree<=3 / <=5 projections must reproduce the theory floors
        self.check = {}
        for name, mx in (("deg3", 3), ("deg5", max_degree)):
            ix = self.index([k for k in set(self.kinds) if KF._FEAT_DEG[k] <= mx])
            self.check[name] = tuple(nmse(self.Psi[s][:, ix] @ self.a_t[ix], self.y[s], cfg) for s in self.sides)

    def index(self, kinds):
        key = tuple(sorted(kinds))
        if key not in self._idx:
            self._idx[key] = torch.tensor([i for i, k in enumerate(self.kinds) if k in kinds],
                                          dtype=torch.long, device=self.Q.device)
        return self._idx[key]

    def summary(self):
        """Per-record constants: the coordinate kinds, the target's coordinates and the sanity check."""
        return dict(kinds=list(self.kinds), a_t=self.a_t.cpu().numpy(), check=self.check,
                    target_power={k: float((self.a_t[self.index([k])] ** 2).sum()) / self.cfg.norm2
                                  for k in dict.fromkeys(self.kinds)})

    @torch.no_grad()
    def measure(self, w, preds):
        """w: the model's weights (P,) or (P, 1). preds: {side: the model's predictions on tests[side]}.
        -> dict(coef=model coordinates a_f, ko={name: {side: error with that group knocked out}})."""
        a_f = self.Q @ w.reshape(-1).to(self.Q.dtype)
        ko = {}
        for name, kinds in KNOCKOUTS.items():
            ix = self.index(kinds)
            ko[name] = {s: nmse(preds[s].double().squeeze() - self.Psi[s][:, ix] @ a_f[ix], self.y[s], self.cfg)
                        for s in self.sides}
        # The model's own degree<=5 part, scored on its own. If the model's error is far BELOW its
        # projection's, the rest of the function (degree >= 6, which the ladder does not track) is
        # cancelling what those channels cost away from the pool -- so a family can read as fully
        # implemented while the curve stays below the floor that implies.
        proj = {s: nmse(self.Psi[s] @ a_f, self.y[s], self.cfg) for s in self.sides}
        rem = {s: float(((preds[s].double().squeeze() - self.Psi[s] @ a_f) ** 2).mean()) / self.cfg.norm2
               for s in self.sides}
        return dict(coef=a_f.float().cpu().numpy(), ko=ko, proj=proj, rem=rem)


def implemented(probe_summary, coef, kinds):
    """Implemented fraction of a group of families: <a_f, a_t> / <a_t, a_t> over their coordinates.

    NaN when the target carries almost nothing in the group (< 1e-4 of its total power: e.g. every
    family but the mean at T=1, where the Bayes predictor is linear), since the ratio is then noise."""
    a_all = np.asarray(probe_summary["a_t"], float)
    sel = np.array([k in kinds for k in probe_summary["kinds"]])
    a_t = a_all[sel]
    if a_t @ a_t < 1e-4 * (a_all @ a_all):
        return float("nan")
    return float(np.asarray(coef, float)[sel] @ a_t / (a_t @ a_t))


# ---------------------------------------------------------------------------
# loading saved results (for a separate plotting notebook)
# ---------------------------------------------------------------------------

def load_closed(path, device=torch.device("cpu")):
    """A closed-form results pickle -> (cfg, records, meta, info).

    info carries the kernel width and the switch-on markers. Files written before they were saved get
    a width recomputed from the reference pool on this machine (a different RNG stream than the GPU,
    so it can differ by ~1%) -- fine for placing markers, which is all it is used for.
    """
    with open(path, "rb") as f:
        blob = pickle.load(f)
    meta, records = blob["meta"], blob["records"]
    cfg = KF.KICLConfig(**meta["config"])
    info = dict(blob.get("extras", {}))
    if "width" not in info:
        td, _ = make_pool_distr(cfg, 64, meta.get("seed", 0), meta.get("pool", "whiten_keepmean"), device)
        X = gen_raw(td, cfg, 4000, "pretrain")[0].double()
        info["width"] = meta["width_factor"] * float(torch.cdist(X[:2000], X[2000:]).flatten().median())
        info["width_recomputed"] = True
    if "markers" not in info:
        info["markers"] = markers(cfg, info["width"])
    return cfg, records, meta, info


def load_sgd(resultpath, pattern="rf_transience_sgd_*_T*_t*.pt"):
    """The SGD runs, read from their per-run checkpoints (these exist even if the notebook was stopped
    before writing its summary). -> {tag: (cfg, records, info)}, one entry per setting, where tag is the
    filename up to _T{T}_t{trial} and info has P, batch, width, eta, pool and the markers."""
    groups = {}
    for p in sorted(glob.glob(os.path.join(resultpath, pattern))):
        m = re.search(r"^(rf_transience_sgd_.*)_T(\d+)_t(\d+)\.pt$", os.path.basename(p))
        if not m:
            continue
        st = torch.load(p, map_location="cpu", weights_only=False)
        fp = st["fingerprint"]
        cfg = KF.make_config(fp["dim"], fp["pairs"], fp["eps"])
        rec = dict(T=int(m.group(2)), trial=int(m.group(3)), seed=fp["seed"], step=int(st["step"]),
                   curve=st["curve"], **st["ref"])
        tag = m.group(1)
        if tag not in groups:
            info = dict(P=fp["P"], batch=fp["batch"], width=fp["width"], eta=fp["eta"],
                        pool=fp.get("pool"), markers=markers(cfg, fp["width"], fp["eta"], fp["batch"]))
            groups[tag] = (cfg, [], info)
        groups[tag][1].append(rec)
    return groups


def markers(cfg, width, eta=None, batch=None, k3="y2XX"):
    """Predicted switch-on points for the n-curves. Closed form (eta=None): n_train at which the ridgeless
    kernel reaches the learnability; SGD: n_seen = steps * batch under gradient flow at rate eta."""
    sp = KF.ChannelSpectrum(cfg, width)
    if eta is None:
        at = lambda kind, L: sp.n_for(kind, L)
    else:
        at = lambda kind, L: sp.steps_for(kind, eta, L) * batch
    return {"Xyq L=.5": at("Xy", 0.5), "XXq L=.9": at("XX", 0.9), "kappa_3 L=.5": at(k3, 0.5)}


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def _mean_sem(vals):
    vals = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    if vals.size == 0:
        return np.nan, np.nan
    return float(vals.mean()), float(vals.std() / math.sqrt(vals.size)) if vals.size > 1 else 0.0


def curve_stats(records, T, side):
    """(n, mean, sem) of the measured curve at task diversity T, over trials."""
    recs = [r for r in records if r["T"] == T and r.get("curve")]
    if not recs:
        return np.array([]), np.array([]), np.array([])
    ns = sorted({c["n"] for r in recs for c in r["curve"]})
    m, s = [], []
    for n in ns:
        mm, ss = _mean_sem([c[side] for r in recs for c in r["curve"] if c["n"] == n])
        m.append(mm)
        s.append(ss)
    return np.array(ns, float), np.array(m), np.array(s)


def ref_mean(records, T, key, side):
    """Mean over trials of a reference value: a ladder kind ('q', 'XX', ...) or an oracle name."""
    i = 0 if side == "id" else 1
    vals = []
    for r in records:
        if r["T"] != T:
            continue
        if key in r["ladder"]:
            vals.append(r["ladder"][key][i])
        else:
            vals.append(r["oracles"][f"{key}_{side}"])
    return _mean_sem(vals)[0]


def value_at(record, n, side):
    """Measured value at the curve point nearest n (log distance)."""
    c = min(record["curve"], key=lambda row: abs(math.log(row["n"]) - math.log(n)))
    return c[side]


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def plot_n_curves(records, cfg, Ts, markers=None, xlabel="n", title="", sides=("ood", "id"),
                  save=None):
    """Error against n at each T in Ts, with the ladder floors, dMMSE and Ridge as horizontal lines.

    markers: {label: n} vertical lines where a channel family is predicted to switch on.
    """
    import matplotlib.pyplot as plt
    Ts = [T for T in Ts if any(r["T"] == T for r in records)]
    if not Ts:
        print("no records for", Ts)
        return None
    fig, axs = plt.subplots(len(sides), len(Ts), figsize=(4.6 * len(Ts), 3.8 * len(sides)),
                            squeeze=False, sharex=True)
    lcols = dict(zip(LADDER_SHOWN, plt.cm.viridis(np.linspace(0.0, 0.85, len(LADDER_SHOWN)))))
    for ci, T in enumerate(Ts):
        for ri, side in enumerate(sides):
            ax = axs[ri, ci]
            n, m, s = curve_stats(records, T, side)
            ax.plot(n, m, "-o", ms=2.5, lw=2.0, color="C0", label="measured", zorder=5)
            if np.any(s > 0):
                ax.fill_between(n, m - s, m + s, color="C0", alpha=0.25, zorder=4)
            preds = [r["pred"] for r in records if r["T"] == T and r.get("pred")]
            if preds:
                pn = np.asarray(preds[0]["n"], float)
                pm = np.mean([p[side] for p in preds if len(p[side]) == len(pn)], 0)
                ax.plot(pn, pm, "--", lw=1.6, color="dimgray", label="theory (bias only)", zorder=6)
            for kind in LADDER_SHOWN:
                y = ref_mean(records, T, kind, side)
                ax.axhline(y, color=lcols[kind], lw=1.0, ls=":", alpha=0.9)
                ax.annotate(KF.LADDER_LABELS[kind], (1.0, y), xycoords=("axes fraction", "data"),
                            fontsize=6.5, color=lcols[kind], ha="right", va="bottom")
            for key, col, lab in (("dmmse", "crimson", "dMMSE"), ("ridge", "black", "Ridge")):
                ax.axhline(ref_mean(records, T, key, side), color=col, lw=1.6, ls="--", label=lab)
            for lab, x in (markers or {}).items():
                if len(n) and n.min() <= x <= n.max():
                    ax.axvline(x, color="gray", lw=0.9, ls="-.", alpha=0.7)
                    ax.annotate(lab, (x, 1.0), xycoords=("data", "axes fraction"), rotation=90,
                                fontsize=6.5, color="gray", ha="right", va="top")
            ax.set_xscale("log")
            ax.grid(alpha=0.25)
            if ri == 0:
                ax.set_title(f"T = {T}")
            if ri == len(sides) - 1:
                ax.set_xlabel(xlabel)
            if ci == 0:
                ax.set_ylabel(f"{side.upper()} test MSE (normalized)")
    axs[0, -1].legend(fontsize=7, loc="upper left")
    fig.suptitle(title or cfg.describe(), fontsize=10)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130, bbox_inches="tight")
    return fig


def plot_T_sweep(records, cfg, T_list, picks, title="", save=None):
    """Figure-4 style: error against T, with the measured model read off at a few n.

    picks: [(label, n)] -- each record contributes its curve point nearest n.
    """
    import matplotlib.pyplot as plt
    T_list = [T for T in T_list if any(r["T"] == T for r in records)]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    for ax, side, name in zip(axs, ("id", "ood"), ("In-distribution", "Out-of-distribution")):
        for kind, col in (("q", "#9ecae1"), ("XX", "#3182bd"), ("yXdotX", "#08519c")):
            ax.plot(T_list, [ref_mean(records, T, kind, side) for T in T_list], "o-", ms=3.5,
                    lw=1.3, color=col, label=f"kernel floor {KF.LADDER_LABELS[kind]}")
        for key, col, mk, lab in (("dmmse", "crimson", "s", "dMMSE"), ("ridge", "black", "^", "Ridge")):
            ax.plot(T_list, [ref_mean(records, T, key, side) for T in T_list], marker=mk, ms=4.5,
                    lw=1.7, color=col, label=lab)
        cols = plt.cm.autumn(np.linspace(0.0, 0.6, max(1, len(picks))))
        for (lab, n), col in zip(picks, cols):
            ms = [_mean_sem([value_at(r, n, side) for r in records if r["T"] == T]) for T in T_list]
            ax.errorbar(T_list, [a for a, _ in ms], yerr=[b for _, b in ms], marker="D", ms=4.5,
                        lw=2.0, ls="--", color=col, capsize=2, label=lab)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("task diversity T")
        ax.set_title(name)
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("test MSE (normalized)")
    axs[1].legend(fontsize=7.5, ncol=2)
    fig.suptitle(title or cfg.describe(), fontsize=10)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130, bbox_inches="tight")
    return fig


def _mean_sd(vals):
    vals = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    if vals.size == 0:
        return np.nan, np.nan
    return float(vals.mean()), float(vals.std()) if vals.size > 1 else 0.0


def probe_series(records, T):
    """{name: (n, mean, sd)} over trials at T, from the channel probe: 'ko_k3_ood', 'ko_deg5_id', ... and
    'frac:<group label>' for each PROBE_GROUPS entry. Empty if these runs have no probe."""
    recs = [r for r in records if r["T"] == T and r.get("probe") and r.get("curve")]
    if not recs:
        return {}
    ns = sorted({c["n"] for r in recs for c in r["curve"] if "probe" in c})
    rows = {}
    for n in ns:
        cs = [(r, c) for r in recs for c in r["curve"] if c["n"] == n and "probe" in c]
        vals = {f"ko_{k}_{s}": [c["probe"]["ko"][k][s] for _, c in cs] for k in KNOCKOUTS for s in ("id", "ood")}
        for key in ("proj", "rem"):                       # absent in runs made before the projection check
            if all(key in c["probe"] for _, c in cs):
                vals.update({f"{key}_{s}": [c["probe"][key][s] for _, c in cs] for s in ("id", "ood")})
        for lab, kinds in PROBE_GROUPS:
            vals[f"frac:{lab}"] = [implemented(r["probe"], c["probe"]["coef"], kinds) for r, c in cs]
        for name, v in vals.items():
            rows.setdefault(name, []).append((n,) + _mean_sd(v))
    return {k: tuple(np.array(x, float) for x in zip(*v)) for k, v in rows.items()}


def plot_channel_probe(records, cfg, T, xlabel="n", title="", save=None):
    """The U, and what the model implements along it. Top rows: OOD and ID error, measured and with
    the kappa_3 family (and all of degree 5) knocked out of the model's function. Bottom row: the
    implemented fraction of each channel group. Bands are +/- 1 SD over trials."""
    import matplotlib.pyplot as plt
    ps = probe_series(records, T)
    if not ps:
        print(f"no channel-probe data at T={T}")
        return None
    fig, axs = plt.subplots(3, 1, figsize=(6.2, 10.5), sharex=True,
                            gridspec_kw=dict(height_ratios=[1.15, 1.0, 1.0]))
    for ax, side in zip(axs[:2], ("ood", "id")):
        n, m, _ = curve_stats(records, T, side)
        sd = np.array([_mean_sd([c[side] for r in records if r["T"] == T for c in r.get("curve", [])
                                 if c["n"] == nn])[1] for nn in n])
        ax.plot(n, m, "-o", ms=3, lw=2.2, color="C0", label="measured", zorder=6)
        ax.fill_between(n, m - sd, m + sd, color="C0", alpha=0.2, zorder=5)
        for k, ls, col, lab in (("k3", "--", "C3", "kappa_3 family knocked out"),
                                ("deg5", ":", "C2", "all degree 5 knocked out")):
            kn, km, ks = ps[f"ko_{k}_{side}"]
            ax.plot(kn, km, ls, lw=1.9, color=col, label=lab, zorder=7)
            ax.fill_between(kn, km - ks, km + ks, color=col, alpha=0.12, zorder=4)
        if f"proj_{side}" in ps:            # the model's own degree<=5 part, scored on its own
            pn, pm, _ = ps[f"proj_{side}"]
            ax.plot(pn, pm, "-", lw=1.2, color="dimgray", alpha=0.9, label="its degree<=5 projection", zorder=3)
        for key, col, lab in (("dmmse", "crimson", "dMMSE"), ("ridge", "black", "Ridge")):
            ax.axhline(ref_mean(records, T, key, side), color=col, lw=1.4, ls="-", alpha=0.8, label=lab)
        for kind, lab in (("XX", "floor deg<=3"), ("yX_yX", "floor +kappa_3")):
            y = ref_mean(records, T, kind, side)
            ax.axhline(y, color="gray", lw=0.9, ls=(0, (1, 2)))
            ax.annotate(lab, (1.0, y), xycoords=("axes fraction", "data"), fontsize=7, color="gray",
                        ha="right", va="bottom")
        ax.set_ylabel(f"{side.upper()} test MSE (normalized)")
        ax.grid(alpha=0.25)
    axs[0].legend(fontsize=7.5, loc="best")
    ax = axs[2]
    cols = plt.cm.viridis(np.linspace(0.0, 0.85, len(PROBE_GROUPS)))
    for (lab, _), col in zip(PROBE_GROUPS, cols):
        fn, fm, fs = ps[f"frac:{lab}"]
        ax.plot(fn, fm, "-o", ms=2.5, lw=1.9, color=col, label=lab)
        ax.fill_between(fn, fm - fs, fm + fs, color=col, alpha=0.18)
    ax.axhline(1.0, color="black", lw=0.8, ls="--")
    ax.axhline(0.0, color="black", lw=0.6)
    ax.set_ylabel("implemented fraction\n<model, target> / <target, target>")
    ax.set_xlabel(xlabel)
    ax.set_xscale("log")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="upper left")
    axs[0].set_title(f"T = {T}")
    fig.suptitle(title or cfg.describe(), fontsize=9)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130, bbox_inches="tight")
    return fig


def probe_attribution(records, T_list, n_from=0):
    """How much of the OOD rise the kappa_3 family carries, per T (mean over trials; SD in brackets).

    From the OOD minimum (over n >= n_from) to the last n: the measured rise, the rise of the same model
    with its kappa_3 coordinates removed, and the share of the rise that removal accounts for. Also the
    kappa_3 implemented fraction at both ends, and what the family does in distribution at the last n
    (knocking it out should RAISE ID error: it is memorization, helpful on the pool).
    """
    print(f"{'T':>4} | {'k3 impl: at min':>15} {'at end':>7} | {'OOD rise':>13} {'rise w/o k3':>13} {'share':>6} |"
          f" {'ID end':>6} {'ID end w/o k3':>13} | {'OOD end':>7} {'its deg<=5 proj':>15}")
    for T in T_list:
        recs = [r for r in records if r["T"] == T and r.get("probe")]
        if not recs:
            continue
        out = {k: [] for k in ("f0", "f1", "rise", "rise_ko", "share", "id", "id_ko", "ood", "proj")}
        for r in recs:
            cv = [c for c in r["curve"] if "probe" in c]
            y = np.array([c["ood"] for c in cv])
            ok = np.array([c["n"] >= n_from for c in cv])
            i = int(np.flatnonzero(ok)[np.argmin(y[ok])]) if ok.any() else int(np.argmin(y))
            ko = np.array([c["probe"]["ko"]["k3"]["ood"] for c in cv])
            rise, rise_ko = y[-1] - y[i], ko[-1] - ko[i]
            out["f0"].append(implemented(r["probe"], cv[i]["probe"]["coef"], K3_KINDS))
            out["f1"].append(implemented(r["probe"], cv[-1]["probe"]["coef"], K3_KINDS))
            out["rise"].append(rise)
            out["rise_ko"].append(rise_ko)
            out["share"].append(1.0 - rise_ko / rise if rise > 0 else np.nan)
            out["id"].append(cv[-1]["id"])
            out["id_ko"].append(cv[-1]["probe"]["ko"]["k3"]["id"])
            out["ood"].append(y[-1])
            out["proj"].append(cv[-1]["probe"].get("proj", {}).get("ood", np.nan))
        f = {k: _mean_sd(v) for k, v in out.items()}
        s = lambda k, w=6: f"{f[k][0]:>{w}.3f}" + (f" ({f[k][1]:.3f})" if len(recs) > 1 else "")
        print(f"{T:>4} | {s('f0', 15)} {s('f1', 7)} | {s('rise', 6):>13} {s('rise_ko', 6):>13} "
              f"{f['share'][0]:>6.0%} | {f['id'][0]:>6.3f} {f['id_ko'][0]:>13.3f} | {f['ood'][0]:>7.3f} "
              f"{f['proj'][0]:>15.3f}")
    print("  A model whose own degree<=5 projection scores much WORSE than it does is cancelling those\n"
          "  channels' OOD cost with the rest of its function -- a family can then read as implemented\n"
          "  while the curve stays below the floor that implies.")


def probe_consistency(records, cfg, ladder_engine, T_list=None, at_n=None):
    """Self-check: push the MEASURED per-family fractions through the floor engine and compare with what
    the model's degree<=5 projection actually scored.

    The floor engine assumes each family is learned as (fraction x its target coefficients). If the two
    agree, the coordinates and the ladder are telling the same story, and any gap between the model and
    its projection is the model's degree >= 6 part. If they disagree, the fractions are not a scaling of
    the target -- read them as overlaps, not as 'how much is implemented'.
    """
    pre5 = ladder_engine[-1][1]
    order = [k for k, _, _ in KF.LADDER]
    for r in records:
        if (T_list is not None and r["T"] not in T_list) or not r.get("probe"):
            continue
        cv = [c for c in r["curve"] if "probe" in c and "proj" in c["probe"]]
        if not cv:
            continue
        c = cv[-1] if at_n is None else min(cv, key=lambda row: abs(math.log(row["n"] / at_n)))
        fr = {k: implemented(r["probe"], c["probe"]["coef"], (k,)) for k in order}
        pc = KF.ladder_pieces(pre5, np.asarray(r["betas"], float), cfg.sigma_beta2)
        L = np.array([0.0 if not np.isfinite(fr[f]) else fr[f] for f in pc["fam"]])
        t_id, t_ood = KF.score_learnability(pc, L)
        p_id, p_ood = c["probe"]["proj"]["id"], c["probe"]["proj"]["ood"]
        print(f"T={r['T']:>3} trial {r['trial']} at n={c['n']:,}: fractions " +
              " ".join(f"{KF.LADDER_LABELS[k]}={fr[k]:.2f}" for k in order if np.isfinite(fr[k])))
        print(f"    projection scored   ID {p_id:.3f}  OOD {p_ood:.3f}\n"
              f"    floor engine at those fractions ID {t_id:.3f}  OOD {t_ood:.3f}"
              f"   {'ok' if abs(p_ood - t_ood) < 0.05 and abs(p_id - t_id) < 0.03 else 'MISMATCH'}")
        print(f"    model itself        ID {c['id']:.3f}  OOD {c['ood']:.3f}   "
              f"(model - projection: ID {c['id'] - p_id:+.3f}, OOD {c['ood'] - p_ood:+.3f} = its degree >= 6 part)")


def transience_verdict(records, T_list, side="ood", n_from=0, require_rise=True):
    """Per run and per T: did the measured curve dip below dMMSE and then rise, by how many standard
    errors, and how does that compare with the theory curve for the same pool?

    min  = the curve's minimum over n >= n_from. Pass n_from past the degree-1 stage: at step 0 the
         model predicts 0 (OOD error 1.0), which at small T is already below dMMSE -- the memorizer is
         worse than predicting nothing there -- so an unrestricted minimum can be the untrained model.
    dip  = min(dMMSE, 1.0) - min   (beat both the memorizer and the trivial predictor)
    rise = (curve at the end) - min
    z    = value / (sqrt2 * se), se the test-set Monte-Carlo error of the curve points. The oracles are
         evaluated on the same contexts, so errors are correlated and this is conservative.
    """
    print(f"{side.upper()}: dip = min(dMMSE, null=1) - min, rise = end - min, over n >= {n_from:,.0f};"
          f"  [theory for the same pool]")
    ok_all = True
    for T in T_list:
        recs = [r for r in records if r["T"] == T and r.get("curve")]
        if not recs:
            continue
        print(f"\nT = {T}")
        dips, rises = [], []
        for r in recs:
            y = np.array([c[side] for c in r["curve"]])
            n = np.array([c["n"] for c in r["curve"]])
            se = np.array([c.get(f"{side}_se", np.nan) for c in r["curve"]])
            ok_n = n >= n_from
            i = int(np.flatnonzero(ok_n)[np.argmin(y[ok_n])]) if ok_n.any() else int(np.argmin(y))
            dm = r["oracles"][f"dmmse_{side}"]
            bar = min(dm, 1.0) if side == "ood" else dm
            dip, rise = bar - y[i], y[-1] - y[i]
            zd = dip / (math.sqrt(2) * se[i]) if np.isfinite(se[i]) else np.nan
            zr = rise / (math.sqrt(2) * max(se[i], se[-1])) if np.isfinite(se[i]) else np.nan
            line = (f"  trial {r['trial']}: dMMSE {dm:.3f}  min {y[i]:.3f} at n={int(n[i]):,}  end {y[-1]:.3f}"
                    f"  | dip {dip:+.3f} (z={zd:.0f})  rise {rise:+.3f} (z={zr:.0f})")
            if r.get("pred"):
                p = np.array(r["pred"][side])
                pn = np.asarray(r["pred"]["n"])
                pok = pn >= n_from
                j = int(np.flatnonzero(pok)[np.argmin(p[pok])]) if pok.any() else int(np.argmin(p))
                line += f"  | [theory: min {p[j]:.3f}, end {p[-1]:.3f}, dip {bar - p[j]:+.3f}, rise {p[-1] - p[j]:+.3f}]"
            print(line)
            dips.append(dip)
            rises.append(rise)
            ok_all &= (dip > 0) and ((rise > 0 and i < len(y) - 1) if require_rise else True)
        if len(recs) > 1:
            print(f"  mean over {len(recs)}: dip {np.mean(dips):+.3f} +/- {np.std(dips) / math.sqrt(len(dips)):.3f}"
                  f"   rise {np.mean(rises):+.3f} +/- {np.std(rises) / math.sqrt(len(rises)):.3f}")
    want = "a dip below dMMSE followed by a rise" if require_rise else "a dip below dMMSE"
    print("\nVERDICT:", f"every run shows {want}" if ok_all
          else f"at least one run does NOT show {want} -- look at it")
    return ok_all


def summary_table(records, T_list, side="ood"):
    """Per T: where the measured curve bottoms out, how far below dMMSE, how far it climbs after."""
    print(f"{'T':>4} {'dMMSE':>7} {'floor3':>7} {'floor5':>7} {'Ridge':>7} | {'min':>7} {'at n':>10}"
          f" {'final':>7} | {'dMMSE-min':>9} {'final-min':>9}")
    for T in T_list:
        n, m, _ = curve_stats(records, T, side)
        if not len(m) or np.all(np.isnan(m)):
            continue
        i = int(np.nanargmin(m))
        dm = ref_mean(records, T, "dmmse", side)
        print(f"{T:>4} {dm:>7.3f} {ref_mean(records, T, 'XX', side):>7.3f} "
              f"{ref_mean(records, T, 'yXdotX', side):>7.3f} {ref_mean(records, T, 'ridge', side):>7.3f} | "
              f"{m[i]:>7.3f} {int(n[i]):>10,} {m[-1]:>7.3f} | {dm - m[i]:>+9.3f} {m[-1] - m[i]:>+9.3f}")
