"""Eigenspace overlap between a predicted (HEA / KICL) eigenbasis and an empirical kernel eigenbasis.

This is the Figure-2-bottom-panel diagnostic of "Predicting kernel regression learning curves from
only raw data statistics" (Karkada, Turnbull, Liu & Simon, arXiv:2510.14878).

Comparing two eigenbases mode-by-mode is meaningless wherever the spectrum is near-degenerate: the
individual eigenvectors of a degenerate block are arbitrary, so `phi_i` vs `phi_i_hat` differences
there should not be penalized.  The paper's fix is spectral binning -- divide R+ into
logarithmically spaced bins, treat all modes landing in one bin as a single near-degenerate
eigenspace, and compare the resulting *subspaces*:

    Overlap(i, j) = (1 / d_j^emp) * || Phi_i^th.T @ Phi_j^emp ||_F^2      (Equation 44)

for bins with d_i^th != 0 and d_j^emp != 0, undefined (drawn grey) otherwise.  Dividing by the
empirical bin dimension makes each COLUMN sum to 1 when the predicted basis is complete, so a
column reads as "what fraction of this empirical eigenspace do the predicted eigenspaces capture".
Concentration on the diagonal means theory and experiment agree.

The binning and plot styling here follow `expts/kicl_validation/show_hea_overlap.py`, which is the
paper's own implementation (its `block.sum() / len(col_inds)` is exactly Equation 44).  That file
is not importable from this repo -- it does `import config`, a module belonging to the HEA paper's
repository -- so the logic is ported here without the `config` dependency.

One deviation, and the reason `orthonormalize` defaults to True: Equation 44 treats `Phi_i` as a
BASIS for an eigenspace, which presumes orthonormal columns.  In the paper the predicted basis is
exactly orthonormal (Hermite polynomials under a Gaussian measure).  A basis evaluated on a finite
sample -- e.g. `kernel_theory_multidim.compute_basis` -- is only approximately orthonormal, and
columns that genuinely overlap each other would be double-counted by the Frobenius norm.  QR within
each bin fixes that: it preserves the span, so the result is the true subspace overlap.  Empirical
eigenvectors come out of a symmetric eigendecomposition already exactly orthonormal and are used
as-is.  Pass `orthonormalize=False` to reproduce `show_hea_overlap.py`.

One further deviation, deliberate: `np.digitize(lam, edges) - 1` returns `nbins` for any eigenvalue
`>= edges[-1]`, and the largest eigenvalue always equals `edges[-1]` exactly because that is how the
edges were built.  `show_hea_overlap.coarse_overlap_matrix` only iterates `range(nbins)`, so it
silently drops the top mode -- which in this setting is the constant mode, the single most important
one.  `spectral_bins` folds that index back into the top bin instead.  Aside from this, with
`orthonormalize=False` the two implementations agree to ~1e-16.
"""

import math
import numpy as np

try:                                    # keep this module importable without torch
    from kernels_can_icl.utils import ensure_numpy
except Exception:                       # pragma: no cover
    def ensure_numpy(x, dtype=np.float64, clone=False):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x, dtype=dtype)


def _positive_lambda_bounds(*arrays):
    """Smallest and largest strictly-positive value across `arrays`, with sane fallbacks.

    Ported from show_hea_overlap.py so the bin edges match the paper's.
    """
    positives = []
    for arr in arrays:
        arr_np = np.asarray(arr, dtype=np.float64).reshape(-1)
        arr_pos = arr_np[arr_np > 0]
        if arr_pos.size:
            positives.append(arr_pos)
    if not positives:
        return 1e-12, 1.0
    merged = np.concatenate(positives)
    lam_min = max(float(np.min(merged)), 1e-12)
    lam_max = max(float(np.max(merged)), lam_min * 10.0)
    if not np.isfinite(lam_max) or lam_max <= lam_min:
        lam_max = lam_min * 10.0
    return lam_min, lam_max


def spectral_bins(th_eigvals, emp_eigvals, nbins=40, bin_edges=None):
    """Shared logarithmic bin edges over the union of both spectra, plus each spectrum's bin index.

    Both spectra must already be on the SAME scale before calling this -- for a kernel matrix that
    usually means `emp_eigvals / n`, and whatever prefactor the theory needs (e.g. `th_eigvals * pp`
    for a Laplace/Gaussian kernel).  Binning two differently-scaled spectra silently shears the
    heatmap off the diagonal, which is the easiest way to get a wrong-looking figure from correct
    eigenvectors.

    Returns (bin_edges, th_bins, emp_bins); bin index -1 or >= nbins means "outside the range".
    """
    th_eigvals = np.asarray(th_eigvals, dtype=np.float64)
    emp_eigvals = np.asarray(emp_eigvals, dtype=np.float64)
    if bin_edges is None:
        lam_min, lam_max = _positive_lambda_bounds(th_eigvals, emp_eigvals)
        bin_edges = np.geomspace(lam_min, lam_max, num=(nbins + 1))
    else:
        bin_edges = np.asarray(bin_edges, dtype=np.float64)
        nbins = len(bin_edges) - 1
    th_bins = np.digitize(th_eigvals, bin_edges) - 1
    emp_bins = np.digitize(emp_eigvals, bin_edges) - 1
    # np.digitize puts anything >= the last edge into bin `nbins`; fold it into the top bin so the
    # largest eigenvalue is not silently dropped.
    th_bins[th_bins == nbins] = nbins - 1
    emp_bins[emp_bins == nbins] = nbins - 1
    return bin_edges, th_bins, emp_bins


def eigenspace_overlap(th_eigvecs, th_eigvals, emp_eigvecs, emp_eigvals,
                       nbins=40, orthonormalize=True, bin_edges=None):
    """Binned subspace overlap (Equation 44) between a predicted and an empirical eigenbasis.

    Args:
        th_eigvecs: (N, P_th) predicted eigenfunctions evaluated on the sample, one per column,
            unit-L2-norm -- e.g. the output of kernel_theory_multidim.compute_basis.
        th_eigvals: (P_th,) predicted eigenvalues, already on the empirical scale.
        emp_eigvecs: (N, P_emp) empirical kernel eigenvectors (orthonormal columns), e.g. from
            kernel_eigsys.eigendecompose.
        emp_eigvals: (P_emp,) empirical eigenvalues, on the same scale as th_eigvals.
        nbins: number of logarithmic spectral bins.
        orthonormalize: QR the predicted columns within each bin before the Frobenius norm.
        bin_edges: supply to reuse another run's edges (e.g. to compare two configs on one scale).

    Returns a dict:
        overlap:    (nbins, nbins) float64, NaN where either bin is empty.  Row = predicted bin,
                    column = empirical bin, matching the paper's axes.
        bin_edges:  (nbins + 1,)
        d_th, d_emp: (nbins,) int, how many modes landed in each bin.
        col_sums:   (nbins,) sum down each column.  Should be <= 1; a shortfall is the predicted
                    basis being truncated to P_th modes rather than complete.
        cond:       (nbins,) per-bin min|diag(R)| / max|diag(R)| from the QR, NaN where the bin is
                    empty or orthonormalize=False.  Near zero means the predicted columns in that
                    bin were nearly linearly dependent, so its dimension is effectively lower than
                    d_th says.
    """
    Th = np.asarray(ensure_numpy(th_eigvecs), dtype=np.float64)
    Emp = np.asarray(ensure_numpy(emp_eigvecs), dtype=np.float64)
    th_eigvals = np.asarray(ensure_numpy(th_eigvals), dtype=np.float64).reshape(-1)
    emp_eigvals = np.asarray(ensure_numpy(emp_eigvals), dtype=np.float64).reshape(-1)

    if Th.ndim != 2 or Emp.ndim != 2:
        raise ValueError(f"eigvec matrices must be 2-D, got {Th.shape} and {Emp.shape}")
    if Th.shape[0] != Emp.shape[0]:
        raise ValueError(f"predicted and empirical eigenvectors must live on the same sample: "
                         f"{Th.shape[0]} rows vs {Emp.shape[0]}")

    # Trim each side to however many modes we have BOTH a vector and an eigenvalue for.
    P_th = min(Th.shape[1], th_eigvals.size)
    P_emp = min(Emp.shape[1], emp_eigvals.size)
    Th, th_eigvals = Th[:, :P_th], th_eigvals[:P_th]
    Emp, emp_eigvals = Emp[:, :P_emp], emp_eigvals[:P_emp]

    bin_edges, th_bins, emp_bins = spectral_bins(th_eigvals, emp_eigvals, nbins, bin_edges)
    nbins = len(bin_edges) - 1

    th_rows = [np.where(th_bins == i)[0] for i in range(nbins)]
    emp_cols = [np.where(emp_bins == j)[0] for j in range(nbins)]
    d_th = np.array([len(r) for r in th_rows], dtype=int)
    d_emp = np.array([len(c) for c in emp_cols], dtype=int)

    overlap = np.full((nbins, nbins), np.nan)
    cond = np.full(nbins, np.nan)

    for i, rows in enumerate(th_rows):
        if len(rows) == 0:
            continue
        block = Th[:, rows]
        if orthonormalize:
            # QR preserves span(block), so this measures the overlap of the SUBSPACE rather than
            # of a particular (possibly non-orthogonal) set of columns.
            Q, R = np.linalg.qr(block)
            diag = np.abs(np.diag(R))
            if diag.size and diag.max() > 0:
                cond[i] = float(diag.min() / diag.max())
        else:
            Q = block
        # (d_i, P_emp) squared cosines between this predicted eigenspace and every empirical mode.
        P2 = (Q.T @ Emp) ** 2
        for j, cols in enumerate(emp_cols):
            if len(cols) == 0:
                continue
            overlap[i, j] = P2[:, cols].sum() / len(cols)     # Equation 44

    with np.errstate(invalid="ignore"):
        col_sums = np.nansum(overlap, axis=0)
    col_sums[d_emp == 0] = np.nan

    # --- scale diagnostics -------------------------------------------------------------------
    # The heatmap is very sensitive to the two spectra being on different scales: a constant factor
    # between them shears the bright band off the diagonal, which looks exactly like the theory
    # getting the eigenbasis wrong.  These two numbers separate "wrong scale" from "wrong basis".
    k = min(th_eigvals.size, emp_eigvals.size)
    th_sorted = np.sort(th_eigvals)[::-1][:k]
    emp_sorted = np.sort(emp_eigvals)[::-1][:k]
    ok = (th_sorted > 0) & (emp_sorted > 0)
    lambda_ratio_median = float(np.median(th_sorted[ok] / emp_sorted[ok])) if ok.any() else np.nan

    offsets = []
    for j in range(nbins):
        col = overlap[:, j]
        if d_emp[j] == 0 or not np.isfinite(col).any():
            continue
        offsets.append(int(np.nanargmax(col)) - j)
    bin_offset_median = float(np.median(offsets)) if offsets else np.nan

    return {
        "overlap": overlap,
        "bin_edges": bin_edges,
        "d_th": d_th,
        "d_emp": d_emp,
        "col_sums": col_sums,
        "cond": cond,
        "lambda_ratio_median": lambda_ratio_median,
        "bin_offset_median": bin_offset_median,
    }


def gap_edges(th_eigvals, emp_eigvals, min_ratio=1.02):
    """Bin edges placed in the multiplicative GAPS of the pooled spectrum -- one bin per plateau.

    Log-uniform edges slice through the exactly-degenerate plateaus of an isotropic-data spectrum,
    splitting one eigenspace across two bins and smearing the diagonal.  Cutting at the gaps
    instead keeps each plateau whole.  Smaller `min_ratio` -> more bins.
    """
    pooled = np.sort(np.concatenate([np.asarray(th_eigvals, float),
                                     np.asarray(emp_eigvals, float)]))[::-1]
    pooled = pooled[pooled > 0]
    if len(pooled) < 2:
        return np.array([pooled.min() * .999, pooled.max() * 1.001])
    cuts = [math.sqrt(a * b) for a, b in zip(pooled[:-1], pooled[1:]) if a / b > min_ratio]
    return np.concatenate([[pooled.min() * .999], np.unique(cuts), [pooled.max() * 1.001]])


def merge_one_sided(edges, th_eigvals, emp_eigvals):
    """Drop interior edges until every bin holds modes from BOTH spectra -- zero grey cells.

    gap_edges cuts on the POOLED spectrum, so wherever the two spectra differ slightly you get a
    sliver bin holding only one side, which renders as an all-NaN row or column.  This merges those
    away; what survives is the resolution both spectra actually support.
    """
    edges = np.asarray(edges, float).copy()
    while len(edges) > 2:
        ct, _ = np.histogram(th_eigvals, bins=edges)
        ce, _ = np.histogram(emp_eigvals, bins=edges)
        bad = np.where((ct == 0) | (ce == 0))[0]
        if bad.size == 0:
            break
        i = int(bad[0])
        edges = np.delete(edges, i + 1 if i + 1 < len(edges) - 1 else i)
    return edges


def build_edges(th_eigvals, emp_eigvals, lam_floor, split,
                min_ratio_head=1.02, n_coarse=2):
    """Fine bins above `split`, `n_coarse` wide bins below, none one-sided.

    The plot gives every bin the same pixel size, so canvas share == BIN share, not eigenvalue
    span.  Concentrating bins above `split` is therefore what buys resolution at large eigenvalues
    while compressing the low-lambda tail -- where, with a degree-capped basis, the empirical
    eigenspaces contain higher-degree content the theory cannot represent anyway.
    """
    th = np.asarray(th_eigvals, float)
    emp = np.asarray(emp_eigvals, float)
    tv, ev = th[th >= lam_floor], emp[emp >= lam_floor]
    fine = gap_edges(tv[tv >= split], ev[ev >= split], min_ratio_head)
    fine[0] = split
    coarse = np.geomspace(min(tv.min(), ev.min()) * 0.999, split, n_coarse + 1)
    return merge_one_sided(np.concatenate([coarse[:-1], fine]), tv, ev)


def compact_bins(result):
    """Drop every bin that is empty on BOTH sides, returning a new result dict over what is left.

    Log-uniform bins over a staircase spectrum leave whole bins with no modes at all on either
    side.  Those carry no information -- they are grey in both their row and their column -- and on
    an ICL spectrum they can be most of the figure.  Removing them shrinks the grid without
    disturbing the diagonal, because the same bins are dropped from rows and from columns, so bin
    i still faces bin i.

    Bins populated on exactly ONE side are kept: "theory puts modes here, experiment puts none"
    (or vice versa) is a real disagreement and should stay visible as a grey line.

    `kept_bins` indexes back into the original grid; `bin_edges` is left as the full original edge
    array so those indices remain meaningful.
    """
    keep = (result["d_th"] > 0) | (result["d_emp"] > 0)
    if not keep.any():
        return dict(result, kept_bins=np.where(keep)[0])
    idx = np.where(keep)[0]
    out = dict(result)
    out["overlap"] = result["overlap"][np.ix_(idx, idx)]
    for key in ("d_th", "d_emp", "col_sums", "cond"):
        if key in result:
            out[key] = np.asarray(result[key])[idx]
    out["kept_bins"] = idx
    return out


def plot_eigenspace_overlap(result, title="eigenvector overlap", outpath=None, ax=None,
                            show_axis_values=False, title_fontsize=10, title_width=58,
                            drop_empty=True, descending=True):
    """Figure-2-bottom-panel heatmap of a `eigenspace_overlap` result.

    Styling follows show_hea_overlap.plot_empirical_vs_theory_eigvecs: magma, empty bins grey,
    origin at lower-left so eigenvalues increase up and to the right, colour fixed to [0, 1].

    Args:
        result: dict from eigenspace_overlap, or a bare (nbins, nbins) array.
        title: figure title (the paper's version reads this off a config module; pass it in).
        outpath: if given, save the figure there.
        ax: draw into an existing axis instead of making a new figure.
        show_axis_values: label the axes with the bin eigenvalue decades instead of leaving them
            bare.  Needs `result` to be the dict form.
        drop_empty: hide bins that are empty on both sides (see compact_bins).  On by default --
            on a staircase spectrum they are most of the grid and they bury the diagonal.  Set
            False for the literal full log-uniform grid.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if drop_empty and isinstance(result, dict) and "d_th" in result:
        result = compact_bins(result)

    M = result["overlap"] if isinstance(result, dict) else np.asarray(result, dtype=np.float64)
    if M.size == 0:
        raise ValueError("empty overlap matrix -- nothing to plot")

    cmap = mpl.colormaps.get_cmap("magma").copy()
    cmap.set_bad(color="#333333")

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    # eigenspace_overlap returns bins ASCENDING in lambda.  descending=True flips both axes so the
    # largest eigenvalue is top-left and the diagonal runs top-left -> bottom-right, reading the
    # same direction as an eigenvalue-vs-index plot.
    if descending:
        M = M[::-1, ::-1]
    image = ax.matshow(M, aspect="equal", cmap=cmap, interpolation="nearest",
                       origin="upper" if descending else "lower", vmin=0.0, vmax=1.0)
    # These titles carry the whole config (kernel, T, d, L, n) and readily run past the axes into
    # the colorbar, so wrap rather than let it overflow.
    import textwrap
    ax.set_title("\n".join(textwrap.wrap(str(title), title_width)), fontsize=title_fontsize)
    if show_axis_values and isinstance(result, dict):
        edges = result["bin_edges"]
        centers = np.sqrt(edges[:-1] * edges[1:])
        # bin_edges always describes the FULL grid, so select the surviving bins when compacted.
        if "kept_bins" in result:
            centers = centers[result["kept_bins"]]
        if descending:
            centers = centers[::-1]
        ticks = np.linspace(0, len(centers) - 1, min(6, len(centers))).astype(int)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([f"{centers[t]:.0e}" for t in ticks], rotation=45, fontsize=7)
        ax.set_yticklabels([f"{centers[t]:.0e}" for t in ticks], fontsize=7)
        ax.xaxis.set_ticks_position("bottom")
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(left=False, bottom=False)
    ax.set_xlabel("Empirical eigenspace")
    ax.set_ylabel("Predicted eigenspace")

    parent = fig if fig is not None else ax.figure
    cbar = parent.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("overlap", fontsize=9)
    cbar.ax.set_yticks([0.0, 0.5, 1.0])

    if fig is not None:
        fig.tight_layout()
    if outpath is not None:
        parent.savefig(outpath, dpi=200)
    return ax


def summarize_overlap(result, top_k=10):
    """One-line-per-quantity text summary: how much of the figure is trustworthy and why.

    `diag_mass` is the mean of the diagonal over bins where both sides are populated -- the single
    number that says "does the theory predict the eigenbasis".  `col_sums` short of 1 means the
    predicted basis was truncated, not that it is wrong.
    """
    M = result["overlap"]
    d_th, d_emp, col_sums, cond = result["d_th"], result["d_emp"], result["col_sums"], result["cond"]
    both = (d_th > 0) & (d_emp > 0)
    diag = np.array([M[i, i] for i in range(M.shape[0])])
    lines = [
        f"bins: {M.shape[0]} total, {int(both.sum())} populated on both sides "
        f"({int((d_th > 0).sum())} predicted, {int((d_emp > 0).sum())} empirical)",
        f"modes binned: {int(d_th.sum())} predicted, {int(d_emp.sum())} empirical",
    ]
    if both.any():
        lines.append(f"diagonal overlap on populated bins: mean={np.nanmean(diag[both]):.3f} "
                     f"min={np.nanmin(diag[both]):.3f} max={np.nanmax(diag[both]):.3f}")
        head = np.where(both)[0][-top_k:]          # highest-eigenvalue populated bins
        lines.append(f"diagonal over the top {len(head)} populated bins: "
                     + " ".join(f"{diag[i]:.2f}" for i in reversed(head)))
    finite_cols = col_sums[np.isfinite(col_sums)]
    if finite_cols.size:
        lines.append(f"column sums (1.0 = predicted basis complete): "
                     f"min={finite_cols.min():.3f} median={np.median(finite_cols):.3f} "
                     f"max={finite_cols.max():.3f}")
        if finite_cols.max() > 1.02:
            # QR makes each bin's predicted basis orthonormal internally, but says nothing about
            # different bins.  If the predicted columns were globally orthonormal every column
            # would sum to at most 1, so an excess is precisely cross-bin non-orthogonality: two
            # predicted eigenspaces both claiming the same empirical direction.
            lines.append(
                f"  ^^ sums above 1 mean the predicted eigenspaces are not orthogonal ACROSS "
                f"bins, so some empirical directions are claimed twice ({finite_cols.max():.2f} "
                f"= up to {100 * (finite_cols.max() - 1):.0f}% over-count). Within a bin QR "
                f"handles it; across bins it is a property of the predicted basis.")
        if finite_cols.min() < 0.9:
            lines.append(
                "  ^^ sums below 1 are truncation, not error: the predicted basis was cut off at "
                "a finite mode budget, so it cannot fully span those empirical eigenspaces.")
    finite_cond = cond[np.isfinite(cond)]
    if finite_cond.size:
        worst = int(np.nanargmin(np.where(np.isfinite(cond), cond, np.inf)))
        lines.append(f"worst predicted-bin conditioning min|R_ii|/max|R_ii| = "
                     f"{finite_cond.min():.2e} (bin {worst}, d_th={d_th[worst]}); "
                     f"small values mean near-dependent predicted columns in that bin")

    ratio = result.get("lambda_ratio_median", np.nan)
    offset = result.get("bin_offset_median", np.nan)
    if np.isfinite(ratio) or np.isfinite(offset):
        lines.append(f"SCALE: median lambda_th/lambda_emp at matched index = {ratio:.3f}; "
                     f"median bin offset of the brightest row per column = {offset:+.1f}")
        if np.isfinite(offset) and abs(offset) >= 1.0:
            lines.append(
                "  ^^ the bright band is off the diagonal by a roughly constant number of bins, "
                "which is what a SPECTRUM SCALE MISMATCH looks like -- not a wrong eigenbasis. "
                "Check that both spectra were passed on the same scale (empirical eigenvalues "
                "divided by n, theory times any kernel prefactor) and that the kernel's level "
                "coefficients were fitted for THIS dim/L (they depend on sum(data_eigvals)).")
    return "\n".join(lines)
