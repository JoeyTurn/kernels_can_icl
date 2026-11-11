"""Random features for the two exact symmetries of the in-context regression target.

The Bayes predictor is  E[y_q | Z] = beta_hat(X, y) . x_q, so the target is

  * LINEAR in the query x_q, and
  * EVEN in the context (X, y): flipping both signs leaves beta_hat unchanged.

The kernel built here has exactly those symmetries,

    k(z, z') = k_even(c, c') * (x_q . x_q'),   k_even = [k_gauss(c - c') + k_gauss(c + c')] / 2,

where c = [X.flatten, y.flatten] is the context. Its random-feature map is phase-free cosines times
the query,

    phi_{j,a}(z) = sqrt(1/M) cos(omega_j . c) * x_q[a],      omega_j ~ N(0, I / w^2),

so M context features give M*d parameters. Every mode outside this span carries zero target power --
a function that is not linear in x_q, or odd in the context, cannot correlate with beta_hat . x_q --
so nothing is lost and the feature budget lands where the target lives. Against a plain Gaussian
kernel on the flat context this cuts the features needed for the degree-3 channel by ~100x at long
contexts, which is what makes L >> d (where the degree-3 truncation is close to Ridge) affordable.

`StructuredSpectrum` sizes a run: which families a given M can hold, and how many steps each needs.
Eigenvalues are quoted RELATIVE to the context constant mode; the absolute scale is set by the
measured lambda_1, so the sizing never depends on the absolute normalization.
"""
import math

import numpy as np
import torch

import kicl_floor as KF


def context_of(Xc, yc):
    """[X.flatten, y.flatten] -- the kernel's argument. Xc (B,L,d), yc (B,L,1) -> (B, L*(d+1))."""
    return torch.cat([Xc.flatten(start_dim=1), yc.flatten(start_dim=1)], dim=1)


class StructuredSpectrum:
    """Ladder for k_even(context) x (query linear), from the Mehler per-coordinate ratios.

    A family with y-degree a and x-degree b has one x factor in the query, so its context part is
    y^a x^(b-1) and its eigenvalue relative to the context constant mode is rb^a * ra^(b-1).
    """

    def __init__(self, cfg, width, ladder=KF.LADDER, kmax=12):
        base = KF.ChannelSpectrum(cfg, width)        # only for the per-coordinate Mehler ratios
        self.ra, self.rb = base.ra, base.rb
        self.cfg, self.width = cfg, width
        self.family = {k: self.rb ** a * self.ra ** (b - 1) for k, a, b in ladder}
        Dx, Dy = cfg.L * cfg.dim, cfg.L              # context coordinates by variance class
        lam, mult = [], []
        for kx in range(kmax):
            for ky in range(kmax):
                if (kx + ky) % 2 or kx + ky > kmax:  # even context modes only
                    continue
                lam.append(self.ra ** kx * self.rb ** ky)
                mult.append(math.comb(Dx + kx - 1, kx) * math.comb(Dy + ky - 1, ky) if kx + ky else 1)
        self.lam = np.array(lam, float)
        self.mult = np.array(mult, float)

    def d_eff(self, kappa):
        """Context modes a kernel at this kappa resolves -- the feature count an RF needs to span them."""
        return float((self.mult * self.lam / (self.lam + kappa)).sum())

    def M_for(self, kind, L=0.9):
        """Context features whose span holds `kind` to learnability L."""
        return self.d_eff(self.family[kind] * (1.0 - L) / L)

    def kappa_at(self, M):
        """The kappa whose d_eff is M (log bisection): what M context features can resolve."""
        lo, hi = 1e-30, 1e3
        for _ in range(200):
            mid = math.sqrt(lo * hi)
            lo, hi = (mid, hi) if self.d_eff(mid) > M else (lo, mid)
        return math.sqrt(lo * hi)

    def L_ceiling(self, kind, M):
        """The most of `kind` an M-feature span can hold, however long you train."""
        lf = self.family[kind]
        return lf / (lf + self.kappa_at(M))

    def steps_for(self, kind, eta, lam1, L=0.9):
        """One-pass SGD steps for `kind` to reach L: 1 - exp(-eta * lambda * t), lambda = lam1 * relative."""
        lf = lam1 * self.family[kind]
        L = min(L, 1 - 1e-12)
        return math.log(1.0 / (1.0 - L)) / (eta * lf)

    def describe(self, M, eta=None, lam1=None, kinds=("q", "y2", "Xy", "XX", "y3X", "y2XX")):
        rows = []
        for k in kinds:
            r = [k, f"{self.family[k]:.3e}", f"{self.M_for(k, 0.9):,.0f}", f"{self.L_ceiling(k, M):.3f}"]
            if eta and lam1:
                r.append(f"{self.steps_for(k, eta, lam1, min(0.9, 0.999 * self.L_ceiling(k, M))):,.0f}")
            rows.append(r)
        return rows


class StructuredRF:
    """f(z) = sum_j cos(omega_j . c) * (U_j . x_q), trained by one-pass SGD on U (M x d).

    U absorbs the sqrt(1/M) of the feature map, so the induced kernel is exactly
    k_even(c, c') (x_q . x_q') and `c2 = 1/M` is the scale that appears in the update.
    """

    def __init__(self, M, D_ctx, dim, width, seed, device, dtype=torch.float32):
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        self.OmegaT = (torch.randn(M, D_ctx, generator=g) / float(width)).T.contiguous().to(device=device,
                                                                                            dtype=dtype)
        self.U = torch.zeros(M, dim, dtype=torch.float64, device=device)
        self.M, self.dim, self.device, self.dtype = M, dim, device, dtype
        self.c2 = 1.0 / M

    def features(self, C):
        """(B, D_ctx) -> (B, M) = cos(C Omega^T), built in place (two live blocks, not four)."""
        Z = C.to(self.dtype) @ self.OmegaT
        torch.cos(Z, out=Z)
        return Z

    @torch.no_grad()
    def predict(self, C, Xq, chunk=256):
        out = []
        for i in range(0, C.shape[0], chunk):
            psi = self.features(C[i:i + chunk]).double()
            out.append(((psi @ self.U) * Xq[i:i + chunk].double()).sum(1))
        return torch.cat(out)

    @torch.no_grad()
    def step(self, C, Xq, y, eta):
        """One SGD step on a fresh batch; returns the batch MSE before the update.

        Everything per-batch runs in `dtype` (float32) and only the (M x d) update is cast up: the
        ACCUMULATOR must be float64 because 1e8 tiny coherent updates are what build the faint
        channels, but each batch's gradient is buried in minibatch noise long before float32 rounding
        matters. Casting the (batch x M) block instead -- which is what this used to do -- costs
        ~33 MB of traffic per step at M=2^14 and dominates the step at these model sizes.
        """
        psi = self.features(C)
        xq = Xq.to(psi.dtype)
        r = (psi @ self.U.to(psi.dtype) * xq).sum(1) - y.to(psi.dtype)
        self.U -= (eta * self.c2 / C.shape[0]) * (psi.T @ (r[:, None] * xq)).double()
        return float((r * r).mean())

    @torch.no_grad()
    def top_eigenvalue(self, C, Xq, iters=30, chunk=256):
        """Power iteration on the operator the update applies -- so eta = frac * 2 / lambda_1 is exact."""
        v = torch.randn(self.M, self.dim, dtype=torch.float64, device=self.device)
        v /= v.norm()
        lam = 0.0
        for _ in range(iters):
            acc = torch.zeros_like(v)
            for i in range(0, C.shape[0], chunk):
                psi = self.features(C[i:i + chunk]).double()
                xq = Xq[i:i + chunk].double()
                acc += psi.T @ (((psi @ v) * xq).sum(1)[:, None] * xq)
            v = acc * (self.c2 / C.shape[0])
            lam = float(v.norm())
            v /= max(lam, 1e-300)
        return lam

    def state(self):
        return dict(U=self.U.cpu())

    def load(self, st):
        self.U = st["U"].to(self.device)
        return self
