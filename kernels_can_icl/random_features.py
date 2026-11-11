"""Random Fourier feature ridge regression: the primal counterpart of `kernels.GaussianKernel`.

Why this exists
---------------
Exact KRR forms an n x n kernel matrix, so memory is O(n^2) and the solve is O(n^3). On an
A100 that caps n near 25k, which is well short of the sample counts needed to drive the
degree-3 ICL channels to learnability ~1 (see the KICL paper, Section G: the fit there is
not empirical because one cannot store enough data nor invert a kernel that large).

Random Fourier features move the problem into a P-dimensional primal space:

    phi(z) = sqrt(2/P) * cos(Omega z + b),   Omega ~ N(0, I/w^2),  b ~ U[0, 2pi]

which is an unbiased estimator of the Gaussian kernel exp(-||z - z'||^2 / 2w^2). Fitting is
then a P x P ridge solve, and Phi^T Phi / Phi^T y can be accumulated over streamed
minibatches, so memory is O(P^2) -- independent of n. n = 1e7 is routine.

The trade is that a P-dimensional readout can represent at most P modes, so P becomes the
new resolution ceiling and has to clear the number of kernel eigenmodes above kappa_n.
Sweep P and check the metric of interest has stopped moving before reading a finite-P
result as a statement about the kernel.

Notes
-----
* The normal equations are accumulated in float64 by default, with the features in float64 too,
  so the Gram is built by a DGEMM straight into the accumulator with no P x P temporary. On an
  A100 that is free (FP64 tensor cores match non-TF32 SGEMM at 19.5 TFLOP/s). Set
  gram_dtype=torch.float32 on hardware with crippled FP64.
* TF32 carries only ~10 mantissa bits. It is off by default for matmul in PyTorch, but this
  class warns if it has been switched on, since the P x P solve is sensitive to it.
* solve() does not consume the accumulator, so a nested grid of n can be snapshot-solved
  from a single pass over the stream.
* The P x P budget is 16 bytes per element on CUDA: the accumulator plus the working copy that
  cuSOLVER's Cholesky makes of its input. There is no in-place factorization on GPU -- LAPACK
  overwrites its input so a CPU solve can be in place, but the CUDA path clones regardless of
  `out=`. Size P against 16 bytes/element, not 8.
"""
import math
import warnings
from typing import Optional

import torch

from .utils import ensure_torch


class RFFKernelRegressor:
    """RFF approximation of Gaussian-kernel KRR, with the same ridge, width and fit/predict API.

    Two usage modes:

        one-shot:   rff.fit(X, y)
        streaming:  rff.reset()
                    for Xs, ys in shards: rff.accumulate(Xs, ys)
                    rff.solve()                      # repeatable at any point in the stream

    Args:
        ridge: the same delta as GaussianKernel(ridge=...), added to the diagonal of
            Phi^T Phi. Note Phi^T Phi grows with n, so a fixed delta is a real regularizer at
            small n and effectively ridgeless at large n.
        kernel_width: w in exp(-||z - z'||^2 / 2w^2).
        num_features: P.
        chunk: rows per feature minibatch; peak activation memory is chunk * P.
        seed: fixes (Omega, b). Two instances with the same seed, P and width use identical
            features.
    """

    _JITTERS = (0.0, 1e-10, 1e-8, 1e-6, 1e-4)

    def __init__(self, ridge, kernel_width, num_features=4096, chunk=20000, seed=0,
                 gram_dtype=torch.float64):
        self.ridge = float(ridge)
        self.kernel_width = float(kernel_width)
        self.P = int(num_features)
        self.chunk = int(chunk)
        self.seed = int(seed)
        self.gram_dtype = gram_dtype
        self.Omega = None
        self.b = None
        self.w = None
        self.device = None
        self._A = None          # Phi^T Phi, float64
        self._rhs = None        # Phi^T y,   float64
        self._count = 0

    # -- feature map ---------------------------------------------------------

    def _init_features(self, d, device):
        # Drawn in float32 from a CPU generator and THEN cast. Requesting float64 from the
        # generator consumes a different random stream, which would silently change the feature
        # draw -- and so the model -- when gram_dtype changes. Omega is a random draw, so its own
        # precision is irrelevant; what matters is that the Gram is accumulated in float64.
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        Om = torch.randn(self.P, d, generator=g) / self.kernel_width
        bb = torch.rand(self.P, generator=g) * (2.0 * math.pi)
        self.Omega = Om.to(device=device, dtype=self.gram_dtype)
        self.b = bb.to(device=device, dtype=self.gram_dtype)

    def _features(self, X):
        """X: (c, d) -> (c, P) in gram_dtype.

        Built in place after the matmul. Written as the plain expression
        `sqrt(2/P) * cos(X @ Omega.T + b)` this allocates four chunk x P temporaries (matmul,
        broadcast add, cos, scale) with two live at once -- 5.4 GB at P=81920, chunk=4096, where
        one block is 2.7 GB. Verified bit-identical to the expression form.
        """
        Z = X.to(self.gram_dtype) @ self.Omega.T
        Z += self.b
        torch.cos(Z, out=Z)
        Z *= math.sqrt(2.0 / self.P)
        return Z

    def _ensure_state(self, d, device):
        if self.Omega is not None:
            return
        if device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32:
            warnings.warn(
                "torch.backends.cuda.matmul.allow_tf32 is True. TF32 has ~10 mantissa bits, "
                "which is not enough for a near-ridgeless P x P normal-equation solve. "
                "Set it to False, or verify agreement against a float32 run.",
                RuntimeWarning, stacklevel=3)
        self._init_features(d, device)
        self.device = device
        self._A = torch.zeros(self.P, self.P, dtype=torch.float64, device=device)
        self._rhs = torch.zeros(self.P, 1, dtype=torch.float64, device=device)
        self._count = 0

    # -- streaming fit -------------------------------------------------------

    def reset(self):
        """Forget the accumulated normal equations, keeping hyperparameters and seed."""
        self.Omega = None
        self.b = None
        self.w = None
        self.device = None
        self._A = None
        self._rhs = None
        self._count = 0
        return self

    @torch.no_grad()
    def accumulate(self, X, y):
        """Add a shard to the running Phi^T Phi / Phi^T y."""
        X = ensure_torch(X, dtype=torch.float32)
        y = ensure_torch(y, dtype=torch.float32).reshape(-1, 1)
        self._ensure_state(X.shape[1], X.device)
        for i in range(0, X.shape[0], self.chunk):
            Z = self._features(X[i:i + self.chunk])
            yb = y[i:i + self.chunk].to(self.gram_dtype)
            if self.gram_dtype == torch.float64:
                # DGEMM straight into the accumulator: no P x P temporary at all. On an A100 the
                # FP64 tensor cores run DGEMM at 19.5 TFLOP/s, the same rate as non-TF32 SGEMM,
                # so this is free. It is also ~9 orders of magnitude more accurate than routing
                # through a float32 product.
                self._A.addmm_(Z.T, Z)
                self._rhs.addmm_(Z.T, yb)
            else:
                # Fallback for GPUs with crippled FP64 (T4, L4, consumer cards). add_ promotes the
                # float32 product in-kernel; writing `+= (Z.T @ Z).double()` would materialize an
                # extra P x P float64 temporary (8.6 GB at P=32768).
                self._A.add_(Z.T @ Z)
                self._rhs.add_(Z.T @ yb)
        self._count += X.shape[0]
        return self

    @torch.no_grad()
    def solve(self):
        """Solve (Phi^T Phi + ridge I) w = Phi^T y at the current accumulation.

        Non-destructive: the diagonal is restored afterwards from a saved copy, so this may be
        called repeatedly as the stream grows, and re-called with a different self.ridge to sweep
        the ridge without re-accumulating.

        MEMORY: a GPU solve needs TWO P x P blocks -- the accumulator and the Cholesky factor
        (cuSOLVER writes it to a new tensor regardless of `out=`). Two things would add a THIRD
        block and are avoided: an explicit scratch buffer (so the ridge goes onto the
        accumulator's own diagonal and is removed again), and torch.cholesky_solve, which clones
        the factor (so the back-substitution is two solve_triangular calls). Budget 16 bytes per
        P^2 element.
        """
        assert self._A is not None, "nothing accumulated; call accumulate() or fit() first"
        rhs = self._rhs
        diag_scale = max(float(torch.diagonal(self._A).mean().item()), 1.0)
        diag0 = self._A.diagonal().clone()          # P elements: exact restore, no P x P copy
        oom = getattr(torch.cuda, "OutOfMemoryError", None)
        last_err = None
        for jitter in self._JITTERS:
            try:
                self._A.diagonal().copy_(diag0)
                self._A.diagonal().add_(self.ridge + jitter * diag_scale)
                chol = torch.linalg.cholesky(self._A)
                # NOT torch.cholesky_solve: it clones the factor into a column-major working copy,
                # a THIRD P x P block (28 GiB at P=61,440) on top of the accumulator and the factor.
                # solve_triangular borrows a row- or column-contiguous factor as-is, so the two
                # triangular solves below allocate only P-vectors. Results are identical.
                z = torch.linalg.solve_triangular(chol, rhs, upper=False)
                self.w = torch.linalg.solve_triangular(chol.mT, z, upper=True)
                del chol, z
                self._A.diagonal().copy_(diag0)
                return self
            except Exception as err:
                self._A.diagonal().copy_(diag0)
                if oom is not None and isinstance(err, oom):
                    # Not a conditioning failure -- retrying with more jitter cannot help, and
                    # reporting it as "not factorizable" would send you after the wrong bug.
                    need = 2 * self.P * self.P * 8 / 2 ** 30
                    raise RuntimeError(
                        f"Out of memory factorizing a {self.P} x {self.P} float64 matrix. "
                        f"A solve needs two P x P blocks ({need:.1f} GiB) plus the feature "
                        f"chunk, because cuSOLVER clones its input. Lower num_features."
                    ) from err
                if not isinstance(err, RuntimeError):
                    raise
                last_err = err
        raise RuntimeError(
            f"RFF normal equations not factorizable even with jitter {self._JITTERS[-1]}"
        ) from last_err

    @torch.no_grad()
    def fit(self, X, y):
        self.reset()
        self.accumulate(X, y)
        return self.solve()

    def fitted(self):
        return self.w is not None

    @torch.no_grad()
    def predict(self, X, batch_size: Optional[int] = None):
        assert self.fitted(), "Call fit() or solve() first."
        X = ensure_torch(X, dtype=torch.float32)
        bs = batch_size or self.chunk
        outs = []
        for i in range(0, X.shape[0], bs):
            outs.append(self._features(X[i:i + bs]).double() @ self.w)
        return torch.cat(outs, dim=0).float()

    # -- bookkeeping ---------------------------------------------------------

    def __repr__(self):
        return (f"RFFKernelRegressor(P={self.P}, width={self.kernel_width:g}, "
                f"ridge={self.ridge:g}, seed={self.seed}, n_seen={self._count})")
