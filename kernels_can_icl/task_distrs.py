import torch
import math
from einops import einsum

class RegressionTaskDistribution:
    def __init__(self, dim, num_weights, noise_variance, sample_beta='stratified',
                 device=None, gram_wanted=False, sigma_x2=None, sigma_beta2=1.0, sigma_q2=None, context_len=1,
                 gen=None):
        """
        Initialize regression task distribution.
        
        Args:
            dim: Dimensionality of input features
            num_weights: Number of different weight vectors in pretrain distribution
            noise_variance: Variance of Gaussian noise added to y
            sample_beta: Distribution to sample weights from ('normal' or 'uniform')
            device: Torch device
            gram_wanted: Whether to compute Gram matrix representation
            sigma_x2: Variance for context input features. If None, defaults to
                1 / (dim * L_scale), matching the old built-in X normalization.
            sigma_beta2: Variance for regression weights (normal case only)
            sigma_q2: Variance for query input features. If None, defaults to sigma_x2.
            gen: optional torch.Generator for reproducibility (must be passed if sample_beta='uniform')
        """
        assert sample_beta in ("normal", "gaussian", "uniform", "stratified", "quantile"), "beta may only be sampled from normal or uniform distributions"

        self.d = dim
        self.T = num_weights
        self.sigma = noise_variance ** .5
        self.sample_beta = sample_beta
        self.sigma_x2 = 1.0 / (self.d * context_len) if sigma_x2 is None else float(sigma_x2)
        self.sigma_beta2 = float(sigma_beta2)
        self.sigma_q2 = 1.0 / (self.d * context_len) if sigma_q2 is None else float(sigma_q2)
        if self.sigma_x2 is not None:
            assert self.sigma_x2 >= 0, "sigma_x2 must be nonnegative"
        assert self.sigma_beta2 >= 0, "sigma_beta2 must be nonnegative"
        if self.sigma_q2 is not None:
            assert self.sigma_q2 >= 0, "sigma_q2 must be nonnegative"
        self.sigma_beta = self.sigma_beta2 ** 0.5
        self.gram_wanted = gram_wanted
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sigma_x2 = sigma_x2
        self.sigma_q2 = sigma_q2
        self.gen = gen
        self.clip_sd = None

        if sample_beta == 'uniform':
            self.pretrain_W = torch.rand(self.T, self.d, device=self.device, generator=gen)
        elif sample_beta in ("gaussian", "normal"):
            self.pretrain_W = torch.normal(
                mean=0.0,
                std=self.sigma_beta,
                size=(self.T, self.d),
                device=self.device,
                generator=gen
            )
        elif sample_beta in ("stratified", "quantile"):
            if sample_beta == "stratified":
                jit = 0.02 + 0.96 * torch.rand((self.T, self.d), device=self.device, generator=gen,
                                            dtype=torch.float64)
            elif sample_beta == "quantile":
                jit = torch.full((self.T, self.d), 0.5, device=self.device, dtype=torch.float64)
            strata = torch.stack(
                [torch.randperm(self.T, device=self.device, generator=gen) for _ in range(self.d)],
                dim=1).to(torch.float64)
            u = (strata + jit) / self.T
            W = self.sigma_beta * math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
            self.pretrain_W = W.to(torch.float32)
        else:
            raise ValueError(f"unknown scheme {sample_beta!r}")
        if self.clip_sd is not None:
            self.pretrain_W = self.pretrain_W.clamp(-self.clip_sd * self.sigma_beta, self.clip_sd * self.sigma_beta)


        # Store test targets for metric computation
        self.y_test = None
        self.y_train = None

    def _resolve_input_stds(self, L_scale):
        """
        Return (sigma_x, sigma_q) for a given context-length scale.

        By default, this matches the legacy behavior where X was sampled from N(0, I)
        and then divided by sqrt(d) and sqrt(L_scale).
        """
        assert L_scale >= 1, "L_scale must be >= 1"
        self.sigma_x2 = self.sigma_x2 if self.sigma_x2 is not None else 1.0 / (self.d * L_scale)
        self.sigma_q2 = self.sigma_q2 if self.sigma_q2 is not None else self.sigma_x2
        return self.sigma_x2 ** 0.5, self.sigma_q2 ** 0.5

    def generate_data(self, context_len, batch_sz, distribution="pretrain", uniform_beta_sampling=False,
                      norm_X=False):
        """
        Generate regression data with Gaussian noise.
        
        Args:
            context_len: Number of examples (context + query)
            batch_sz: Batch size
            distribution: 'pretrain' or 'true'
            uniform_beta_sampling: If True and distribution='pretrain', sample each
                pretrain_W roughly batch_sz/T times (stratified) rather than i.i.d.
                Order is still randomized. Default: False.
        
        Returns:
            X_ctx, y_ctx, X_q, y_q
        """
        assert distribution in ("pretrain", "true")
        L = context_len
        sigma_x, sigma_q = self._resolve_input_stds(L-1)

        # Draw context/query inputs iid with separate variances.
        X_ctx = torch.randn(batch_sz, L - 1, self.d, device=self.device, generator=self.gen) * sigma_x
        X_ctx = X_ctx / torch.linalg.norm(X_ctx, dim=1, keepdim=True) if norm_X else X_ctx
        X_q = torch.randn(batch_sz, 1, self.d, device=self.device, generator=self.gen) * sigma_q
        X = torch.cat([X_ctx, X_q], dim=1)

        if distribution == 'pretrain':
            if uniform_beta_sampling:
                # Tile [0..T-1] enough times to cover batch_sz, truncate, then shuffle
                repeats = (batch_sz + self.T - 1) // self.T
                idxs = torch.arange(self.T, device=self.device).repeat(repeats)[:batch_sz]
                idxs = idxs[torch.randperm(batch_sz, device=self.device, generator=self.gen)]
            else:
                idxs = torch.randint(0, self.T, (batch_sz,), device=self.device, generator=self.gen)
            W = self.pretrain_W[idxs]  # shape [B, d]
        else:
            if self.sample_beta == 'uniform':
                W = torch.rand(batch_sz, self.d, device=self.device, generator=self.gen)
            else:
                W = torch.normal(
                    mean=0.0,
                    std=self.sigma_beta,
                    size=(batch_sz, self.d),
                    device=self.device,
                    generator=self.gen
                )

        # Compute y = X @ W + noise
        noise = torch.randn(batch_sz, L, device=self.device, generator=self.gen) * self.sigma
        y = torch.einsum('BLd,Bd->BL', X, W) + noise  # shape [B, L]
        y = y[:, :, None] #made to be O(1) labels

        X_ctx, X_q = X[:, :-1, :], X[:, -1:, :]
        y_ctx, y_q = y[:, :-1, :], y[:, -1:, :]

        return X_ctx, y_ctx, X_q, y_q

    def _get_input(self, X, y, Xq):
        """Transform data into the required input format."""
        if not self.gram_wanted:
            input = torch.cat([X.flatten(start_dim=1), y.flatten(start_dim=1), Xq.flatten(start_dim=1)], dim=1)
        else:
            X_concat = torch.cat([X, Xq], dim=1)
            input = einsum(X_concat, X_concat, "b l d, b m d -> b l m")
            input = torch.cat([input.flatten(start_dim=1), y.flatten(start_dim=1)], dim=1)

        return input

