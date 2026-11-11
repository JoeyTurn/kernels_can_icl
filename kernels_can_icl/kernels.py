import math
from typing import Optional
import torch


def kernel_matrix(X: torch.Tensor, kernel_class, kernel_params: dict) -> torch.Tensor:
    if isinstance(kernel_class, str):
        kernel_class = {"PolynomialKernel": PolynomialKernel, "GaussianKernel": GaussianKernel}[kernel_class]
    
    if kernel_class is PolynomialKernel:
        max_degree = kernel_params.get("max_degree", 3)
        coeffs = kernel_params.get("coeffs", None)
        if coeffs is None:
            coeffs = torch.tensor([1.0 / math.factorial(ell) for ell in range(max_degree + 1)])
        elif not isinstance(coeffs, torch.Tensor):
            coeffs = torch.tensor(coeffs)
        return polynomial_kernel(X, X, coeffs.to(device=X.device, dtype=X.dtype), max_degree)
    
    elif kernel_class is GaussianKernel:
        return gaussian_kernel(X, X, kernel_params["kernel_width"])
    
    raise ValueError(f"Unsupported kernel class: {kernel_class}")


def eigendecompose(K: torch.Tensor):
    """
    Eigendecompose a symmetric PSD matrix K.
    Returns eigenvalues and eigenvectors sorted in descending order.
    
    Returns:
        eigvals: (n,) eigenvalues
        eigvecs: (n, n) eigenvectors as columns
    """
    eigvals, eigvecs = torch.linalg.eigh(K)  # eigh assumes symmetric, returns ascending order
    eigvals = eigvals.flip(0)
    eigvecs = eigvecs.flip(1)
    return eigvals, eigvecs


def polynomial_kernel(X: torch.Tensor, Z: torch.Tensor, coeffs: torch.Tensor, max_degree: int) -> torch.Tensor:
    """
    Compute polynomial kernel: K(x, z) = sum_{ell=0}^{max_degree} c_ell * (<x, z>)^ell
    
    X: (n, d)
    Z: (m, d)
    coeffs: (max_degree + 1,) tensor of coefficients [c_0, c_1, ..., c_{max_degree}]
    
    Returns: (n, m) kernel matrix
    """
    # Compute inner products: (n, m)
    inner_products = X @ Z.T
    
    # Compute kernel as sum of powers weighted by coefficients
    K = torch.zeros_like(inner_products)
    for ell in range(max_degree + 1):
        K = K + coeffs[ell] * (inner_products ** ell)
    
    return K


# ----------------------------
# KRR with Polynomial kernel
# ----------------------------
class PolynomialKernel:
    def __init__(self, ridge: float, max_degree: int = 3, coeffs: Optional[torch.Tensor] = None, **kwargs):
        """
        ridge: ridge parameter λ >= 0
        max_degree: maximum degree of the polynomial
        coeffs: coefficients [c_0, c_1, ..., c_{max_degree}]. If None, defaults to 1/ell! for ell >= 0.
        """
        self.ridge = float(ridge)
        self.max_degree = max_degree
        
        # Default coefficients: 1/ell!
        if coeffs is None:
            import math
            self.coeffs = torch.tensor([1.0 / math.factorial(ell) for ell in range(max_degree + 1)])
        else:
            assert len(coeffs) == max_degree + 1, f"coeffs must have length {max_degree + 1}"
            self.coeffs = coeffs if isinstance(coeffs, torch.Tensor) else torch.tensor(coeffs)
        
        self.Xtr = None
        self.alpha = None  # (n_train, d_out)
        self._chol = None  # Cholesky factor of (K + λI)

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        X: (n, d), y: (n,) or (n, d_out). Types: float32/float64; device: cpu/cuda.
        """
        if y.dim() == 1:
            y = y.unsqueeze(1)
        assert X.shape[0] == y.shape[0], "X and y must have same n"

        self.Xtr = X
        
        # Move coefficients to same device and dtype as X
        coeffs = self.coeffs.to(device=X.device, dtype=X.dtype)
        
        K = polynomial_kernel(X, X, coeffs, self.max_degree)
        n = K.shape[0]
        K = K + self.ridge * torch.eye(n, device=K.device, dtype=K.dtype)

        # Cholesky solve: (K) alpha = y
        L = torch.linalg.cholesky(K)                 # K = L L^T
        z = torch.cholesky_solve(y, L)               # returns K^{-1} y
        self.alpha = z                               # alpha = (K)^{-1} y
        self._chol = L
        return self

    @torch.no_grad()
    def predict(self, X: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Predict y_hat at X. Returns shape (n_eval, d_out). Uses batching for K(X, Xtr) if requested.
        """
        assert self.Xtr is not None and self.alpha is not None, "Call fit() first."
        
        # Move coefficients to same device and dtype as X
        coeffs = self.coeffs.to(device=X.device, dtype=X.dtype)
        
        if batch_size is None:
            Kxe = polynomial_kernel(X, self.Xtr, coeffs, self.max_degree)
            return Kxe @ self.alpha
        # Batched to keep memory in check
        outs = []
        for i in range(0, X.shape[0], batch_size):
            xb = X[i:i+batch_size]
            Kxb = polynomial_kernel(xb, self.Xtr, coeffs, self.max_degree)
            outs.append(Kxb @ self.alpha)
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def fitted(self) -> torch.Tensor:
        """Return in-sample fitted values on Xtr."""
        return self.predict(self.Xtr)
def _solve_kernel_system(K: torch.Tensor, y: torch.Tensor):
    """
    Solve K alpha = y for a symmetric kernel matrix.

    Prefer Cholesky with tiny diagonal jitter retries, which is much more stable
    for nearly-PSD Gram matrices than jumping directly to a generic least-squares solve.
    """
    K = 0.5 * (K + K.T)
    n = K.shape[0]
    eye = torch.eye(n, device=K.device, dtype=K.dtype)

    # Scale jitter to the matrix magnitude so it stays negligible unless needed.
    diag_scale = torch.trace(K).abs() / max(n, 1)
    diag_scale = float(diag_scale.item()) if torch.isfinite(diag_scale) else 1.0
    diag_scale = max(diag_scale, 1.0)

    jitters = [0.0, 1e-10, 1e-8, 1e-6, 1e-4]
    last_err = None
    for jitter_scale in jitters:
        try:
            K_solve = K if jitter_scale == 0.0 else K + (jitter_scale * diag_scale) * eye
            L = torch.linalg.cholesky(K_solve)
            alpha = torch.cholesky_solve(y, L)
            return alpha, L
        except RuntimeError as err:
            last_err = err

    # Final fallback for badly conditioned systems.
    alpha = torch.linalg.pinv(K) @ y
    return alpha, None


def relu_ntk_kernel(
    X: torch.Tensor,
    Z: torch.Tensor,
    weight_variance: float = 1.0,
    bias_variance: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    ReLU NTK (infinite-width, 1-hidden-layer) with standard weight/bias variances.

    X: (n, d), Z: (m, d)
    Returns: (n, m) kernel matrix.

    Uses the common convention:
      qx_i = b_var + w_var * ||x_i||^2
      qz_j = b_var + w_var * ||z_j||^2
      cov_ij = b_var + w_var * <x_i, z_j>
    """
    assert X.dim() == 2 and Z.dim() == 2 and X.shape[1] == Z.shape[1]
    w_var = float(weight_variance)
    b_var = float(bias_variance)

    # qx_i, qz_j
    qx = b_var + w_var * (X * X).sum(dim=1, keepdim=True)          # (n, 1)
    qz = b_var + w_var * (Z * Z).sum(dim=1, keepdim=True)          # (m, 1)

    # cov_ij
    cov = b_var + w_var * (X @ Z.t())                              # (n, m)

    # cos(theta_ij) = cov / sqrt(qx qz)
    denom = torch.sqrt(qx) * torch.sqrt(qz).t()                    # (n, m)
    denom = denom.clamp_min(eps)

    cos_theta = (cov / denom).clamp(-1.0 + 1e-7, 1.0 - 1e-7)       # keep acos stable
    theta = torch.acos(cos_theta)                                  # (n, m)

    # NNGP term: E[ReLU(u_x) ReLU(u_z)]
    # = sqrt(qx qz) * (1/(2π)) * (sinθ + (π-θ)cosθ)
    sqrt_q = denom
    nngp = sqrt_q * (1.0 / (2.0 * math.pi)) * (
        torch.sin(theta) + (math.pi - theta) * torch.cos(theta)
    )

    # Gate correlation: E[1_{u_x>0} 1_{u_z>0}] = (π-θ)/(2π)
    gate = (math.pi - theta) * (1.0 / (2.0 * math.pi))

    # NTK = NNGP + cov * gate
    ntk = nngp + cov * gate
    return ntk


# ----------------------------
# KRR with ReLU NTK kernel
# ----------------------------
class ReluNTK:
    def __init__(self, ridge: float, weight_variance: float = 1.0, bias_variance: float = 1.0,
    eps: float = 1e-12, **kwargs):
        """
        ridge: ridge parameter λ >= 0
        weight_variance: w_var (standard ReLU NTK setup)
        bias_variance: b_var (standard ReLU NTK setup)
        eps: small value for denominator clamps in kernel eval
        """
        self.ridge = float(ridge)
        self.weight_variance = float(weight_variance)
        self.bias_variance = float(bias_variance)
        self.eps = float(eps)

        self.Xtr = None
        self.alpha = None  # (n_train, d_out)

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        X: (n, d), y: (n,) or (n, d_out). Types: float32/float64; device: cpu/cuda.
        """
        if y.dim() == 1:
            y = y.unsqueeze(1)
        assert X.shape[0] == y.shape[0], "X and y must have same n"

        self.Xtr = X
        K = relu_ntk_kernel(
            X, X,
            weight_variance=self.weight_variance,
            bias_variance=self.bias_variance,
            eps=self.eps,
        )
        n = K.shape[0]
        if self.ridge != 0.0:
            K = K + self.ridge * torch.eye(n, device=K.device, dtype=K.dtype)

        # Prefer Cholesky when possible; fall back to least-squares if not PD / singular.
        try:
            L = torch.linalg.cholesky(K)
            alpha = torch.cholesky_solve(y, L)  # K^{-1} y
        except RuntimeError:
            # Works even if K is only PSD or singular.
            alpha = torch.linalg.lstsq(K, y).solution

        self.alpha = alpha
        return self

    @torch.no_grad()
    def predict(self, X: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Predict y_hat at X. Returns shape (n_eval, d_out). Uses batching for K(X, Xtr) if requested.
        """
        assert self.Xtr is not None and self.alpha is not None, "Call fit() first."

        if batch_size is None:
            Kxe = relu_ntk_kernel(
                X, self.Xtr,
                weight_variance=self.weight_variance,
                bias_variance=self.bias_variance,
                eps=self.eps,
            )
            return Kxe @ self.alpha

        outs = []
        for i in range(0, X.shape[0], batch_size):
            xb = X[i:i + batch_size]
            Kxb = relu_ntk_kernel(
                xb, self.Xtr,
                weight_variance=self.weight_variance,
                bias_variance=self.bias_variance,
                eps=self.eps,
            )
            outs.append(Kxb @ self.alpha)
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def fitted(self) -> torch.Tensor:
        """Return in-sample fitted values on Xtr."""
        return self.predict(self.Xtr)


def laplace_kernel(X: torch.Tensor, Z: torch.Tensor, kernel_width: float) -> torch.Tensor:
    return torch.exp(-torch.cdist(X, Z, p=2)/kernel_width)

# ----------------------------
# KRR with Laplace kernel
# ----------------------------
class LaplaceKernel:
    def __init__(self, ridge: float, kernel_width: Optional[float] = None, **kwargs):
        """
        ridge: ridge parameter λ >= 0
        kernel_width: Laplace kernel width. If None, uses median-distance heuristic from the training X.
        """
        self.ridge = float(ridge)
        self.kernel_width = kernel_width
        self.Xtr = None
        self.alpha = None  # (n_train, d_out)
        self._chol = None  # Cholesky factor of (K + λI)

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        X: (n, d), y: (n,) or (n, d_out). Types: float32/float64; device: cpu/cuda.
        """
        if y.dim() == 1:
            y = y.unsqueeze(1)
        assert X.shape[0] == y.shape[0], "X and y must have same n"

        self.Xtr = X
        
        K = laplace_kernel(X, X, self.kernel_width)
        n = K.shape[0]
        K = K + self.ridge * torch.eye(n, device=K.device, dtype=K.dtype)

        self.alpha, self._chol = _solve_kernel_system(K, y)
        return self

    @torch.no_grad()
    def predict(self, X: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Predict y_hat at X. Returns shape (n_eval, d_out). Uses batching for K(X, Xtr) if requested.
        """
        assert self.Xtr is not None and self.alpha is not None, "Call fit() first."
        if batch_size is None:
            Kxe = laplace_kernel(X, self.Xtr, self.kernel_width)
            return Kxe @ self.alpha
        # Batched to keep memory in check
        outs = []
        for i in range(0, X.shape[0], batch_size):
            xb = X[i:i+batch_size]
            Kxb = laplace_kernel(xb, self.Xtr, self.kernel_width)
            outs.append(Kxb @ self.alpha)
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def fitted(self) -> torch.Tensor:
        """Return in-sample fitted values on Xtr."""
        return self.predict(self.Xtr)


def gaussian_kernel(X: torch.Tensor, Z: torch.Tensor, kernel_width: float) -> torch.Tensor:
    """
    Compute Gaussian (RBF) kernel: exp(-||x - z||^2 / (2 * kernel_width^2))
    """
    dists_sq = torch.cdist(X, Z, p=2).pow(2)
    return torch.exp(-dists_sq / (2 * kernel_width ** 2))


# ----------------------------
# KRR with Gaussian kernel
# ----------------------------
class GaussianKernel:
    def __init__(self, ridge: float, kernel_width: Optional[float] = None, **kwargs):
        """
        ridge: ridge parameter λ >= 0
        kernel_width: Gaussian kernel width. If None, uses median-distance heuristic from the training X.
        """
        self.ridge = float(ridge)
        self.kernel_width = kernel_width
        self.Xtr = None
        self.alpha = None
        self._chol = None

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        X: (n, d), y: (n,) or (n, d_out). Types: float32/float64; device: cpu/cuda.
        """
        if y.dim() == 1:
            y = y.unsqueeze(1)
        assert X.shape[0] == y.shape[0], "X and y must have same n"

        self.Xtr = X
        
        K = gaussian_kernel(X, X, self.kernel_width)
        n = K.shape[0]
        K = K + self.ridge * torch.eye(n, device=K.device, dtype=K.dtype)

        self.alpha, self._chol = _solve_kernel_system(K, y)
        return self

    @torch.no_grad()
    def predict(self, X: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Predict y_hat at X. Returns shape (n_eval, d_out). Uses batching for K(X, Xtr) if requested.
        """
        assert self.Xtr is not None and self.alpha is not None, "Call fit() first."
        if batch_size is None:
            Kxe = gaussian_kernel(X, self.Xtr, self.kernel_width)
            return Kxe @ self.alpha
        # Batched to keep memory in check
        outs = []
        for i in range(0, X.shape[0], batch_size):
            xb = X[i:i+batch_size]
            Kxb = gaussian_kernel(xb, self.Xtr, self.kernel_width)
            outs.append(Kxb @ self.alpha)
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def fitted(self) -> torch.Tensor:
        """Return in-sample fitted values on Xtr."""
        return self.predict(self.Xtr)


# def polynomial_kernel(X: torch.Tensor, Z: torch.Tensor, coeffs: torch.Tensor, max_degree: int) -> torch.Tensor:
#     """
#     Compute polynomial kernel: K(x, z) = sum_{ell=0}^{max_degree} c_ell * (<x, z>)^ell
    
#     X: (n, d)
#     Z: (m, d)
#     coeffs: (max_degree + 1,) tensor of coefficients [c_0, c_1, ..., c_{max_degree}]
    
#     Returns: (n, m) kernel matrix
#     """
#     # Compute inner products: (n, m)
#     inner_products = X @ Z.T
    
#     # Compute kernel as sum of powers weighted by coefficients
#     K = torch.zeros_like(inner_products)
#     for ell in range(max_degree + 1):
#         K = K + coeffs[ell] * (inner_products ** ell)
    
#     return K


# # ----------------------------
# # KRR with Polynomial kernel
# # ----------------------------
# class PolynomialKernel:
#     def __init__(self, ridge: float, max_degree: int = 3, coeffs: Optional[torch.Tensor] = None, **kwargs):
#         """
#         ridge: ridge parameter λ >= 0
#         max_degree: maximum degree of the polynomial
#         coeffs: coefficients [c_0, c_1, ..., c_{max_degree}]. If None, defaults to 1/ell! for ell >= 0.
#         """
#         self.ridge = float(ridge)
#         self.max_degree = max_degree
        
#         # Default coefficients: 1/ell!
#         if coeffs is None:
#             import math
#             self.coeffs = torch.tensor([1.0 / math.factorial(ell) for ell in range(max_degree + 1)])
#         else:
#             assert len(coeffs) == max_degree + 1, f"coeffs must have length {max_degree + 1}"
#             self.coeffs = coeffs if isinstance(coeffs, torch.Tensor) else torch.tensor(coeffs)
        
#         self.Xtr = None
#         self.alpha = None  # (n_train, d_out)
#         self._chol = None  # Cholesky factor of (K + λI)

#     def fit(self, X: torch.Tensor, y: torch.Tensor):
#         """
#         X: (n, d), y: (n,) or (n, d_out). Types: float32/float64; device: cpu/cuda.
#         """
#         if y.dim() == 1:
#             y = y.unsqueeze(1)
#         assert X.shape[0] == y.shape[0], "X and y must have same n"

#         self.Xtr = X
        
#         # Move coefficients to same device and dtype as X
#         coeffs = self.coeffs.to(device=X.device, dtype=X.dtype)
        
#         K = polynomial_kernel(X, X, coeffs, self.max_degree)
#         n = K.shape[0]
#         K = K + self.ridge * torch.eye(n, device=K.device, dtype=K.dtype)

#         # Cholesky solve: (K) alpha = y
#         L = torch.linalg.cholesky(K)                 # K = L L^T
#         z = torch.cholesky_solve(y, L)               # returns K^{-1} y
#         self.alpha = z                               # alpha = (K)^{-1} y
#         self._chol = L
#         return self

#     @torch.no_grad()
#     def predict(self, X: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
#         """
#         Predict y_hat at X. Returns shape (n_eval, d_out). Uses batching for K(X, Xtr) if requested.
#         """
#         assert self.Xtr is not None and self.alpha is not None, "Call fit() first."
        
#         # Move coefficients to same device and dtype as X
#         coeffs = self.coeffs.to(device=X.device, dtype=X.dtype)
        
#         if batch_size is None:
#             Kxe = polynomial_kernel(X, self.Xtr, coeffs, self.max_degree)
#             return Kxe @ self.alpha
#         # Batched to keep memory in check
#         outs = []
#         for i in range(0, X.shape[0], batch_size):
#             xb = X[i:i+batch_size]
#             Kxb = polynomial_kernel(xb, self.Xtr, coeffs, self.max_degree)
#             outs.append(Kxb @ self.alpha)
#         return torch.cat(outs, dim=0)

#     @torch.no_grad()
#     def fitted(self) -> torch.Tensor:
#         """Return in-sample fitted values on Xtr."""
#         return self.predict(self.Xtr)
