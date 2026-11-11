import math
import torch

from kernels_can_icl.kernels import *


def median_distance(X , max_samples = 2048):
    if X.shape[0] > max_samples:
        idx = torch.randperm(X.shape[0], device=X.device)[:max_samples]
        X = X[idx]
    dists = torch.cdist(X, X, p=2)
    dists = dists[dists > 0]
    return torch.median(dists).item()


def kernel_matrix(X, kernel_class, kernel_params):
    name = kernel_class
    params = dict(kernel_params)

    if name == "GaussianKernel":
        kernel_width = params.get("kernel_width", None)
        if kernel_width is None:
            kernel_width = median_distance(X)
        return gaussian_kernel(X, X, kernel_width)

    if name == "LaplaceKernel":
        kernel_width = params.get("kernel_width", None)
        if kernel_width is None:
            kernel_width = median_distance(X)
        return laplace_kernel(X, X, kernel_width)

    if name == "ReluNTK":
        return relu_ntk_kernel(X, X, weight_variance=params.get("weight_variance", 1.0),
            bias_variance=params.get("bias_variance", 1.0), eps=params.get("eps", 1e-12),)

    if name == "PolynomialKernel":
        max_degree = params.get("max_degree", 3)
        coeffs = params.get("coeffs", None)
        if coeffs is None:
            coeffs = torch.tensor(
                [1.0 / math.factorial(ell) for ell in range(max_degree + 1)],
                device=X.device,
                dtype=X.dtype,
            )
        else:
            if not isinstance(coeffs, torch.Tensor):
                coeffs = torch.tensor(coeffs, device=X.device, dtype=X.dtype)
            else:
                coeffs = coeffs.to(device=X.device, dtype=X.dtype)
        return polynomial_kernel(X, X, coeffs, max_degree)

    raise ValueError(f"Unsupported kernel_class: {name}")


def eigendecompose(K):
    K = 0.5 * (K + K.T)
    evals, evecs = torch.linalg.eigh(K)
    order = torch.argsort(evals, descending=True)
    return evals[order], evecs[:, order]


def phi1_x_query(X_flat, context_len=None, dim=None):
    """
    First candidate eigenfunction: phi(X) = x_q (the query point).
    
    The input X_flat is the flattened format from _get_input:
    X_flat = [X.flatten(), y.flatten(), Xq.flatten()] where shapes are:
    - X: (batch_size, context_len-1, dim)
    - y: (batch_size, context_len-1)
    - Xq: (batch_size, 1, dim)
    
    Args:
        X_flat: Flattened input of shape (batch_size, (context_len-1)*dim + (context_len-1) + dim)
        context_len: Length of context (L), if None will try to infer from global config
        dim: Dimension of input space, if None will try to infer from global config
        
    Returns:
        Query points first coordinate, shape (batch_size,)
    """
    
    # Parse the flattened input
    # Format: [X.flatten(start_dim=1), y.flatten(start_dim=1), Xq.flatten(start_dim=1)]
    # X: (context_len-1) * dim elements
    # y: (context_len-1) elements  
    # Xq: 1 * dim = dim elements
    
    X_size = (context_len - 1) * dim
    y_size = (context_len - 1)
    Xq_start = X_size + y_size
    Xq_end = Xq_start + dim
    
    # Extract Xq (the query point)
    Xq_flat = X_flat[:, Xq_start:Xq_end]  # (batch_size, dim)
    
    # Return first dimension as scalar output
    return Xq_flat[:, 0]


def phi2_y_X_x_query(X_flat, context_len=None, dim=None):
    """
    Second candidate eigenfunction: phi(X) = y_ctx^T X_ctx x_q
    
    This computes the inner product of context labels with context data projected onto x_q.
    
    The input X_flat is the flattened format from _get_input:
    X_flat = [X.flatten(), y.flatten(), Xq.flatten()] where shapes are:
    - X: (batch_size, context_len-1, dim) - context inputs
    - y: (batch_size, context_len-1) - context labels
    - Xq: (batch_size, 1, dim) - query point
    
    Args:
        X_flat: Flattened input of shape (batch_size, (context_len-1)*dim + (context_len-1) + dim)
        context_len: Length of context (L), if None will try to infer from global config
        dim: Dimension of input space, if None will try to infer from global config
        
    Returns:
        Values of shape (batch_size,)
    """
    
    batch_size = X_flat.shape[0]
    
    # Parse the flattened input
    X_size = (context_len - 1) * dim
    y_size = (context_len - 1)
    Xq_start = X_size + y_size
    Xq_end = Xq_start + dim
    
    # Extract components
    X_ctx_flat = X_flat[:, :X_size]  # (batch_size, (L-1)*dim)
    y_ctx = X_flat[:, X_size:X_size + y_size]  # (batch_size, L-1)
    x_q = X_flat[:, Xq_start:Xq_end]  # (batch_size, dim)
    
    # Reshape X_ctx to (batch_size, L-1, dim)
    X_ctx = X_ctx_flat.reshape(batch_size, context_len - 1, dim)
    
    # Compute X_ctx @ x_q for each context point
    # X_ctx: (batch_size, L-1, dim), x_q: (batch_size, dim)
    X_dot_xq = torch.einsum('bli,bi->bl', X_ctx, x_q)  # (batch_size, L-1)
    
    # Compute y_ctx^T (X_ctx @ x_q)
    result = torch.einsum('bl,bl->b', y_ctx, X_dot_xq)  # (batch_size,)
    
    return result


