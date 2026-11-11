import numpy as np
import torch
from scipy.optimize import bisect
from kernels_can_icl.utils import ensure_numpy, ensure_torch


def compute_learnabilities(n, eigvals, ridge=0):
    
    def _compute_lrn(n, eigvals, ridge):
        eigvals = ensure_numpy(eigvals)
        conservation_law = lambda kap: (eigvals/(eigvals+kap)).sum() + ridge/kap - n
        kappa = bisect(conservation_law, 1e-25, 1e10, maxiter=128)
        learnabilities = eigvals / (eigvals + kappa)
        return learnabilities, kappa
    
    n = ensure_numpy(n)
    if np.ndim(n) == 0:
        return _compute_lrn(float(n), eigvals, ridge)
    if np.ndim(n) == 1:
        learnabilities = np.zeros((len(n), len(eigvals)))
        kappas = np.zeros(len(n))
        for i in range(len(n)):
            learnabilities[i], kappas[i] = _compute_lrn(n[i], eigvals, ridge)
        return learnabilities, kappas
    else:
        raise ValueError("n must be a scalar or 1D array/tensor")


def learning_curve(n, learnabilities, eigcoeffs, noise_var=0):
    n = ensure_numpy(n)
    if len(eigcoeffs) < learnabilities.shape[-1]:
        learnabilities = learnabilities[..., :len(eigcoeffs)]
    sum_axis = None if learnabilities.ndim == 1 else 1
    e0 = n / (n - (learnabilities**2).sum(axis=sum_axis))
    test_mse = e0 * (((1-learnabilities)**2 * eigcoeffs**2).sum(axis=sum_axis) + noise_var)
    return test_mse


def learning_curve_OOD(n, learnabilities, eigcoeffs_train, eigcoeffs_test, test_gram_OOD,
                        noise_var=0):
    """Test risk under train/test distribution shift, evaluated against a (possibly
    distinct) test-time target function.

    Generalizes `learning_curve` to the case where training data is drawn from p,
    test points are drawn from a different distribution q, and the target function
    eigencoefficients are allowed to differ between train (eigcoeffs_train, i.e. v^(p))
    and test (eigcoeffs_test, i.e. v^(q)), all expressed in the shared p-eigenbasis.
    Pure covariate shift is the special case eigcoeffs_test == eigcoeffs_train.

    n: scalar or 1D array/tensor, training set size(s)
    learnabilities: output of compute_learnabilities; shape (n_modes,) or (len(n), n_modes)
    eigcoeffs_train: v^(p), the train-time target's eigencoefficients, shape (n_modes,)
    eigcoeffs_test: v^(q), the test-time target's eigencoefficients, shape (n_modes,)
    test_gram_OOD: S_ij = <phi_i, phi_j>_q, the Gram matrix of the train-distribution
        eigenbasis under the test distribution q. Shape (n_modes, n_modes). S = I
        recovers the in-distribution case exactly.
    noise_var: eps^2, label noise variance (same role as in `learning_curve`)
    """
    n = ensure_numpy(n)
    eigcoeffs_train = ensure_numpy(eigcoeffs_train)
    eigcoeffs_test = ensure_numpy(eigcoeffs_test)
    test_gram_OOD = ensure_numpy(test_gram_OOD)

    n_modes = test_gram_OOD.shape[-1]
    if len(eigcoeffs_train) < learnabilities.shape[-1]:
        learnabilities = learnabilities[..., :len(eigcoeffs_train)]
    if learnabilities.shape[-1] < n_modes:
        n_modes = learnabilities.shape[-1]
        test_gram_OOD = test_gram_OOD[:n_modes, :n_modes]
    eigcoeffs_train = eigcoeffs_train[:n_modes]
    eigcoeffs_test = eigcoeffs_test[:n_modes]

    sum_axis = None if learnabilities.ndim == 1 else 1
    e0 = n / (n - (learnabilities**2).sum(axis=sum_axis))
    in_dist_risk_train = e0 * (((1-learnabilities)**2 * eigcoeffs_train**2).sum(axis=sum_axis) + noise_var)

    delta = eigcoeffs_test - learnabilities * eigcoeffs_train  # Delta_i, shape matches learnabilities
    diag_S = np.diagonal(test_gram_OOD)

    if learnabilities.ndim == 1:
        bias_OOD = delta @ test_gram_OOD @ delta
        var_OOD = (diag_S * learnabilities**2).sum() * in_dist_risk_train / n
    else:
        bias_OOD = np.einsum('bi,ij,bj->b', delta, test_gram_OOD, delta)
        var_OOD = (diag_S * learnabilities**2).sum(axis=1) * in_dist_risk_train / n

    test_mse_OOD = bias_OOD + var_OOD + noise_var
    return test_mse_OOD


def learning_curve_OOD_diag(n, learnabilities, eigcoeffs_train, eigcoeffs_test, noise_var=0):
    """Special case of `learning_curve_OOD` where test_gram_OOD = I, i.e. the
    p-eigenbasis remains orthonormal under the test distribution q (no cross-terms
    <phi_i, phi_j>_q for i != j). This still allows eigcoeffs_test != eigcoeffs_train
    (concept shift), it just assumes away any covariate-shift-induced mode mixing.

    Equivalent to calling learning_curve_OOD(..., test_gram_OOD=np.eye(n_modes)),
    but skips ever forming the (n_modes, n_modes) matrix, so bias reduces to a plain
    per-mode sum: sum_i Delta_i^2 with Delta_i = eigcoeffs_test_i - L_i * eigcoeffs_train_i.

    n: scalar or 1D array/tensor, training set size(s)
    learnabilities: output of compute_learnabilities; shape (n_modes,) or (len(n), n_modes)
    eigcoeffs_train: v^(p), the train-time target's eigencoefficients, shape (n_modes,)
    eigcoeffs_test: v^(q), the test-time target's eigencoefficients, shape (n_modes,)
    noise_var: eps^2, label noise variance
    """
    n = ensure_numpy(n)
    eigcoeffs_train = ensure_numpy(eigcoeffs_train)
    eigcoeffs_test = ensure_numpy(eigcoeffs_test)

    n_modes = min(len(eigcoeffs_train), len(eigcoeffs_test), learnabilities.shape[-1])
    if learnabilities.shape[-1] > n_modes:
        learnabilities = learnabilities[..., :n_modes]
    eigcoeffs_train = eigcoeffs_train[:n_modes]
    eigcoeffs_test = eigcoeffs_test[:n_modes]

    sum_axis = None if learnabilities.ndim == 1 else 1
    e0 = n / (n - (learnabilities**2).sum(axis=sum_axis))
    in_dist_risk_train = e0 * (((1-learnabilities)**2 * eigcoeffs_train**2).sum(axis=sum_axis) + noise_var)

    delta = eigcoeffs_test - learnabilities * eigcoeffs_train  # Delta_i, shape matches learnabilities
    bias_OOD = (delta**2).sum(axis=sum_axis)
    var_OOD = (learnabilities**2).sum(axis=sum_axis) * in_dist_risk_train / n

    test_mse_OOD = bias_OOD + var_OOD + noise_var
    return test_mse_OOD
