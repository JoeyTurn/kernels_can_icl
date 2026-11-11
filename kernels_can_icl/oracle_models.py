import torch


class Ridge:
    """
    Ridge regression "oracle" with ridge strength set to noise_var (matching your original).
    Usage:
        est = Ridge(noise_var=0.2)
        yq = est.predict(X, y, query)
    """
    def __init__(self, noise_var: float):
        self.ridge = float(noise_var)

    def predict(self, X: torch.Tensor, y: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """
        X: (L, d)
        y: (L, 1)
        query: (1, d)
        returns: (1, 1)
        """
        d = X.shape[1]
        eye = torch.eye(d, dtype=X.dtype, device=X.device)
        A = X.T @ X + self.ridge * eye          # (d, d)
        b = X.T @ y                             # (d, 1)
        w_hat = torch.linalg.lstsq(A, b).solution  # (d, 1)
        return query @ w_hat


