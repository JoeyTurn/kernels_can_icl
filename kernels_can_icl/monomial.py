from kernels_can_icl.utils import ensure_torch
import torch


class Monomial(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __lt__(self, other):
        assert isinstance(other, Monomial)
        return self.degree() < other.degree()

    def degree(self):
        if len(self) == 0:
            return 0
        return sum(self.values())

    def max_degree(self):
        if len(self) == 0:
            return 0
        return max(self.values())

    def copy(self):
        return Monomial(super().copy())

    def __str__(self) -> str:
        if self.degree() == 0:
            return "1"
        monostr = ""
        for idx, exp in self.items():
            expstr = f"^{exp}" if exp > 1 else ""
            monostr += f"x_{{{idx}}}{expstr}"
        return f"${monostr}$"

    def __repr__(self):
        return self.__str__()

    def evaluate(self, X, gamma=1.0):
        X = ensure_torch(X)
        
        N = X.shape[0]
        if self.degree() == 0:
            return torch.ones(N, device=X.device, dtype=X.dtype)

        y = torch.ones(N, device=X.device, dtype=X.dtype)
        for index, exponent in self.items():
            index = int(index)
            exponent = int(exponent)
            
            y.mul_((gamma * X[:, index]) ** exponent)

        return y

    def basis(self, canonical: bool = True) -> dict:
        if self.degree() == 0:
            return {}

        items = sorted(self.items()) if canonical else self.items()
        return {idx: int(exp) for idx, exp in items}


