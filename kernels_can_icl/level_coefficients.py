def get_polynomial_coefficient(level, coeffs):
    """Returns coeffs[level] if in range, else 0."""
    if level < len(coeffs):
        return coeffs[level]
    return 0.0

def get_polynomial_coefficients(coeffs):
    return lambda level: get_polynomial_coefficient(level, coeffs=coeffs)

