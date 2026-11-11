import numpy as np
import math
import functools
from itertools import product, combinations_with_replacement, permutations
from scipy.optimize import linear_sum_assignment
import heapq

from kernels_can_icl.utils import ensure_numpy, ensure_torch
from kernels_can_icl.monomial import Monomial


# Automation
def get_partitions(n):
    """Generates integer partitions defining the Sequence Families."""
    answer = set()
    answer.add((n, ))
    for x in range(1, n):
        for y in get_partitions(n - x):
            answer.add(tuple(sorted((x, ) + y, reverse=True)))
    return sorted(list(answer), key=lambda x: (-len(x), x))

def get_token_states(degree):
    """Generates all (X, y) combinations for a given token degree."""
    return [(degree - i, i) for i in range(degree + 1)]

def combine_states(state1, state2):
    """Combines two sequence states to pass to the moment generator."""
    all_tokens = set(state1.keys()) | set(state2.keys())
    combined = []
    for t in all_tokens:
        x1, y1 = state1.get(t, (0,0))
        x2, y2 = state2.get(t, (0,0))
        if (x1+x2) > 0 or (y1+y2) > 0:
            combined.append((x1+x2, y1+y2))
    return combined

def generate_full_sequence_basis(ell, L_seq):
    """Generates the absolute complete basis of degree ell across the sequence."""
    if ell == 0: return [{}]
    basis = []
    # Pick which tokens absorb the degree ell
    for tokens in combinations_with_replacement(range(L_seq), ell):
        deg_map = {}
        for t in tokens: deg_map[t] = deg_map.get(t, 0) + 1
           
        t_indices = list(deg_map.keys())
        t_options = [get_token_states(deg_map[t]) for t in t_indices]
       
        for state_combo in product(*t_options):
            basis.append({t_indices[i]: state_combo[i] for i in range(len(t_indices))})
    return basis

def get_family_basis(part, L_seq):
    """Generates the specific sequence basis for a single Family (Partition)."""
    unique_assignments = set()
    for tokens in permutations(range(L_seq), len(part)):
        assignment = {}
        for t, deg in zip(tokens, part):
            assignment.setdefault(deg, []).append(t)
        canonical = tuple(sorted((deg, tuple(sorted(ts))) for deg, ts in assignment.items()))
        unique_assignments.add(canonical)
       
    basis = []
    for canonical in unique_assignments:
        t_indices, t_degrees = [], []
        for deg, ts in canonical:
            for t in ts:
                t_indices.append(t)
                t_degrees.append(deg)
               
        t_options = [get_token_states(deg) for deg in t_degrees]
        for state_combo in product(*t_options):
            basis.append({t_indices[i]: state_combo[i] for i in range(len(t_indices))})
    return basis

def get_weight(state_dict):
    """Calculates the multinomial denominator 1 / sqrt(k!)."""
    denom = 1.0
    for x, y in state_dict.values():
        denom *= math.factorial(x) * math.factorial(y)
    return 1.0 / np.sqrt(denom)


# -----------------------------------------------------------------------------
# Multidim state helpers: a token state is a (dim+1)-tuple (x_0,...,x_{dim-1}, b),
# where x_d is the power of X_{token,d} and b is the power of the scalar y_token.
# These parallel the 1D (x, y) helpers but represent monomials spanning multiple
# X-dimensions (cross-X-dimension), needed for the multidim higher-degree engine.
# -----------------------------------------------------------------------------

def get_token_states_md(degree, dim):
    """All (dim+1)-tuples of non-negative ints summing to `degree`."""
    def comps(n, parts):
        if parts == 1:
            yield (n,)
            return
        for i in range(n + 1):
            for rest in comps(n - i, parts - 1):
                yield (i,) + rest
    return list(comps(degree, dim + 1))


def generate_full_sequence_basis_md(ell, L_seq, dim):
    """Complete degree-`ell` basis across the sequence, with multidim token states."""
    if ell == 0:
        return [{}]
    basis = []
    for tokens in combinations_with_replacement(range(L_seq), ell):
        deg_map = {}
        for t in tokens:
            deg_map[t] = deg_map.get(t, 0) + 1
        t_indices = list(deg_map.keys())
        t_options = [get_token_states_md(deg_map[t], dim) for t in t_indices]
        for combo in product(*t_options):
            basis.append({t_indices[i]: combo[i] for i in range(len(t_indices))})
    return basis


def get_family_basis_md(part, L_seq, dim):
    """Specific sequence basis for one family (partition), with multidim token states."""
    unique_assignments = set()
    for tokens in permutations(range(L_seq), len(part)):
        assignment = {}
        for t, deg in zip(tokens, part):
            assignment.setdefault(deg, []).append(t)
        canonical = tuple(sorted((deg, tuple(sorted(ts))) for deg, ts in assignment.items()))
        unique_assignments.add(canonical)

    basis = []
    for canonical in unique_assignments:
        t_indices, t_degrees = [], []
        for deg, ts in canonical:
            for t in ts:
                t_indices.append(t)
                t_degrees.append(deg)
        t_options = [get_token_states_md(deg, dim) for deg in t_degrees]
        for combo in product(*t_options):
            basis.append({t_indices[i]: combo[i] for i in range(len(t_indices))})
    return basis


def combine_states_md(state1, state2, dim):
    """Combine two multidim sequence states into a list of per-token feature tuples."""
    zero = (0,) * (dim + 1)
    combined = []
    for t in set(state1) | set(state2):
        a = state1.get(t, zero)
        b = state2.get(t, zero)
        summ = tuple(a[i] + b[i] for i in range(dim + 1))
        if any(summ):
            combined.append(summ)
    return combined


def get_weight_md(state_dict):
    """Multinomial Hermite normalization 1 / sqrt(prod_d x_d! * b!)."""
    denom = 1.0
    for st in state_dict.values():
        for e in st:
            denom *= math.factorial(e)
    return 1.0 / np.sqrt(denom)


def _orbit_key_md(state):
    """Permutation-orbit identifier: the multiset of per-token (dim+1)-exponent states."""
    return tuple(sorted(state.values()))


def _is_canonical_state_md(state):
    """True if `state` is the canonical (symmetric-mode) representative of its orbit: active
    tokens are exactly {0,...,k-1} and the per-token states are in descending order (by total
    degree, then exponents lexicographically) on ascending tokens."""
    toks = sorted(state.keys())
    k = len(toks)
    if toks != list(range(k)):
        return False
    states_in_token_order = [state[t] for t in toks]
    canonical_order = sorted(state.values(), key=lambda st: (-sum(st),) + tuple(-e for e in st))
    return states_in_token_order == canonical_order


def _state_key_md(state):
    """Hashable canonical key for a multidim sequence-state dict {token: state_tuple}."""
    return tuple(sorted(state.items()))

# def build_theoretical_shelves(max_degree, L_seq, betas_flat, v, eps):
#     """The Master Engine. Run this ONCE to generate the spectrum dict."""
#     E_func = get_moment_generator(betas_flat, v, eps) # Use your existing moment generator
#     shelves = {}
   
#     # 1. Precompute all complete levels for Gram-Schmidt projection
#     level_bases = {ell: generate_full_sequence_basis(ell, L_seq) for ell in range(max_degree + 1)}
       
#     for ell in range(1, max_degree + 1):
#         # Gather all lower parity states
#         lower_basis = []
#         for d_low in range(ell - 2, -1, -2): lower_basis.extend(level_bases[d_low])
           
#         # Build inverse lower matrix once per level
#         size_L = len(lower_basis)
#         C_LL = np.zeros((size_L, size_L))
#         if size_L > 0:
#             for r in range(size_L):
#                 for c in range(size_L):
#                     C_LL[r, c] = E_func(*combine_states(lower_basis[r], lower_basis[c]))
#             C_LL_pinv = np.linalg.pinv(C_LL)
           
#         # Build orthogonalized shelves per Family
#         for part in get_partitions(ell):
#             basis_F = get_family_basis(part, L_seq)
#             size_F = len(basis_F)
#             if size_F == 0: continue
           
#             C_FF = np.zeros((size_F, size_F))
#             for r in range(size_F):
#                 for c in range(size_F):
#                     C_FF[r, c] = E_func(*combine_states(basis_F[r], basis_F[c]))
                   
#             if size_L > 0:
#                 C_FL = np.zeros((size_F, size_L))
#                 for r in range(size_F):
#                     for c in range(size_L):
#                         C_FL[r, c] = E_func(*combine_states(basis_F[r], lower_basis[c]))
#                 C_perp = C_FF - C_FL @ C_LL_pinv @ C_FL.T
#             else:
#                 C_perp = C_FF
               
#             W = np.array([get_weight(s) for s in basis_F])
#             C_weighted = np.outer(W, W) * C_perp
           
#             roots = sorted(np.abs(np.linalg.eigvalsh(C_weighted)), reverse=True)
#             shelves[part] = {'basis': basis_F, 'roots': roots}

#     return shelves

@functools.lru_cache(maxsize=None)
def _mX_md(m, var):
    """E[Z^m] for Z ~ N(0, var) = (m-1)!! var^(m/2) for even m, else 0."""
    if m % 2 != 0:
        return 0.0
    res = 1
    for i in range(m - 1, 0, -2):
        res *= i
    return res * (var ** (m // 2))


def get_moment_generator_md(betas, v, eps, dim):
    """Exact multidim context moment generator
    E[ prod_token (prod_d X_{t,d}^{x_d}) * y_t^b ] under the shared-beta task model
    y = sum_d beta_d X_d + noise. Each token feature is a (dim+1)-tuple (x_0,...,x_{dim-1}, b).

    Per token, y^b = (sum_d beta_d X_d + xi)^b is expanded multinomially; integrating X (each
    N(0,v)) and xi (N(0,eps)) gives a polynomial in beta, and the shared beta is marginalized with
    the empirical joint moments M[kvec] = E[prod_d beta_d^{kvec_d}]. E depends only on the multiset
    of token features (tokens exchangeable, beta shared), so everything is memoized.
    """
    betas = ensure_numpy(betas)
    Nb = betas.shape[0]

    Mcache = {}
    def M(kvec):
        val = Mcache.get(kvec)
        if val is None:
            arr = np.ones(Nb)
            for d, p in enumerate(kvec):
                if p:
                    arr = arr * betas[:, d] ** p
            val = float(np.mean(arr))
            Mcache[kvec] = val
        return val

    polycache = {}
    def token_poly(xvec, b):
        key = (xvec, b)
        p = polycache.get(key)
        if p is not None:
            return p
        poly = {}
        def gen(parts, remaining):
            if parts == 0:
                yield ()
                return
            for i in range(remaining + 1):
                for rest in gen(parts - 1, remaining - i):
                    yield (i,) + rest
        for kvec in gen(dim, b):
            knoise = b - sum(kvec)
            coeff = math.factorial(b)
            for k in kvec:
                coeff //= math.factorial(k)
            coeff //= math.factorial(knoise)
            c = coeff * _mX_md(knoise, eps)
            if c == 0.0:
                continue
            for d in range(dim):
                c *= _mX_md(xvec[d] + kvec[d], v)
                if c == 0.0:
                    break
            if c != 0.0:
                poly[kvec] = poly.get(kvec, 0.0) + c
        polycache[key] = poly
        return poly

    Ecache = {}
    def E(*token_features):
        feats = tuple(sorted(token_features))
        val = Ecache.get(feats)
        if val is not None:
            return val
        P = {(0,) * dim: 1.0}
        for st in feats:
            if not any(st):
                continue
            xvec, b = st[:dim], st[dim]
            tp = token_poly(xvec, b)
            newP = {}
            for k1, c1 in P.items():
                for k2, c2 in tp.items():
                    k = tuple(k1[i] + k2[i] for i in range(dim))
                    newP[k] = newP.get(k, 0.0) + c1 * c2
            P = newP
        val = sum(c * M(k) for k, c in P.items() if c != 0.0)
        Ecache[feats] = val
        return val

    return E


_MD_GEN_CACHE = {}
_MD_LOWER_CACHE = {}
_MD_ENGINE_CACHE = {}


def _betas_key(betas):
    return hash(ensure_numpy(betas).tobytes())


def _get_md_generator(betas, v, eps, dim):
    """Cached moment generator per (betas content, v, eps, dim)."""
    key = (_betas_key(betas), float(v), float(eps), dim)
    gen = _MD_GEN_CACHE.get(key)
    if gen is None:
        gen = get_moment_generator_md(betas, v, eps, dim)
        _MD_GEN_CACHE[key] = gen
    return gen


def _get_lower_inverse_md(betas, deg, L_seq, v, eps, dim):
    """Lower-degree (same-parity) sequence basis and pinv(C_LL); cached per degree so all families
    of a degree reuse the expensive C_LL."""
    key = (_betas_key(betas), deg, L_seq, float(v), float(eps), dim)
    cached = _MD_LOWER_CACHE.get(key)
    if cached is not None:
        return cached
    E = _get_md_generator(betas, v, eps, dim)
    lower_basis = []
    for d_low in range(deg - 2, -1, -2):
        lower_basis.extend(generate_full_sequence_basis_md(d_low, L_seq, dim))
    size_L = len(lower_basis)
    if size_L == 0:
        result = ([], None)
    else:
        C_LL = np.zeros((size_L, size_L))
        for r in range(size_L):
            for c in range(size_L):
                C_LL[r, c] = E(*combine_states_md(lower_basis[r], lower_basis[c], dim))
        result = (lower_basis, np.linalg.pinv(C_LL))
    _MD_LOWER_CACHE[key] = result
    return result


def get_cached_general_roots(betas, deg, family_part, L_seq, v, eps, dim):
    """Multidim higher-degree family engine: builds the weighted, lower-orthogonalized covariance for
    a family, diagonalizes it, and returns (eigvals, eigvecs, basis_F, trace, assignment) where
    `assignment` bijectively maps each monomial (sequence state) to one eigenvalue via the
    symmetry-adapted convention. Cached on the betas content + family parameters."""
    key = (_betas_key(betas), deg, family_part, L_seq, float(v), float(eps), dim)
    cached = _MD_ENGINE_CACHE.get(key)
    if cached is not None:
        return cached

    E = _get_md_generator(betas, v, eps, dim)
    basis_F = get_family_basis_md(family_part, L_seq, dim)
    size_F = len(basis_F)

    C_FF = np.zeros((size_F, size_F))
    for r in range(size_F):
        for c in range(size_F):
            C_FF[r, c] = E(*combine_states_md(basis_F[r], basis_F[c], dim))

    lower_basis, C_LL_pinv = _get_lower_inverse_md(betas, deg, L_seq, v, eps, dim)
    if lower_basis:
        size_L = len(lower_basis)
        C_FL = np.zeros((size_F, size_L))
        for r in range(size_F):
            for c in range(size_L):
                C_FL[r, c] = E(*combine_states_md(basis_F[r], lower_basis[c], dim))
        C_perp = C_FF - C_FL @ C_LL_pinv @ C_FL.T
    else:
        C_perp = C_FF

    W = np.array([get_weight_md(s) for s in basis_F])
    C_weighted = np.outer(W, W) * C_perp
    eigvals, eigvecs = np.linalg.eigh(C_weighted)
    trace = float(np.trace(C_weighted))
    assignment = _build_higher_degree_assignment_md(eigvals, eigvecs, basis_F)
    result = (eigvals, eigvecs, basis_F, trace, assignment)
    _MD_ENGINE_CACHE[key] = result
    return result


def _build_higher_degree_assignment_md(eigvals, eigvecs, basis_F):
    """Bijectively map every multidim monomial to one family eigenvalue, reproducing the
    symmetric/antisymmetric convention (direct port of the 1D _build_higher_degree_assignment):
    diagonalize the reduced symmetric matrix over orbit-sum probes, match each orbit to a symmetric
    eigenvalue (Hungarian) and give it to the orbit's canonical representative, then match the
    remaining shifted states to the leftover eigenvalues. Returns {state_key: eigenvalue}."""
    sF = len(basis_F)
    Cw = (eigvecs * eigvals) @ eigvecs.T
    orbit_of = [_orbit_key_md(b) for b in basis_F]
    orbit_keys = list(dict.fromkeys(orbit_of))

    probes = np.zeros((sF, len(orbit_keys)))
    for col, k in enumerate(orbit_keys):
        sel = np.array([1.0 if ok == k else 0.0 for ok in orbit_of])
        probes[:, col] = sel / np.linalg.norm(sel)
    S = probes.T @ Cw @ probes
    sev, svec = np.linalg.eigh(S)

    ri, ci = linear_sum_assignment(-(svec ** 2))
    matched = sorted(zip(ri, ci), key=lambda tc: -sev[tc[1]])

    assignment = {}
    used = set()
    canonical_eigval = {}
    for t, j_sym in matched:
        full_vec = probes @ svec[:, j_sym]
        ov = np.abs(eigvecs.T @ full_vec)
        for j in np.argsort(-ov):
            if j not in used:
                used.add(int(j))
                canonical_eigval[orbit_keys[t]] = abs(eigvals[int(j)])
                break

    for i, b in enumerate(basis_F):
        if _is_canonical_state_md(b):
            assignment[_state_key_md(b)] = canonical_eigval[orbit_of[i]]

    rem_rows = [i for i, b in enumerate(basis_F) if _state_key_md(b) not in assignment]
    rem_cols = [j for j in range(sF) if j not in used]
    if rem_rows:
        sub = np.abs(eigvecs[np.ix_(rem_rows, rem_cols)]) ** 2
        rr, cc = linear_sum_assignment(-sub)
        for a, c in zip(rr, cc):
            assignment[_state_key_md(basis_F[rem_rows[a]])] = abs(eigvals[rem_cols[c]])
    return assignment


def assign_higher_degree_eigval_md(context_state, assignment):
    """Look up the eigenvalue assigned to a multidim degree>=3 monomial's sequence state."""
    return assignment.get(_state_key_md(context_state), 0.0)


# -----------------------------------------------------------------------------
# Indexing and Mapping Helpers
# -----------------------------------------------------------------------------

def parse_idx(idx, C, dim):
    x_end = (C - 1) * dim
    y_end = x_end + (C - 1)
    if idx < x_end:
        return 'X', idx // dim, idx % dim
    elif idx < y_end:
        token = idx - x_end
        return 'y', token, 0
    else:
        feat = idx - y_end
        return 'q', None, feat

def get_same_token_idx(type1, d1, type2, d2, dim):
    """Returns the matrix row/col index for a same-token monomial."""
    d_mix = dim * (dim - 1) // 2
    N = 2 * dim + d_mix + 1
    
    if type1 == 'X' and type2 == 'X':
        if d1 == d2: 
            return d1 # Pure Square block
        a, b = min(d1, d2), max(d1, d2)
        mix_idx = 0
        for i in range(dim):
            for j in range(i + 1, dim):
                if i == a and j == b: return dim + mix_idx
                mix_idx += 1
                
    elif type1 == 'X' and type2 == 'y':
        return dim + d_mix + d1
    elif type1 == 'y' and type2 == 'X':
        return dim + d_mix + d2
    elif type1 == 'y' and type2 == 'y':
        return N - 1
    return 0

def get_cross_token_idx(type1, d1, type2, d2, dim):
    """Returns the matrix row/col index for a cross-token monomial."""
    M = dim**2 + 2 * dim + 1
    
    if type1 == 'X' and type2 == 'X':
        if d1 == d2: 
            return d1 # Aligned cross-token
        perp_idx = 0
        for i in range(dim):
            for j in range(dim):
                if i == j: continue
                if i == d1 and j == d2: return dim + perp_idx
                perp_idx += 1
                
    elif type1 == 'X' and type2 == 'y':
        return dim + dim * (dim - 1) + d1
    elif type1 == 'y' and type2 == 'X':
        return dim + dim * (dim - 1) + dim + d2
    elif type1 == 'y' and type2 == 'y':
        return M - 1
    return 0

# -----------------------------------------------------------------------------
# Matrix Generators
# -----------------------------------------------------------------------------

def build_same_token_matrices(betas, v, eps):
    """
    Builds the Expected Covariance matrix C_same and sequence coupling C_seq.
    Uses the Law of Total Covariance: Cov = E[Cov|beta] + Cov(E|beta)
    """
    d = betas.shape[1]
    d_mix = d * (d - 1) // 2
    N = 2 * d + d_mix + 1

    mu1 = np.mean(betas, axis=0)
    mu2 = np.mean(betas**2, axis=0)
    sum_beta2 = np.sum(betas**2, axis=1)
    E_S = np.mean(sum_beta2)
    E_S2 = np.mean(sum_beta2**2)
    
    E_C = np.zeros((N, N))
    
    idx_X2 = 0
    idx_XX = d
    idx_Xy = d + d_mix
    idx_y2 = N - 1

    # 1. X^2 block
    for a in range(d):
        E_C[idx_X2 + a, idx_X2 + a] = 2 * v**2
        
        E_C[idx_X2 + a, idx_Xy + a] = 2 * v**2 * mu1[a]
        E_C[idx_Xy + a, idx_X2 + a] = 2 * v**2 * mu1[a]
        
        E_C[idx_X2 + a, idx_y2] = 2 * v**2 * mu2[a]
        E_C[idx_y2, idx_X2 + a] = 2 * v**2 * mu2[a]

    # 2. XX block
    mix_idx = 0
    for a in range(d):
        for b in range(a + 1, d):
            E_C[idx_XX + mix_idx, idx_XX + mix_idx] = v**2
            
            E_C[idx_XX + mix_idx, idx_Xy + a] = v**2 * mu1[b]
            E_C[idx_Xy + a, idx_XX + mix_idx] = v**2 * mu1[b]
            
            E_C[idx_XX + mix_idx, idx_Xy + b] = v**2 * mu1[a]
            E_C[idx_Xy + b, idx_XX + mix_idx] = v**2 * mu1[a]
            
            E_bab = np.mean(betas[:, a] * betas[:, b])
            E_C[idx_XX + mix_idx, idx_y2] = 2 * v**2 * E_bab
            E_C[idx_y2, idx_XX + mix_idx] = 2 * v**2 * E_bab
            
            mix_idx += 1

    # 3. Xy and y^2 block expected covariances
    for a in range(d):
        for b in range(d):
            if a == b:
                E_C[idx_Xy + a, idx_Xy + a] = v**2 * (mu2[a] + E_S) + v * eps
            else:
                E_C[idx_Xy + a, idx_Xy + b] = v**2 * np.mean(betas[:, a] * betas[:, b])
        
        val = 2 * v**2 * np.mean(betas[:, a] * sum_beta2) + 2 * v * eps * mu1[a]
        E_C[idx_Xy + a, idx_y2] = val
        E_C[idx_y2, idx_Xy + a] = val

    E_C[idx_y2, idx_y2] = 2 * v**2 * E_S2 + 4 * v * eps * E_S + 2 * eps**2

    # 4. Sequence Coupling (Covariance of Conditional Expectations)
    C_seq = np.zeros((N, N))
    V = np.zeros((betas.shape[0], d + 1))
    for a in range(d):
        V[:, a] = v * betas[:, a]
    V[:, d] = v * sum_beta2 + eps
    
    C_seq[idx_Xy:, idx_Xy:] = np.cov(V, rowvar=False, bias=True)
    
    C_same = E_C + C_seq
    
    return C_same, C_seq

def build_cross_token_matrix(betas, v, eps):
    """
    Builds the Cross-Token Expected Covariance Matrix.
    Because E[Z_cross | beta] = 0, C_cross = E[Cov(Z_cross | beta)] exactly.
    """
    d = betas.shape[1]
    M = d**2 + 2 * d + 1
    
    mu1 = np.mean(betas, axis=0)
    mu2 = np.mean(betas**2, axis=0)
    sum_beta2 = np.sum(betas**2, axis=1)
    E_S = np.mean(sum_beta2)
    E_S2 = np.mean(sum_beta2**2)
    
    C_cross = np.zeros((M, M))
    
    idx_para = 0
    idx_perp = d
    idx_Xy = d + d * (d - 1)
    idx_yX = idx_Xy + d
    idx_yy = M - 1
    
    for a in range(d):
        C_cross[idx_para + a, idx_para + a] = v**2
        
        C_cross[idx_para + a, idx_Xy + a] = C_cross[idx_Xy + a, idx_para + a] = v**2 * mu1[a]
        C_cross[idx_para + a, idx_yX + a] = C_cross[idx_yX + a, idx_para + a] = v**2 * mu1[a]
        C_cross[idx_para + a, idx_yy]     = C_cross[idx_yy, idx_para + a]     = v**2 * mu2[a]
        
        val_Xy_Xy = v**2 * E_S + v * eps
        C_cross[idx_Xy + a, idx_Xy + a] = val_Xy_Xy
        C_cross[idx_yX + a, idx_yX + a] = val_Xy_Xy
        
        val_Xy_yy = v**2 * np.mean(betas[:, a] * sum_beta2) + v * eps * mu1[a]
        C_cross[idx_Xy + a, idx_yy] = C_cross[idx_yy, idx_Xy + a] = val_Xy_yy
        C_cross[idx_yX + a, idx_yy] = C_cross[idx_yy, idx_yX + a] = val_Xy_yy
        
        for b in range(d):
            val = v**2 * (mu2[a] if a == b else np.mean(betas[:, a] * betas[:, b]))
            C_cross[idx_Xy + a, idx_yX + b] = C_cross[idx_yX + b, idx_Xy + a] = val

    perp_idx = 0
    for a in range(d):
        for b in range(d):
            if a == b: continue
            
            C_cross[idx_perp + perp_idx, idx_perp + perp_idx] = v**2
            C_cross[idx_perp + perp_idx, idx_Xy + a] = C_cross[idx_Xy + a, idx_perp + perp_idx] = v**2 * mu1[b]
            C_cross[idx_perp + perp_idx, idx_yX + b] = C_cross[idx_yX + b, idx_perp + perp_idx] = v**2 * mu1[a]
            
            E_bab = np.mean(betas[:, a] * betas[:, b])
            C_cross[idx_perp + perp_idx, idx_yy] = C_cross[idx_yy, idx_perp + perp_idx] = v**2 * E_bab
            
            perp_idx += 1
            
    C_cross[idx_yy, idx_yy] = v**2 * E_S2 + 2 * v * eps * E_S + eps**2
    return C_cross

# -----------------------------------------------------------------------------
# Main Evaluation Function
# -----------------------------------------------------------------------------

def _build_sigma_A(betas, v, eps, dim):
    """Degree-1 linear covariance over [X_0, ..., X_{dim-1}, y] for one token."""
    Sigma_A = np.zeros((dim + 1, dim + 1))
    mu1 = np.mean(betas, axis=0)
    Sigma_A[:dim, :dim] = np.eye(dim) * v
    Sigma_A[:dim, -1] = v * mu1
    Sigma_A[-1, :dim] = v * mu1
    Sigma_A[-1, -1] = v * np.mean(np.sum(betas ** 2, axis=1)) + eps
    return Sigma_A


def _md_feature_eigval(M_eval, row_idx):
    """Bijective (Hungarian, collision-free) eigenvalue assignment over a deg-1/2 covariance block;
    returns the eigenvalue of the eigenmode that feature `row_idx` most overlaps."""
    eigvals, eigvecs = np.linalg.eigh(M_eval)
    ri, ci = linear_sum_assignment(-(eigvecs ** 2))
    col = dict(zip(ri, ci))[row_idx]
    return abs(eigvals[col])


def _md_impt_flag(context_state, d, total_deg, dim):
    """Importance tag for an odd-degree query term whose query is X_{q,d}.

    Degree 3 now tags the FULL anisotropic deg-3 block, not just the diagonal: alongside
    `y2q{d}`, `Xyq{d}`, `X2q{d}` (context X-power in the query dimension d), it also tags the
    off-diagonal gradient feature (sum_i y_i X_i[a]) X_q[d] with a != d as `X{a}yq{d}`,
    matching the coupled deg-3 block in build_eigcoeffs_expanded (a == d reproduces `Xyq{d}`).
    Degrees 5/7 are unchanged -- their off-diagonal/cross-axis blocks aren't built yet, so they
    still tag only the diagonal (per-dimension) modes."""
    def st(xd, b):
        s = [0] * (dim + 1)
        s[d] = xd
        s[dim] = b
        return tuple(s)
    cs = context_state
    if total_deg == 3:
        # context is a single canonical token: cs == {0: (x_0,...,x_{dim-1}, y_power)}
        if len(cs) == 1 and 0 in cs:
            s = cs[0]
            nx = sum(s[:dim]); ny = s[dim]
            if nx == 0 and ny == 2:
                return f"y2q{d}"
            if nx == 1 and ny == 1:                       # (sum_i y_i X_i[a]) X_q[d]
                a = list(s[:dim]).index(1)
                return f"Xyq{d}" if a == d else f"X{a}yq{d}"
            if nx == 2 and ny == 0 and s[d] == 2:         # (sum_i X_i[d]^2) X_q[d]
                return f"X2q{d}"
        return None
    elif total_deg == 5:
        # Full anisotropic deg-5: read beta-markers off the monomial (eigenvalue is rotation-
        # invariant, set by the context shape via assign_higher_degree_eigval_md; only the label
        # carries the marker). q-dim = d. Labels match build_eigcoeffs_expanded (eigcoeffs_deg5).
        toks = list(cs.values())
        def info(s):
            nx = sum(s[:dim]); ny = s[dim]
            xd = [i for i in range(dim) for _ in range(s[i])]   # X-dims w/ multiplicity, ascending
            return nx, ny, xd
        if len(toks) == 1:
            nx, ny, xd = info(toks[0])
            if nx == 0 and ny == 4:                              # A   (sum y^4) X_q[d]
                return f"y4q{d}"
            if nx == 1 and ny == 3:                              # B   (sum y^3 X[a]) X_q[d]
                a = xd[0]
                return f"y3Xq{d}" if a == d else f"y3X{a}q{d}"
            if nx == 2 and ny == 2:                              # C_same (sum y^2 X[a]X[c]) X_q[d]
                a, c = xd[0], xd[1]
                return f"y2X{a}X{c}q{d}"
            return None
        if len(toks) == 2:
            i0, i1 = info(toks[0]), info(toks[1])
            if i0[:2] == (1, 1) and i1[:2] == (1, 1):           # C_cross  yX[a]_i yX[c]_j X_q[d]
                a, c = sorted([i0[2][0], i1[2][0]])
                return f"yX{a}_yX{c}q{d}_cross"
            xx = i0 if i0[:2] == (2, 0) else (i1 if i1[:2] == (2, 0) else None)
            yx = i0 if i0[:2] == (1, 1) else (i1 if i1[:2] == (1, 1) else None)
            if xx is not None and yx is not None:               # D_xtr signal term: X[a]^2_i (yX[a])_j
                a = yx[2][0]
                if xx[2] == [a, a]:                             # square c==a carries the overlap
                    return f"yXdotX{a}q{d}_cross"
                return None                                     # cross c!=a -> no signal -> '0'
            return None
    elif total_deg == 7:
        # Full anisotropic deg-7: clean families read markers directly (eigenvalue is rotation-
        # invariant via assign_higher_degree_eigval_md; only the label carries the marker). Labels
        # match build_eigcoeffs_expanded (eigcoeffs_deg5, max_deg=7). q-dim = d. Trace families
        # (E7 y2XXr2, F7 yXr4, F7_cross X5yq) re-probe Sigma_ab -- already carried by deg-3 X{a}yq,
        # deg-5 D-family and deg-7 B7/D7 -- so they are left untagged ("0") here.
        toks = list(cs.values())
        def info(s):
            nx = sum(s[:dim]); ny = s[dim]
            xd = [i for i in range(dim) for _ in range(s[i])]   # X-dims w/ multiplicity, ascending
            return nx, ny, xd
        if len(toks) == 1:
            nx, ny, xd = info(toks[0])
            if nx == 0 and ny == 6:                              # A7  y6q{d}
                return f"y6q{d}"
            if nx == 1 and ny == 5:                              # B7  y5X[a]q{d}  E[||b||^4 b_a b_b]
                a = xd[0]
                return f"y5Xq{d}" if a == d else f"y5X{a}q{d}"
            if nx == 2 and ny == 4:                              # C7  y4X[a]X[c]q{d}
                return f"y4X{xd[0]}X{xd[1]}q{d}"
            if nx == 3 and ny == 3:                              # D7  y3X[a]X[c]X[e]q{d}  full 4-tensor
                return f"y3X{xd[0]}X{xd[1]}X{xd[2]}q{d}"
            return None
        if len(toks) == 2:
            i0, i1 = info(toks[0]), info(toks[1])
            # C7_cross  (y^3 X[a])_i (y X[c])_j X_q[d]  -- y-powers fix roles (a cubed, c linear)
            y3 = i0 if i0[:2] == (1, 3) else (i1 if i1[:2] == (1, 3) else None)
            y1 = i0 if i0[:2] == (1, 1) else (i1 if i1[:2] == (1, 1) else None)
            if y3 is not None and y1 is not None:
                return f"y3X{y3[2][0]}_yX{y1[2][0]}q{d}_cross"
            return None
        return None
    return None


def correction_terms_general(monomial, data_eigvals, betas, noise_var, context_length, dim):
    """Multidim corrected eigenvalue factor (and importance tag) for one monomial. The monomial is
    split into a context block (X/y features, coupled across X-dimensions and the sequence) and an
    independent query block. The query contributes prod_q data_eigvals[q]^exp; the context
    contributes the eigenvalue of the eigenmode it most overlaps -- deg 1/2 from the hand-built
    covariance blocks (Hungarian, collision-free), deg>=3 from the symmetry-adapted family engine."""
    C_len = context_length
    betas = ensure_numpy(betas)
    total_deg = sum(monomial.values())
    is_impt = "0"
    if total_deg == 0:
        return 1.0, is_impt

    eps = noise_var
    # 1. Split context vs query features.
    context_idx, query_idx = {}, {}
    for idx, exp in monomial.items():
        typ = parse_idx(idx, C_len, dim)[0]
        (query_idx if typ == 'q' else context_idx)[idx] = exp

    # Query factor and (expanded) query dimensions.
    query_factor = 1.0
    query_dims = []
    for idx, exp in query_idx.items():
        query_factor *= data_eigvals[idx] ** exp
        query_dims += [parse_idx(idx, C_len, dim)[2]] * exp

    ctx_deg = sum(context_idx.values())
    v = float(data_eigvals[0])   # isotropic X context variance (sigma_x^2)

    # Multidim context state: token -> (x_0,...,x_{dim-1}, b).
    context_state = {}
    for idx, exp in context_idx.items():
        typ, t, dd = parse_idx(idx, C_len, dim)
        s = list(context_state.get(t, (0,) * (dim + 1)))
        if typ == 'X':
            s[dd] += exp
        else:
            s[dim] += exp
        context_state[t] = tuple(s)

    # 2. Context factor by context degree.
    if ctx_deg == 0:
        context_factor = 1.0
    elif ctx_deg == 1:
        idx = next(iter(context_idx))
        typ, t, dd = parse_idx(idx, C_len, dim)
        row = dd if typ == 'X' else dim
        context_factor = _md_feature_eigval(_build_sigma_A(betas, v, eps, dim), row)
    elif ctx_deg == 2:
        keys = []
        for idx, exp in context_idx.items():
            keys += [idx] * exp
        idx1, idx2 = keys[0], keys[1]
        type1, t1, d1 = parse_idx(idx1, C_len, dim)
        type2, t2, d2 = parse_idx(idx2, C_len, dim)
        if t1 == t2:
            C_same, C_seq = build_same_token_matrices(betas, v, eps)
            N = C_same.shape[0]
            norm = np.ones(N)
            norm[:dim] = 1 / np.sqrt(2)
            norm[-1] = 1 / np.sqrt(2)
            NM = np.diag(norm)
            D_full = NM @ C_same @ NM
            C_seq_norm = NM @ C_seq @ NM
            modifier = (C_len - 1) if t1 == 0 else -1
            M_eval = D_full + modifier * C_seq_norm
            row = get_same_token_idx(type1, d1, type2, d2, dim)
            context_factor = _md_feature_eigval(M_eval, row)
        else:
            if t1 > t2:
                type1, type2 = type2, type1
                d1, d2 = d2, d1
            C_cross = build_cross_token_matrix(betas, v, eps)
            row = get_cross_token_idx(type1, d1, type2, d2, dim)
            context_factor = _md_feature_eigval(C_cross, row)
    else:
        family_part = tuple(sorted([sum(s) for s in context_state.values() if sum(s) > 0], reverse=True))
        L_seq = C_len - 1
        *_, assignment = get_cached_general_roots(betas, ctx_deg, family_part, L_seq, v, eps, dim)
        context_factor = assign_higher_degree_eigval_md(context_state, assignment)

    total_root = context_factor * query_factor

    # 3. Importance flags ({d} = query dimension).
    if total_deg == 1 and len(query_dims) == 1 and ctx_deg == 0:
        is_impt = f"q{query_dims[0]}"
    elif len(query_dims) == 1 and ctx_deg >= 2:
        flag = _md_impt_flag(context_state, query_dims[0], total_deg, dim)
        if flag is not None:
            is_impt = flag

    return total_root * math.factorial(total_deg), is_impt

def compute_kicl_eigval_multidim(data_eigvals, monomial, eval_level_coeff, context_length, dim, betas, noise_var=0.0):
    r"""
    Computes the eigenvalue for a given monomial, correctly handling the
    X-y task coupling across the sequence in the \ell=1 subspace.
    
    X_flat = [X.flatten(), y.flatten(), Xq.flatten()] where shapes are:
    - X: (context_len-1, dim)
    - Xq: (dim)
    len(X_flat) = (context_len-1)*dim + (context_len-1) + dim
    """
    assert betas is not None, "Eigvals need beta stats to compute X_ctx-like eigvals"
    
    # betabar, beta_var = get_beta_stats(ensure_numpy(betas), dim)

    degree = sum(monomial.values())
    kicl_eigval = eval_level_coeff(degree)
    is_impt = "0"
    if kicl_eigval == 0: return 0, "0"
    # idx_x_ctx_end = (context_length-1) * dim
    # idx_y_ctx_end = 2 * (context_length-1) * dim

    if degree == 0:
        return kicl_eigval, is_impt
    else:# degree <= 3:
        coupled_root, is_impt = correction_terms_general(monomial, data_eigvals, ensure_numpy(betas), noise_var, context_length, dim)
        return kicl_eigval * coupled_root, is_impt
    #fallback
    return kicl_eigval, is_impt


def generate_multidim_kicl_monomials(data_eigvals, num_monomials, eval_level_coeff, context_length=10, dim=1, betas=None, noise_var=0.0, kmax=10, aggregate_high_degree=False):
    """
    Generates KICL eigenvalues and monomials in canonical learning order.

    Args:
        data_eigvals (iterable): data covariance eigenvalues
        num_monomials (int): Number of monomials to generate.
        eval_level_coeff (function): Function to evaluate kernel level coefficients.
        kmax (int): Search monomials up to degree kmax
        aggregate_high_degree (bool): If True, stop enumerating individual degree>=4 monomials;
            the returned lists hold only degree<=3 monomials plus the important degree-5 ICL terms,
            and `degree_mass` carries the per-degree summed eigenvalue mass for degrees >=4.
            If False (default), enumerate the full spectrum; `degree_mass` is still returned.

    Returns:
        - kicl_eigvals (np.ndarray): Array of KICL eigenvalues.
        - monomials (list): List of generated monomials.
        - impt_list (list): Importance tag per monomial.
        - degree_mass (dict): {degree: total spectral mass} for degrees 4..kmax.
    """
    try:
        num_monomials = abs(int(num_monomials))
    except Exception as e:
        raise ValueError(f"type(num_monomials) must be int, not {type(num_monomials)}") from e
    assert num_monomials >= 1
    data_eigvals = ensure_numpy(data_eigvals)
    d = len(data_eigvals)

    # When aggregating, only individual monomials up to degree 3 are enumerated; the degree>=4
    # bulk is summarized by degree_mass and the important degree-5 terms are appended explicitly.
    enum_kmax = 3 if aggregate_high_degree else kmax

    pq = []
    seen_monomials = set()
    
    first_kicl_eigval, is_impt = compute_kicl_eigval_multidim(data_eigvals, Monomial({}), eval_level_coeff, context_length, dim, betas=betas, noise_var=noise_var)
    first_kicl_eigval = abs(first_kicl_eigval)
    
    # for k in range(1, kmax+1):
    #     monomial = Monomial({0: k})
    #     kicl_eigval, is_impt = compute_kicl_eigval(data_eigvals, monomial, eval_level_coeff, context_length, dim, betas=betas, noise_var=noise_var)
    #     kicl_eigval = abs(kicl_eigval)
    #     pq.append((-kicl_eigval, monomial, is_impt))
    #     seen_monomials.add(frozenset(monomial.items()))

    for k in range(1, enum_kmax+1):
        for i in range(d):
            monomial = Monomial({i: k})
            kicl_eigval, is_impt = compute_kicl_eigval_multidim(data_eigvals, monomial, eval_level_coeff, context_length, dim, betas=betas, noise_var=noise_var)
            kicl_eigval = abs(kicl_eigval)
            pq.append((-kicl_eigval, monomial, is_impt))
            seen_monomials.add(frozenset(monomial.items()))
        
    heapq.heapify(pq)
    
    monomials = [Monomial({})]
    kicl_eigvals = [first_kicl_eigval]
    impt_list = [is_impt]
    
    for _ in range(num_monomials-1):
        if not pq:
            if not aggregate_high_degree:
                print("Warning: priority queue exhausted before reaching num_monomials.")
            break

        neg_kicl_eigval, monomial, is_impt = heapq.heappop(pq)
        
        kicl_eigvals.append(-neg_kicl_eigval)
        monomials.append(monomial)
        impt_list.append(is_impt)
        
        # --- Neighbor generation ---
        total_deg = sum(monomial.values())

        def maybe_push(nm):
            key = frozenset(nm.items())
            if key not in seen_monomials:
                ev, ii = compute_kicl_eigval_multidim(data_eigvals, nm, eval_level_coeff, context_length, dim, betas=betas, noise_var=noise_var)
                heapq.heappush(pq, (-abs(ev), nm, ii))
                seen_monomials.add(key)

        # 1. Non-adjacent single-unit shifts (generalises old idx -> idx+1)
        for idx in list(monomial.keys()):
            for idx2 in range(idx + 1, d):
                nm = monomial.copy()
                nm[idx] -= 1
                if nm[idx] == 0:
                    del nm[idx]
                nm[idx2] = nm.get(idx2, 0) + 1
                maybe_push(nm)

        # 2. Full shift (all indices up by 1)
        shifted = Monomial({})
        valid = True
        for idx, exp in monomial.items():
            if idx + 1 < d:
                shifted[idx + 1] = exp
            else:
                valid = False
                break
        if valid:
            maybe_push(shifted)

        # 3. Degree-increase neighbors (up to enum_kmax)
        if total_deg < enum_kmax:
            for idx in list(monomial.keys()):
                nm = monomial.copy()
                nm[idx] += 1
                maybe_push(nm)
            for new_idx in range(d):
                if new_idx not in monomial:
                    nm = monomial.copy()
                    nm[new_idx] = 1
                    maybe_push(nm)

    # When aggregating, append the important degree-5/7 ICL terms (not otherwise enumerated).
    if aggregate_high_degree:
        for nm in _important_query_monomials(context_length, dim):
            key = frozenset(nm.items())
            if key in seen_monomials:
                continue
            ev, ii = compute_kicl_eigval_multidim(data_eigvals, nm, eval_level_coeff, context_length, dim, betas=betas, noise_var=noise_var)
            kicl_eigvals.append(abs(ev))
            monomials.append(nm)
            impt_list.append(ii)
            seen_monomials.add(key)

    # Per-degree summed eigenvalue mass for degrees >=4 (via family traces). Only computed in
    # aggregate mode, where it replaces the un-enumerated deg>=4 bulk; in full-spectrum mode it is
    # left as None so the default path does not pay the (one-time, cached) family-building cost.
    degree_mass = None
    if aggregate_high_degree:
        degree_mass = compute_degree_masses(
            data_eigvals, eval_level_coeff, context_length, dim, betas, noise_var, kmax, min_degree=4)

    return np.array(kicl_eigvals), monomials, impt_list, degree_mass


# =============================================================================
# =============================================================================
# ===                                                                       ===
# ===   EIGENFUNCTION BASIS  --  ALL NEW CODE BELOW THIS LINE               ===
# ===                                                                       ===
# ===   Everything ABOVE is the pre-existing eigenvalue engine, left        ===
# ===   completely unmodified.  Everything BELOW is new.                    ===
# ===                                                                       ===
# ===   compute_basis(...) is to compute_kicl_eigval_multidim(...) what an  ===
# ===   eigenvector is to an eigenvalue.                                    ===
# ===                                                                       ===
# ===   THE PRINCIPLE.  Every branch of correction_terms_general builds a   ===
# ===   covariance matrix M whose ROWS are specific feature functions       ===
# ===   f_r(X, y) of the data, diagonalizes it, and hands the monomial the  ===
# ===   eigenvalue lam_j = |eigvals[j]| of the column j picked by that      ===
# ===   branch's assignment rule.  The matching eigenfunction is therefore  ===
# ===                                                                       ===
# ===       psi = ( sum_r U[r, j] * f_r(data) ) / sqrt(lam_j)               ===
# ===                                                                       ===
# ===   with the SAME M, the SAME f_r and the SAME j.  compute_basis is     ===
# ===   that one formula, branch for branch, so H[:, i] is the             ===
# ===   eigenfunction whose eigenvalue is the i-th entry returned by        ===
# ===   generate_multidim_kicl_monomials.  E[psi^2] = 1 by construction.    ===
# ===                                                                       ===
# ===   NOTE.  kernel_theory.compute_corrected_basis is NOT the model for   ===
# ===   this code.  It parses the flat index with a y-block of width        ===
# ===   (C-1)*dim instead of (C-1); hardcodes 1-D 3x3 / 4x4 blocks instead  ===
# ===   of the dim-aware matrices the engine actually diagonalizes; applies ===
# ===   a "D_scale" rescale with no counterpart in the eigenvalue path;     ===
# ===   picks eigenvectors by hardcoded column index rather than the        ===
# ===   Hungarian bijection; partitions by feature dimension (severing the  ===
# ===   X_a-X_b and X_a-y coupling); evaluates the D - C contrast           ===
# ===   covariance on a single literal token; never centers X*y or y^2;     ===
# ===   and never divides by sqrt(lam).  None of that is carried over.      ===
# ===                                                                       ===
# =============================================================================
# =============================================================================


def _he_norm(k, z):
    """Normalized probabilists Hermite He_k(z) / sqrt(k!), evaluated elementwise.

    Built from the recurrence He_{n+1}(z) = z He_n(z) - n He_{n-1}(z), so it is not capped at
    degree 20 the way kernel_theory.get_hermite_polynomials is, and so this module stays
    torch-free.  E[He_k(Z)^2 / k!] = 1 for Z ~ N(0, 1), the unit-variance convention every basis
    element below uses.
    """
    z = np.asarray(z, dtype=np.float64)
    if k == 0:
        return np.ones_like(z)
    hm1 = np.ones_like(z)
    h = z.copy()
    for n in range(1, k):
        hm1, h = h, z * h - n * hm1
    return h / np.sqrt(float(math.factorial(k)))


def _md_feature_mode(M_eval, row_idx):
    """Mode counterpart of _md_feature_eigval: the same Hungarian (bijective, collision-free)
    assignment over a deg-1/2 covariance block, but returning the eigenvalue, the eigenvector AND
    the column index of the eigenmode that feature `row_idx` most overlaps.

    _md_feature_eigval(M_eval, row_idx) == _md_feature_mode(M_eval, row_idx)[0], exactly.
    """
    eigvals, eigvecs = np.linalg.eigh(M_eval)
    ri, ci = linear_sum_assignment(-(eigvecs ** 2))
    col = int(dict(zip(ri, ci))[row_idx])
    return abs(eigvals[col]), eigvecs[:, col], col


def _build_higher_degree_mode_assignment_md(eigvals, eigvecs, basis_F):
    """Column-index counterpart of _build_higher_degree_assignment_md.

    Line for line the same symmetry-adapted matching (orbit-sum probes -> reduced symmetric
    matrix -> Hungarian -> canonical representatives -> leftover shifted states), but stores the
    eigenvector COLUMN INDEX where the original stores abs(eigvals[...]).  So

        assignment[key] == abs(eigvals[ modes[key] ])

    for every key, which is what keeps compute_basis normalization consistent with
    assign_higher_degree_eigval_md.  Returns {state_key: column_index}.
    """
    sF = len(basis_F)
    Cw = (eigvecs * eigvals) @ eigvecs.T
    orbit_of = [_orbit_key_md(b) for b in basis_F]
    orbit_keys = list(dict.fromkeys(orbit_of))

    probes = np.zeros((sF, len(orbit_keys)))
    for col, k in enumerate(orbit_keys):
        sel = np.array([1.0 if ok == k else 0.0 for ok in orbit_of])
        probes[:, col] = sel / np.linalg.norm(sel)
    S = probes.T @ Cw @ probes
    sev, svec = np.linalg.eigh(S)

    ri, ci = linear_sum_assignment(-(svec ** 2))
    matched = sorted(zip(ri, ci), key=lambda tc: -sev[tc[1]])

    modes = {}
    used = set()
    canonical_col = {}
    for t, j_sym in matched:
        full_vec = probes @ svec[:, j_sym]
        ov = np.abs(eigvecs.T @ full_vec)
        for j in np.argsort(-ov):
            if j not in used:
                used.add(int(j))
                canonical_col[orbit_keys[t]] = int(j)
                break

    for i, b in enumerate(basis_F):
        if _is_canonical_state_md(b):
            modes[_state_key_md(b)] = canonical_col[orbit_of[i]]

    rem_rows = [i for i, b in enumerate(basis_F) if _state_key_md(b) not in modes]
    rem_cols = [j for j in range(sF) if j not in used]
    if rem_rows:
        sub = np.abs(eigvecs[np.ix_(rem_rows, rem_cols)]) ** 2
        rr, cc = linear_sum_assignment(-sub)
        for a, c in zip(rr, cc):
            modes[_state_key_md(basis_F[rem_rows[a]])] = int(rem_cols[c])
    return modes


def _helmert_contrasts(L):
    """Orthonormal (L, L) sequence-mode matrix.  Row 0 is the symmetric mode ones/sqrt(L); rows
    1..L-1 are Helmert contrasts (zero-sum, mutually orthogonal, unit norm).

    This is what a monomial token index MEANS in the symmetry-adapted convention that
    correction_terms_general uses for degree-2 same-token terms.  With g_t the feature vector of
    token t, Cov(g_t) = D and Cov(g_t, g_s) = C for t != s, so

        token 0  ->  (1/sqrt(L)) sum_t g_t   has covariance  D + (L-1) C   (symmetric mode)
        token k  ->  sum_t helm[k,t] g_t     has covariance  D - C         (contrast mode)

    which is exactly the `modifier` split in correction_terms_general.  The L-1 contrast modes are
    degenerate, so any orthonormal basis of the zero-sum complement is a valid eigenbasis; Helmert
    is chosen because it is deterministic, so monomials carrying different token indices come out
    orthogonal to each other.
    """
    Cmat = np.zeros((L, L))
    Cmat[0, :] = 1.0 / np.sqrt(float(L))
    for k in range(1, L):
        nrm = 1.0 / np.sqrt(float(k) * (k + 1.0))
        Cmat[k, :k] = nrm
        Cmat[k, k] = -float(k) * nrm
    return Cmat


def _state_values_md(X, state, C_len, dim):
    """Evaluate one raw multidim sequence-state monomial on the sample.

    `state` is {token: (x_0, ..., x_{dim-1}, b)}, the same object the engine feeds to
    combine_states_md, and this returns prod_t [ prod_d X_{t,d}^{x_d} ] * y_t^b as an (N,) array.
    Sample-space twin of get_moment_generator_md: E[_state_values_md(...)] == E(*state.values()).
    """
    x_end = (C_len - 1) * dim
    out = np.ones(X.shape[0])
    for t, st in state.items():
        for d in range(dim):
            e = st[d]
            if e:
                out = out * X[:, t * dim + d] ** e
        b = st[dim]
        if b:
            out = out * X[:, x_end + t] ** b
    return out


def _same_token_features_md(X, t, C_len, dim, v, mu1, E_S, eps):
    """The (N, 2*dim + dim(dim-1)/2 + 1) feature block whose covariance is
    NM @ C_same @ NM, in the row order get_same_token_idx indexes.

    Rows, matching build_same_token_matrices exactly (centered, then NM-weighted):

        [0, dim)            (X_{t,a}^2 - v) / sqrt(2)                 <- norm[:dim] = 1/sqrt(2)
        [dim, dim+d_mix)    X_{t,a} X_{t,b},  a < b                   <- mean 0
        [dim+d_mix, ...)    X_{t,a} y_t - v * mu1[a]
        last                (y_t^2 - (v * E_S + eps)) / sqrt(2)       <- norm[-1] = 1/sqrt(2)

    The centering is what makes these *covariance* rows rather than second-moment rows, which is
    what build_same_token_matrices computes (e.g. E_C[X_a^2, X_a^2] = 2 v^2 = Var(X_a^2), not
    E[X_a^4] = 3 v^2).  The 1/sqrt(2) factors are the NM normalization applied inside
    correction_terms_general.
    """
    N = X.shape[0]
    d_mix = dim * (dim - 1) // 2
    n_feat = 2 * dim + d_mix + 1
    x_end = (C_len - 1) * dim
    y = X[:, x_end + t]

    G = np.empty((N, n_feat))
    for a in range(dim):
        xa = X[:, t * dim + a]
        G[:, a] = (xa * xa - v) / np.sqrt(2.0)
    m = 0
    for a in range(dim):
        for b in range(a + 1, dim):
            G[:, dim + m] = X[:, t * dim + a] * X[:, t * dim + b]
            m += 1
    for a in range(dim):
        G[:, dim + d_mix + a] = X[:, t * dim + a] * y - v * mu1[a]
    G[:, n_feat - 1] = (y * y - (v * E_S + eps)) / np.sqrt(2.0)
    return G


def _cross_token_features_md(X, ti, tj, C_len, dim):
    """The (N, dim^2 + 2*dim + 1) feature block whose covariance is build_cross_token_matrix, in
    the row order get_cross_token_idx indexes.  `ti` is the EARLIER token (ti < tj), matching the
    swap correction_terms_general performs before calling get_cross_token_idx.

        [0, dim)                X_{i,a} X_{j,a}                    (para)
        [dim, dim+dim(dim-1))   X_{i,a} X_{j,b},  a != b           (perp, a-major loop)
        idx_Xy + a              X_{i,a} y_j
        idx_yX + a              y_i X_{j,a}
        last                    y_i y_j

    Every cross-token product has mean zero (the two tokens are conditionally independent given
    beta and each factor is conditionally centered), so no centering is needed here -- unlike the
    same-token block.  Cross-token features supported on DIFFERENT unordered token pairs are
    exactly uncorrelated, which is why the literal pair (ti, tj) is already an eigenfunction and no
    sequence-contrast machinery is needed on this branch.
    """
    N = X.shape[0]
    M = dim ** 2 + 2 * dim + 1
    x_end = (C_len - 1) * dim
    yi = X[:, x_end + ti]
    yj = X[:, x_end + tj]

    G = np.empty((N, M))
    for a in range(dim):
        G[:, a] = X[:, ti * dim + a] * X[:, tj * dim + a]
    p = 0
    for a in range(dim):
        for b in range(dim):
            if a == b:
                continue
            G[:, dim + p] = X[:, ti * dim + a] * X[:, tj * dim + b]
            p += 1
    idx_Xy = dim + dim * (dim - 1)
    for a in range(dim):
        G[:, idx_Xy + a] = X[:, ti * dim + a] * yj
        G[:, idx_Xy + dim + a] = yi * X[:, tj * dim + a]
    G[:, M - 1] = yi * yj
    return G


def _deg1_token_features_md(X, t, C_len, dim):
    """The (N, dim+1) feature block [X_{t,0}, ..., X_{t,dim-1}, y_t] whose covariance is
    _build_sigma_A.  All rows are mean zero.

    Degree-1 features on different tokens are exactly uncorrelated (Cov(y_i, y_j) = 0 because
    E[y|beta] = 0), so the literal token t is already an eigenfunction and gives columns that are
    orthogonal across monomials -- there is no symmetric/contrast split at this degree, which is
    why correction_terms_general applies no `modifier` here either.
    """
    N = X.shape[0]
    x_end = (C_len - 1) * dim
    G = np.empty((N, dim + 1))
    for a in range(dim):
        G[:, a] = X[:, t * dim + a]
    G[:, dim] = X[:, x_end + t]
    return G


_MD_BASIS_CACHE = {}


def _get_general_basis_data(betas, deg, family_part, L_seq, v, eps, dim):
    """Everything needed to REBUILD the degree>=3 family eigenfunctions, not just their eigenvalues.

    get_cached_general_roots residualizes the family monomials against the lower same-parity basis
    (C_perp = C_FF - C_FL C_LL^+ C_FL^T) but does not return C_FL, so the projection coefficients
    A = C_FL @ C_LL_pinv are recomputed here.  That is cheap: the moment generator E is memoized
    per (betas, v, eps, dim), so every entry is a dictionary hit after the eigenvalue path has run.

    With m_r the raw family monomial, l_c the lower-basis monomials and W = get_weight_md, the
    covariance rows of C_weighted are the features

        f_r = W_r * ( m_r - sum_c A[r, c] * l_c )

    and the eigenfunction of column j is sum_r eigvecs[r, j] * f_r, with E[.^2] = eigvals[j].
    For even degrees the lower basis contains the constant, so this residualization also centers.

    Returns (eigvals, eigvecs, basis_F, lower_basis, A, W, modes).
    """
    key = (_betas_key(betas), deg, family_part, L_seq, float(v), float(eps), dim)
    cached = _MD_BASIS_CACHE.get(key)
    if cached is not None:
        return cached

    eigvals, eigvecs, basis_F, _trace, _assignment = get_cached_general_roots(
        betas, deg, family_part, L_seq, v, eps, dim)

    E = _get_md_generator(betas, v, eps, dim)
    lower_basis, C_LL_pinv = _get_lower_inverse_md(betas, deg, L_seq, v, eps, dim)

    size_F = len(basis_F)
    if lower_basis:
        size_L = len(lower_basis)
        C_FL = np.zeros((size_F, size_L))
        for r in range(size_F):
            for c in range(size_L):
                C_FL[r, c] = E(*combine_states_md(basis_F[r], lower_basis[c], dim))
        A = C_FL @ C_LL_pinv
    else:
        A = None

    W = np.array([get_weight_md(s) for s in basis_F])
    modes = _build_higher_degree_mode_assignment_md(eigvals, eigvecs, basis_F)

    result = (eigvals, eigvecs, basis_F, lower_basis, A, W, modes)
    _MD_BASIS_CACHE[key] = result
    return result



def _finish(psi_ctx, q_col, lam, N):
    """psi_ctx * psi_query / sqrt(lam), then the 1/sqrt(N) that makes columns unit L2 norm."""
    if lam is None or lam < 1e-300:
        return np.zeros(N)
    return psi_ctx * q_col / (np.sqrt(lam) * np.sqrt(N))


def compute_basis(X, monomials, context_length, dim, betas, noise_var, data_eigvals,
                  is_X_PCAd=True, high_degree="exact", max_family_size=4000,
                  return_info=False):
    r"""Theoretical KICL eigenfunctions, evaluated on data -- the eigenvector counterpart of
    compute_kicl_eigval_multidim.

    For each monomial this walks the SAME branches as correction_terms_general, rebuilds the SAME
    covariance matrix, picks the SAME eigenvector column via the SAME assignment rule, and
    evaluates the corresponding linear combination of feature functions on the sample:

        psi_ctx = ( sum_r U[r, j] f_r ) / sqrt(lam_j)      E[psi_ctx^2] = 1
        psi_q   = prod_q He_{k_q}( X_q / sqrt(sigma_q) ) / sqrt(k_q!)
        H[:, i] = psi_ctx * psi_q / sqrt(N)

    so H has unit-L2-norm columns (the convention compute_hermite_basis uses and the convention
    kernel_eigsys.eigendecompose returns emp_eigvecs in), and column i is the eigenfunction whose
    eigenvalue is entry i of generate_multidim_kicl_monomials.  The level coefficient and the
    factorial(total_deg) that compute_kicl_eigval_multidim multiplies in are kernel scalars, not
    part of the eigenfunction, so they do not appear here.

    Branch table (mirroring correction_terms_general exactly):

        ctx_deg 0        constant
        ctx_deg 1        _build_sigma_A, row = d (X) or dim (y), literal token
        ctx_deg 2, same  NM C_same NM + modifier * NM C_seq NM, row = get_same_token_idx,
                         evaluated on the Helmert sequence mode named by the token index
        ctx_deg 2, cross build_cross_token_matrix, row = get_cross_token_idx, literal token pair
        ctx_deg >= 3     get_cached_general_roots family engine, column from the mirrored
                         symmetry-adapted assignment

    Args:
        X: (N, D) RAW natural-scale flat features
            [X_ctx.flatten (L-1, dim) | y_ctx.flatten (L-1) | X_q.flatten (dim)],
            i.e. exactly RegressionTaskDistribution._get_input output.  NOT whitened: the whole
            theory is stated in natural variances (v = data_eigvals[0],
            Var(y) = v E||beta||^2 + eps, Cov(X_a, y) = v mu1[a]) and SVD-whitening destroys
            precisely the structure being modelled.
        monomials: Monomial (or list of them) over flat feature indices, as returned by
            generate_multidim_kicl_monomials.
        context_length: L, so there are L-1 context tokens.
        dim: X feature dimension.
        betas: (T, dim) pretrain task weights.
        noise_var: label noise variance eps.
        data_eigvals: per-flat-index data variances; data_eigvals[0] is the isotropic context X
            variance v, and the query entries set the query Hermite rescale.
        is_X_PCAd: kept for signature compatibility with compute_hermite_basis.  Defaults to True
            (= "X is already in the space I want"), which is the correct setting here.  Passing
            False SVD-whitens X first and is NOT recommended -- see the note on X above.
        high_degree: "exact" (default) rebuilds degree>=3 context eigenfunctions from the family
            engine; "hermite" falls back to plain normalized Hermite products for those, which is
            far cheaper but is not an eigenfunction of the ICL kernel.
        max_family_size: degree>=3 families larger than this fall back to the Hermite product and
            print a one-time warning.  Guards the O(|basis_F| * N * k) rebuild.
        return_info: also return a per-monomial list of dicts with the branch taken, the
            eigenvalue used, the eigenvector column, and the realized column norm.

    Returns:
        H: (N, len(monomials)) float64.  Or (H, info) if return_info.
    """
    if high_degree not in ("exact", "hermite"):
        raise ValueError(f"high_degree must be 'exact' or 'hermite', got {high_degree!r}")

    X = ensure_numpy(X)
    if not is_X_PCAd:
        # Signature compatibility with compute_hermite_basis only.  Whitening flattens the natural
        # variances the corrections are built from, so the resulting columns are not the theory's
        # eigenfunctions.
        N_ = X.shape[0]
        U, _, _ = np.linalg.svd(X, full_matrices=False)
        X = np.sqrt(N_) * U
    X = np.asarray(X, dtype=np.float64)

    N = X.shape[0]
    C_len = int(context_length)
    L_seq = C_len - 1
    x_end = L_seq * dim

    # Double conversion on purpose: ensure_numpy returns float32 for a torch tensor but casts a
    # numpy array to float64, and the engine calls it twice (compute_kicl_eigval_multidim, then
    # correction_terms_general), so it always ends up working in float64.  Matching that matters
    # for more than rounding: _betas_key hashes the raw bytes, so a float32 array is a different
    # family-engine cache key, and the deg-2 cross block has exactly-degenerate rows (X_i y_j vs
    # y_i X_j) whose Hungarian pairing a 1e-9 perturbation is enough to flip.
    betas = ensure_numpy(ensure_numpy(betas)).reshape(-1, dim)
    data_eigvals = ensure_numpy(data_eigvals)
    v = float(data_eigvals[0])
    eps = float(noise_var)

    mu1 = np.mean(betas, axis=0)
    E_S = float(np.mean(np.sum(betas ** 2, axis=1)))
    var_y = v * E_S + eps

    if not isinstance(monomials, list):
        monomials = [monomials]
    n_mono = len(monomials)
    H = np.zeros((N, n_mono))
    info = [None] * n_mono

    # ---- covariance blocks, built once (they do not depend on which token/pair) --------------
    Sigma_A = _build_sigma_A(betas, v, eps, dim)
    C_same, C_seq = build_same_token_matrices(betas, v, eps)
    n_same = C_same.shape[0]
    norm = np.ones(n_same)
    norm[:dim] = 1 / np.sqrt(2)
    norm[-1] = 1 / np.sqrt(2)
    NM = np.diag(norm)
    D_full = NM @ C_same @ NM
    C_seq_norm = NM @ C_seq @ NM
    # `modifier` is read from the same expression correction_terms_general uses, so eigenvalue and
    # eigenvector can never disagree.  NOTE: the symmetric mode uses (C_len - 1) = L_seq here,
    # whereas the 1-D correction_terms (kernel_theory.py, "D_full + (L_seq - 1) * C_full") and the
    # algebra for a unit-norm symmetric combination both give L_seq - 1.  Deliberately mirrored,
    # not fixed: if the engine is corrected, this picks it up automatically.
    M_same = {0: D_full + (C_len - 1) * C_seq_norm, 1: D_full - C_seq_norm}
    C_cross = build_cross_token_matrix(betas, v, eps)

    helm = _helmert_contrasts(L_seq) if L_seq > 0 else np.ones((1, 1))

    # ---- lazily-built sample feature blocks and cached mode lookups --------------------------
    _deg1_cache = {}
    _same_cache = {}
    _cross_cache = {}
    _mode_cache = {}

    def deg1_feats(t):
        G = _deg1_cache.get(t)
        if G is None:
            G = _deg1_token_features_md(X, t, C_len, dim)
            _deg1_cache[t] = G
        return G

    def same_feats(t):
        G = _same_cache.get(t)
        if G is None:
            G = _same_token_features_md(X, t, C_len, dim, v, mu1, E_S, eps)
            _same_cache[t] = G
        return G

    def cross_feats(ti, tj):
        G = _cross_cache.get((ti, tj))
        if G is None:
            G = _cross_token_features_md(X, ti, tj, C_len, dim)
            _cross_cache[(ti, tj)] = G
        return G

    def feature_mode(tag, M_eval, row):
        got = _mode_cache.get((tag, row))
        if got is None:
            got = _md_feature_mode(M_eval, row)
            _mode_cache[(tag, row)] = got
        return got

    def hermite_context(context_idx):
        """Uncorrected fallback: independent normalized Hermites in each context feature."""
        h = np.ones(N)
        for idx, exp in context_idx.items():
            typ = parse_idx(idx, C_len, dim)[0]
            scale = np.sqrt(v if typ == 'X' else var_y)
            h = h * _he_norm(exp, X[:, idx] / scale)
        return h

    pending = {}          # (ctx_deg, family_part) -> [(monomial index, state_key, context_idx)]
    warned_families = set()

    # =========================== pass 1: split, query factor, deg <= 2 =======================
    for i, monomial in enumerate(monomials):
        context_idx, query_idx = {}, {}
        for idx, exp in monomial.items():
            typ = parse_idx(idx, C_len, dim)[0]
            (query_idx if typ == 'q' else context_idx)[idx] = exp

        # Query block: independent of the context, unit variance, so it simply multiplies.  The
        # data_eigvals[idx]**exp the engine charges is the EIGENVALUE, not part of the function,
        # so it is accumulated separately as query_factor for compute_kicl_eigsys_multidim.
        q_col = np.ones(N)
        query_factor = 1.0
        query_dims = []
        for idx, exp in query_idx.items():
            sq = float(data_eigvals[idx])
            z = X[:, idx] / np.sqrt(sq) if sq > 0 else X[:, idx]
            q_col = q_col * _he_norm(exp, z)
            query_factor *= sq ** exp
            query_dims += [parse_idx(idx, C_len, dim)[2]] * exp

        ctx_deg = sum(context_idx.values())
        total_deg = ctx_deg + sum(query_idx.values())

        # Multidim context state: token -> (x_0,...,x_{dim-1}, b).  Same construction as
        # correction_terms_general; needed for the deg>=3 family key and for the importance tag.
        context_state = {}
        for idx, exp in context_idx.items():
            typ, t, dd = parse_idx(idx, C_len, dim)
            s = list(context_state.get(t, (0,) * (dim + 1)))
            if typ == 'X':
                s[dd] += exp
            else:
                s[dim] += exp
            context_state[t] = tuple(s)

        # Importance flag, mirroring step 3 of correction_terms_general exactly.
        is_impt = "0"
        if total_deg == 1 and len(query_dims) == 1 and ctx_deg == 0:
            is_impt = f"q{query_dims[0]}"
        elif len(query_dims) == 1 and ctx_deg >= 2:
            flag = _md_impt_flag(context_state, query_dims[0], total_deg, dim)
            if flag is not None:
                is_impt = flag

        base = {"degree": total_deg, "ctx_deg": ctx_deg,
                "query_factor": query_factor, "is_impt": is_impt}

        if ctx_deg == 0:
            H[:, i] = q_col / np.sqrt(N)
            info[i] = dict(base, branch="query-only", **{"lambda": 1.0, "column": None})
            continue

        if ctx_deg == 1:
            idx = next(iter(context_idx))
            typ, t, dd = parse_idx(idx, C_len, dim)
            row = dd if typ == 'X' else dim
            lam, evec, col = feature_mode("sigmaA", Sigma_A, row)
            psi = deg1_feats(t) @ evec
            H[:, i] = _finish(psi, q_col, lam, N)
            info[i] = dict(base, branch="deg1", **{"lambda": lam, "column": col})
            continue

        if ctx_deg == 2:
            keys = []
            for idx, exp in context_idx.items():
                keys += [idx] * exp
            idx1, idx2 = keys[0], keys[1]
            type1, t1, d1 = parse_idx(idx1, C_len, dim)
            type2, t2, d2 = parse_idx(idx2, C_len, dim)

            if t1 == t2:
                row = get_same_token_idx(type1, d1, type2, d2, dim)
                tag = "same0" if t1 == 0 else "same1"
                lam, evec, col = feature_mode(tag, M_same[0 if t1 == 0 else 1], row)
                # The eigenvector mixes feature TYPES; the Helmert row mixes TOKENS.  The joint
                # covariance is exchangeable (D on the diagonal, C off it), so its eigenvectors
                # factorize as (sequence mode) x (eigenvector of D + modifier*C) -- which is why
                # the same evec is used for every token in the combination.
                c = helm[t1]
                psi = np.zeros(N)
                for t in range(L_seq):
                    if c[t] != 0.0:
                        psi += c[t] * (same_feats(t) @ evec)
                H[:, i] = _finish(psi, q_col, lam, N)
                info[i] = dict(base, branch="deg2-same", **{"lambda": lam, "column": col})
            else:
                if t1 > t2:
                    type1, type2 = type2, type1
                    d1, d2 = d2, d1
                ti, tj = (t1, t2) if t1 < t2 else (t2, t1)
                row = get_cross_token_idx(type1, d1, type2, d2, dim)
                lam, evec, col = feature_mode("cross", C_cross, row)
                psi = cross_feats(ti, tj) @ evec
                H[:, i] = _finish(psi, q_col, lam, N)
                info[i] = dict(base, branch="deg2-cross", **{"lambda": lam, "column": col})
            continue

        # ---- ctx_deg >= 3: defer, grouped by family, so each family is built once ------------
        family_part = tuple(sorted([sum(s) for s in context_state.values() if sum(s) > 0],
                                   reverse=True))

        if high_degree == "hermite":
            # Already unit variance by construction, so no sqrt(lam) division.
            H[:, i] = _finish(hermite_context(context_idx), q_col, 1.0, N)
            info[i] = dict(base, branch="hermite-fallback",
                           **{"lambda": None, "column": None})
            continue

        H[:, i] = q_col / np.sqrt(N)   # park the query factor; pass 2 multiplies psi_ctx in
        info[i] = dict(base, branch="pending", **{"lambda": None, "column": None})
        pending.setdefault((ctx_deg, family_part), []).append(
            (i, _state_key_md(context_state), context_idx))

    # =========================== pass 2: degree >= 3 families ================================
    for (ctx_deg, family_part), items in pending.items():
        # Size the family from the (cheap, purely combinatorial) basis enumeration BEFORE
        # _get_general_basis_data, whose C_FL rebuild is the expensive part.
        size_F = len(get_family_basis_md(family_part, L_seq, dim))

        if size_F > max_family_size:
            if family_part not in warned_families:
                print(f"Warning: degree-{ctx_deg} family {family_part} has {size_F} basis "
                      f"elements (> max_family_size={max_family_size}); falling back to "
                      f"uncorrected Hermite products for its {len(items)} monomial(s).")
                warned_families.add(family_part)
            for i, _skey, context_idx in items:
                H[:, i] *= hermite_context(context_idx)
                info[i].update(branch="hermite-fallback")
            continue

        eigvals, eigvecs, basis_F, lower_basis, A, W, modes = _get_general_basis_data(
            betas, ctx_deg, family_part, L_seq, v, eps, dim)

        # Distinct eigenvector columns actually needed by this family.  Bounded by size_F, since
        # the assignment is a bijection over the family basis.
        cols, col_pos = [], {}
        for _i, skey, _ci in items:
            j = modes.get(skey)
            if j is not None and j not in col_pos:
                col_pos[j] = len(cols)
                cols.append(j)

        if cols:
            # Psi = (M_F - M_L A^T) diag(W) U[:, cols]
            #     = M_F @ G - M_L @ (A^T G),   G = W[:, None] * U[:, cols]
            # M_F is (N, size_F) and never materialized: it is accumulated one basis element at a
            # time.  Columns are processed in blocks so peak memory is O(N * block) rather than
            # O(N * len(cols)) -- with a full spectrum, len(cols) can reach size_F, and N * size_F
            # doubles is hundreds of MB.  The block loop re-walks the size_F monomial evaluations
            # once per block, which is the time-for-memory trade.
            acc = np.empty((N, len(cols)))
            block = max(1, int(8_000_000 // max(N, 1)))
            for b0 in range(0, len(cols), block):
                b1 = min(b0 + block, len(cols))
                G = W[:, None] * eigvecs[:, cols[b0:b1]]
                part = np.zeros((N, b1 - b0))
                for r in range(size_F):
                    gr = G[r]
                    if not np.any(gr):
                        continue
                    part += _state_values_md(X, basis_F[r], C_len, dim)[:, None] * gr[None, :]
                if A is not None and lower_basis:
                    GL = A.T @ G
                    for c in range(len(lower_basis)):
                        gc = GL[c]
                        if not np.any(gc):
                            continue
                        part -= (_state_values_md(X, lower_basis[c], C_len, dim)[:, None]
                                 * gc[None, :])
                acc[:, b0:b1] = part
        else:
            acc = None

        for i, skey, _ci in items:
            j = modes.get(skey)
            if j is None:
                # Mirrors assign_higher_degree_eigval_md returning 0.0 for an unassigned state.
                H[:, i] = 0.0
                info[i].update(branch="deg>=3-unassigned", **{"lambda": 0.0})
                continue
            lam = abs(float(eigvals[j]))
            if lam < 1e-300:
                H[:, i] = 0.0
                info[i].update(branch="deg>=3-zero", **{"lambda": lam, "column": j})
                continue
            H[:, i] *= acc[:, col_pos[j]] / np.sqrt(lam)
            info[i].update(branch="deg>=3", **{"lambda": lam, "column": j})

    if return_info:
        norms = np.linalg.norm(H, axis=0)   # vectorized: a per-column Python loop here costs
        for i in range(n_mono):             # real time on wide H, and this is only a diagnostic
            if info[i] is not None:
                info[i]["norm"] = float(norms[i])
        return H, info
    return H


