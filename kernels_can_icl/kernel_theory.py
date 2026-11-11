import numpy as np
import heapq
import math
from itertools import product, combinations_with_replacement, permutations
import functools
from scipy.optimize import linear_sum_assignment

from kernels_can_icl.utils import ensure_numpy
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

def get_moment_generator(betas_flat, v, eps):
    def mu(k): return float(np.mean(betas_flat**k))

    def dfact(n):
        if n <= 0: return 1
        res = 1
        for i in range(n, 0, -2): res *= i
        return res

    def mX(m, var): return dfact(m-1) * (var**(m/2)) if m % 2 == 0 else 0.0

    def poly_g(a, b):
        c = np.zeros(b + 1)
        for k in range(b + 1):
            if (a+k)%2 == 0 and (b-k)%2 == 0:
                c[k] = math.comb(b, k) * mX(a+k, v) * mX(b-k, eps)
        return c

    def pmul(p1, p2):
        if len(p1)==0: return p2
        if len(p2)==0: return p1
        r = np.zeros(len(p1)+len(p2)-1)
        for i, x in enumerate(p1):
            for j, y in enumerate(p2): r[i+j] += x*y
        return r

    def E(*token_features, q_power=0):
        if q_power % 2 != 0: return 0.0
        P = np.array([1.0])
        for a, b in token_features:
            if a==0 and b==0: continue
            P = pmul(P, poly_g(a, b))
        res = sum(c * mu(p) for p, c in enumerate(P) if c != 0)
        return res * mX(q_power, v)
    return E


@functools.lru_cache(maxsize=None)
def _mX(m, var):
    """E[Z^m] for Z ~ N(0, var) = (m-1)!! var^(m/2) for even m, else 0. Memoized."""
    if m % 2 != 0:
        return 0.0
    res = 1
    for i in range(m - 1, 0, -2):
        res *= i
    return res * (var ** (m // 2))


@functools.lru_cache(maxsize=None)
def _poly_g(a, b, v, eps):
    """Beta-polynomial coefficients of E[X^a y^b | beta] for one token; depends only on
    (a, b, v, eps) (not the task moments), so it memoizes heavily across matrix entries."""
    c = [0.0] * (b + 1)
    for k in range(b + 1):
        if (a + k) % 2 == 0 and (b - k) % 2 == 0:
            c[k] = math.comb(b, k) * _mX(a + k, v) * _mX(b - k, eps)
    return tuple(c)


@functools.lru_cache(maxsize=None)
def _make_fast_E(v, eps, moments):
    """Moment generator E[ prod_token X^a y^b * X_q^q_power ] for the shared-beta task model.

    E depends only on the *multiset* of per-token (a, b) pairs (tokens are exchangeable and beta is
    shared symmetrically), so results are memoized on the small canonical (features, q_power) key in
    a local dict. The long `moments` tuple stays out of that per-lookup key (it is captured here and
    folded into the @lru_cache key of this builder), which keeps the millions of matrix-fill lookups
    cheap. Returns a reusable E shared across all families of a given (v, eps, moments)."""
    cache = {}

    def mu(k):
        if k == 0:
            return 1.0
        return moments[k - 1] if 0 < k <= len(moments) else 0.0

    def E(*token_features, q_power=0):
        feats = tuple(sorted((a, b) for a, b in token_features if (a or b)))
        key = (feats, q_power)
        val = cache.get(key)
        if val is not None:
            return val
        if q_power % 2 != 0:
            cache[key] = 0.0
            return 0.0
        P = (1.0,)
        for a, b in feats:
            g = _poly_g(a, b, v, eps)
            r = [0.0] * (len(P) + len(g) - 1)
            for i, x in enumerate(P):
                if x == 0.0:
                    continue
                for j, y in enumerate(g):
                    if y != 0.0:
                        r[i + j] += x * y
            P = tuple(r)
        val = sum(c * mu(p) for p, c in enumerate(P) if c != 0.0) * _mX(q_power, v)
        cache[key] = val
        return val

    return E


@functools.lru_cache(maxsize=None)
def _get_lower_inverse(deg, L_seq, v, eps, moments):
    """Lower-degree (same-parity) sequence basis and pinv(C_LL) used by the Gram-Schmidt
    projection. Cached per degree so all families of a degree reuse the (expensive) C_LL."""
    E = _make_fast_E(v, eps, moments)
    lower_basis = []
    for d_low in range(deg - 2, -1, -2):
        lower_basis.extend(generate_full_sequence_basis(d_low, L_seq))
    size_L = len(lower_basis)
    if size_L == 0:
        return [], None
    C_LL = np.zeros((size_L, size_L))
    for r in range(size_L):
        for c in range(size_L):
            C_LL[r, c] = E(*combine_states(lower_basis[r], lower_basis[c]))
    return lower_basis, np.linalg.pinv(C_LL)


@functools.lru_cache(maxsize=None)
def get_cached_higher_degree_roots(deg, family_part, L_seq, v, eps, moments):
    """
    Runs the exact generalized Gram-Schmidt process, but only ONCE per
    partition, sequence length, and data eigenvalue.
    """
    E = _make_fast_E(v, eps, moments)

    basis_F = get_family_basis(family_part, L_seq)
    size_F = len(basis_F)

    C_FF = np.zeros((size_F, size_F))
    for r in range(size_F):
        for c in range(size_F):
            C_FF[r, c] = E(*combine_states(basis_F[r], basis_F[c]))

    lower_basis, C_LL_pinv = _get_lower_inverse(deg, L_seq, v, eps, moments)
    if lower_basis:
        size_L = len(lower_basis)
        C_FL = np.zeros((size_F, size_L))
        for r in range(size_F):
            for c in range(size_L):
                C_FL[r, c] = E(*combine_states(basis_F[r], lower_basis[c]))
        C_perp = C_FF - C_FL @ C_LL_pinv @ C_FL.T
    else:
        C_perp = C_FF

    # Apply weights and diagonalize.
    W = np.array([get_weight(s) for s in basis_F])
    C_weighted = np.outer(W, W) * C_perp
    eigvals, eigvecs = np.linalg.eigh(C_weighted)
    trace = float(np.trace(C_weighted))   # == sum of eigenvalues (total degree-spectral mass)

    # Precompute the bijective monomial -> eigenvalue map (symmetry-adapted assignment),
    # instead of pairing sorted eigenvalues with arbitrary generation-order indices.
    assignment = _build_higher_degree_assignment(eigvals, eigvecs, basis_F)
    return eigvals, eigvecs, basis_F, trace, assignment


def _state_key(state):
    """Hashable canonical key for a sequence-state dict {token: (x, y)}."""
    return tuple(sorted(state.items()))


def _orbit_key(state):
    """Permutation-orbit identifier for a sequence state: the multiset of per-token
    (X-power, y-power) pairs, independent of which tokens carry them. Two states are
    related by a token permutation iff they share this key."""
    return tuple(sorted(state.values()))


def _is_canonical_state(state):
    """True if `state` is the canonical (symmetric-mode) representative of its orbit:
    its active tokens are exactly the lowest positions {0, ..., k-1} and the per-token
    (x, y) states sit in descending order (largest degree, then largest X-power) on
    ascending tokens. This mirrors the deg 1-3 convention of putting the symmetric mode
    on token 0 (and tokens 0,1,... for multi-token families)."""
    toks = sorted(state.keys())
    k = len(toks)
    if toks != list(range(k)):
        return False
    states_in_token_order = [state[t] for t in toks]
    canonical_order = sorted(state.values(), key=lambda xy: (-(xy[0] + xy[1]), -xy[0]))
    return states_in_token_order == canonical_order


def _build_higher_degree_assignment(eigvals, eigvecs, basis_F):
    """Bijectively map every degree>=4 monomial (sequence state) to one eigenvalue of the
    weighted covariance, reproducing the deg 1-3 symmetric/antisymmetric convention.

    The symmetric ("important") modes span the per-type orbit-sum probes. We diagonalize the
    reduced symmetric matrix, match each token-permutation orbit to one symmetric eigenvalue
    (maximizing eigenvector overlap), and hand that eigenvalue to the orbit's canonical
    (lowest-position) representative. The remaining shifted monomials are matched bijectively
    to the leftover eigenvalues (the antisymmetric / standard-irrep copies), again by overlap.

    The map is a bijection over the family spectrum, so the assigned eigenvalues sum to
    trace(C_weighted) -- the per-degree spectral mass.

    Returns {_state_key(state): eigenvalue}.
    """
    sF = len(basis_F)
    Cw = (eigvecs * eigvals) @ eigvecs.T            # reconstruct C_weighted

    orbit_of = [_orbit_key(b) for b in basis_F]
    orbit_keys = list(dict.fromkeys(orbit_of))      # distinct orbits, stable order

    # Reduced symmetric matrix over the normalized orbit-sum probes (one per type pattern).
    probes = np.zeros((sF, len(orbit_keys)))
    for col, k in enumerate(orbit_keys):
        sel = np.array([1.0 if ok == k else 0.0 for ok in orbit_of])
        probes[:, col] = sel / np.linalg.norm(sel)
    S = probes.T @ Cw @ probes
    sev, svec = np.linalg.eigh(S)

    # Match each type pattern to a symmetric eigenvector (Hungarian, maximize |overlap|^2).
    ri, ci = linear_sum_assignment(-(svec ** 2))
    matched = sorted(zip(ri, ci), key=lambda tc: -sev[tc[1]])   # process strongest modes first

    assignment = {}
    used = set()
    canonical_eigval = {}
    for t, j_sym in matched:
        full_vec = probes @ svec[:, j_sym]          # lift symmetric mode to the full basis
        ov = np.abs(eigvecs.T @ full_vec)
        for j in np.argsort(-ov):                   # pick the closest unused full eigenvector
            if j not in used:
                used.add(int(j))
                canonical_eigval[orbit_keys[t]] = abs(eigvals[int(j)])
                break

    for i, b in enumerate(basis_F):
        if _is_canonical_state(b):
            assignment[_state_key(b)] = canonical_eigval[orbit_of[i]]

    # Remaining (shifted / antisymmetric) states -> leftover eigenvalues, bijectively.
    rem_rows = [i for i, b in enumerate(basis_F) if _state_key(b) not in assignment]
    rem_cols = [j for j in range(sF) if j not in used]
    if rem_rows:
        sub = np.abs(eigvecs[np.ix_(rem_rows, rem_cols)]) ** 2
        rr, cc = linear_sum_assignment(-sub)
        for a, c in zip(rr, cc):
            assignment[_state_key(basis_F[rem_rows[a]])] = abs(eigvals[rem_cols[c]])
    return assignment


def assign_higher_degree_eigval(context_state, assignment):
    """Look up the eigenvalue assigned to a degree>=4 monomial's sequence state."""
    return assignment.get(_state_key(context_state), 0.0)


def correction_terms(monomial, data_eigvals, betas, noise_var, context_length, dim):
    # betas_flat = np.array(betas).flatten()
    C = context_length

    def parse_idx(idx):
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

    total_deg = sum(monomial.values())

    partitions = {}
    for idx, exp in monomial.items():
        type_, t, d = parse_idx(idx)
        key = (d, type_ == 'q')
        if key not in partitions: partitions[key] = {}
        partitions[key][idx] = exp

    total_root = 1.0
    is_impt = "0"

    for (d, is_query), part in partitions.items():
        deg = sum(part.values())
        if deg == 0: continue
        beta_d = betas[:, d] 
        
        # Calculate moments ONLY over this dimension
        mu1 = float(np.mean(beta_d))
        mu2 = float(np.mean(beta_d**2))
        mu3 = float(np.mean(beta_d**3))
        mu4 = float(np.mean(beta_d**4))
        mu5 = float(np.mean(beta_d**5))
        mu6 = float(np.mean(beta_d**6))    
         
        v = data_eigvals[list(part.keys())[0]]
        eps = noise_var
        
        # Deg 1 stuff
        Sigma_A = np.array([
            [v,       mu1 * v], 
            [mu1 * v, mu2 * v + eps]
        ])
        roots_lin = sorted(np.linalg.eigvalsh(Sigma_A), reverse=True)
        
        # --- QUERY TOKEN BLOCK ---
        if is_query: 
            total_root *= v**deg
            if total_deg == 1: is_impt = f"q{d}"
            continue

        # --- DEGREE 1 ---
        if deg == 1:
            type_ = parse_idx(list(part.keys())[0])[0]
            total_root *= roots_lin[0] if type_ == 'X' else roots_lin[1]
            # if total_deg == 1: print(total_root, type_, d)

        # --- DEGREE 2 ---
        elif deg == 2:
            idx1, idx2 = list(part.keys())[0], list(part.keys())[-1]
            t1, t2 = parse_idx(idx1)[1], parse_idx(idx2)[1]
            d1, d2 = parse_idx(idx1)[2], parse_idx(idx2)[2]
            
            if t1 == t2: # Same Token (Full 3x3 Block)
                L_seq = context_length - 1
                
                c11 = 2 * v**2
                c22 = (3 * mu2 - mu1**2) * v**2 + v * eps
                c33 = (3 * mu4 - mu2**2) * v**2 + 4 * mu2 * v * eps + 2 * eps**2
                c12 = 2 * mu1 * v**2
                c13 = 2 * mu2 * v**2
                c23 = (3 * mu3 - mu1 * mu2) * v**2 + 2 * mu1 * v * eps
                
                D_full = np.array([
                    [c11 / 2,          c12 / np.sqrt(2), c13 / 2],
                    [c12 / np.sqrt(2), c22,              c23 / np.sqrt(2)],
                    [c13 / 2,          c23 / np.sqrt(2), c33 / 2]
                ])
                a = (mu2 - mu1**2) * v**2
                b = (mu3 - mu1 * mu2) * v**2 / np.sqrt(2)
                c = (mu4 - mu2**2) * v**2 / 2
                C_full = np.array([[0, 0, 0], [0, a, b], [0, b, c]])
                
                if t1 == 0: roots = sorted(np.abs(np.linalg.eigvalsh(D_full + (L_seq - 1) * C_full)), reverse=True)
                else: roots = sorted(np.abs(np.linalg.eigvalsh(D_full - C_full)), reverse=True)
                
                type_ = parse_idx(idx1)[0] if len(part)==1 else 'Xy'
                
                if type_ == 'y':
                    total_root *= roots[0]
                    if total_deg == 3 and (d, True) in partitions and t1==0:
                        is_impt = f"y2q{d}"
                elif type_ == 'Xy':
                    total_root *= roots[1]
                    if total_deg == 3 and (d, True) in partitions and t1==0:
                        is_impt = f"Xyq{d}"
                else:
                    total_root *= roots[2]
                    if total_deg == 3 and (d, True) in partitions and t1==0:
                        is_impt = f"X2q{d}"
                
            else: # Cross Token (4x4 Block)
                c11 = v**2
                
                # if d1 == d2:
                c12 = mu1 * v**2
                c14 = mu2 * v**2
                c22 = mu2 * v**2 + v * eps
                c23 = mu2 * v**2
                c24 = mu3 * v**2 + mu1 * v * eps
                c44 = mu4 * v**2 + 2 * mu2 * v * eps + eps**2
                    
                
                C4_cross = np.array([
                    [c11, c12, c12, c14],
                    [c12, c22, c23, c24],
                    [c12, c23, c22, c24],
                    [c14, c24, c24, c44]
                ])
                
                roots = sorted(np.abs(np.linalg.eigvalsh(C4_cross)), reverse=True)
                
                type1, type2 = parse_idx(idx1)[0], parse_idx(idx2)[0]
                types = {type1, type2}
                
                if types == {'y'}: total_root *= roots[0]
                elif types == {'X', 'y'}:
                    if (t1 < t2 and type1 == 'y') or (t2 < t1 and type2 == 'y'):
                        total_root *= roots[1]
                    else: total_root *= roots[2]
                else: total_root *= roots[3]

        # --- DEGREE 3 ---
        elif deg == 3:
            E = get_moment_generator(beta_d, v, eps)
            
            token_counts = {}
            for idx, exp in part.items():
                _, t, _ = parse_idx(idx)
                token_counts[t] = token_counts.get(t, 0) + exp
            
            # FAMILY 1: 1-Token Cubics
            if len(token_counts) == 1:
                F4 = [(3,0), (2,1), (1,2), (0,3)]
                L2 = [(1,0), (0,1)]
                C_FF = np.zeros((4, 4))
                C_FL = np.zeros((4, 2))
                C_LL = np.zeros((2, 2))
                
                for r in range(4):
                    for c in range(4): C_FF[r, c] = E((F4[r][0]+F4[c][0], F4[r][1]+F4[c][1]))
                    for c in range(2): C_FL[r, c] = E((F4[r][0]+L2[c][0], F4[r][1]+L2[c][1]))
                for r in range(2):
                    for c in range(2): C_LL[r, c] = E((L2[r][0]+L2[c][0], L2[r][1]+L2[c][1]))
                        
                C_perp = C_FF - C_FL @ np.linalg.inv(C_LL) @ C_FL.T
                W_denom = np.array([1/6, 1/2, 1/2, 1/6])
                W_root = np.sqrt(W_denom)
                C_perp_weighted = np.outer(W_root, W_root) * C_perp
                
                roots_4 = sorted(np.linalg.eigvalsh(C_perp_weighted), reverse=True)
                x_count = [parse_idx(idx)[0] for idx in part.keys()].count('X')
                total_root *= roots_4[3 - x_count]
                
            # FAMILY 4: Context Passenger (1 Linear, 1 Quadratic distinct token)
            elif len(token_counts) == 2:
                t_keys = list(token_counts.keys())
                t_lin = t_keys[0] if token_counts[t_keys[0]] == 1 else t_keys[1]
                t_quad = t_keys[1] if token_counts[t_keys[0]] == 1 else t_keys[0]
                
                F6 = [((1,0), (2,0)), ((1,0), (1,1)), ((1,0), (0,2)),
                      ((0,1), (2,0)), ((0,1), (1,1)), ((0,1), (0,2))]
                L2 = [(1,0), (0,1)]
                
                C_FF_D = np.zeros((6, 6))
                C_FF_C = np.zeros((6, 6))
                C_FL = np.zeros((6, 2))
                C_LL = np.zeros((2, 2))
                
                for r in range(6):
                    for c in range(6):
                        k_pow = (F6[r][0][0]+F6[c][0][0], F6[r][0][1]+F6[c][0][1])
                        i_pow = (F6[r][1][0]+F6[c][1][0], F6[r][1][1]+F6[c][1][1])
                        C_FF_D[r, c] = E(k_pow, i_pow)
                        
                        i_pow_cross = F6[r][1]
                        j_pow_cross = F6[c][1]
                        C_FF_C[r, c] = E(k_pow, i_pow_cross, j_pow_cross)
                        
                    for c in range(2):
                        k_pow = (F6[r][0][0]+L2[c][0], F6[r][0][1]+L2[c][1])
                        i_pow = F6[r][1]
                        C_FL[r, c] = E(k_pow, i_pow)
                        
                for r in range(2):
                    for c in range(2):
                        k_pow = (L2[r][0]+L2[c][0], L2[r][1]+L2[c][1])
                        C_LL[r, c] = E(k_pow)
                        
                D_6 = C_FF_D - C_FL @ np.linalg.inv(C_LL) @ C_FL.T
                C_6 = C_FF_C - C_FL @ np.linalg.inv(C_LL) @ C_FL.T
                
                W_denom = np.array([1/2, 1.0, 1/2, 1/2, 1.0, 1/2])
                W_root = np.sqrt(W_denom)
                D_6 = np.outer(W_root, W_root) * D_6
                C_6 = np.outer(W_root, W_root) * C_6
                
                L_seq = context_length - 1
                first_available_quad = 0 if t_lin != 0 else 1
                
                if t_quad == first_available_quad:
                    roots_6 = sorted(np.linalg.eigvalsh(D_6 + (L_seq - 2) * C_6), reverse=True)
                else:
                    roots_6 = sorted(np.linalg.eigvalsh(D_6 - C_6), reverse=True)
                    
                lin_key = [k for k in part.keys() if parse_idx(k)[1] == t_lin][0]
                l_type = parse_idx(lin_key)[0]
                
                quad_keys = [k for k in part.keys() if parse_idx(k)[1] == t_quad]
                q_type = parse_idx(quad_keys[0])[0] if len(quad_keys) == 1 else 'Xy'
                
                state_idx = 0
                if l_type == 'y': state_idx += 3
                if q_type == 'Xy': state_idx += 1
                elif q_type == 'y': state_idx += 2
                
                total_root *= roots_6[5 - state_idx]

            # FAMILY 5: 3 Distinct Tokens
            elif len(token_counts) == 3:
                L_modes = [(1,0), (0,1)]
                F8 = [(l1, l2, l3) for l1 in L_modes for l2 in L_modes for l3 in L_modes]
                C8 = np.zeros((8, 8))
                
                for r in range(8):
                    for c in range(8):
                        t1 = (F8[r][0][0] + F8[c][0][0], F8[r][0][1] + F8[c][0][1])
                        t2 = (F8[r][1][0] + F8[c][1][0], F8[r][1][1] + F8[c][1][1])
                        t3 = (F8[r][2][0] + F8[c][2][0], F8[r][2][1] + F8[c][2][1])
                        C8[r, c] = E(t1, t2, t3)
                        
                roots_8 = sorted(np.linalg.eigvalsh(C8), reverse=True)
                
                t_ids = sorted(list(token_counts.keys()))
                type_a = parse_idx([k for k in part.keys() if parse_idx(k)[1] == t_ids[0]][0])[0]
                type_b = parse_idx([k for k in part.keys() if parse_idx(k)[1] == t_ids[1]][0])[0]
                type_c = parse_idx([k for k in part.keys() if parse_idx(k)[1] == t_ids[2]][0])[0]
                
                state_idx = 0
                if type_a == 'y': state_idx += 4
                if type_b == 'y': state_idx += 2
                if type_c == 'y': state_idx += 1
                
                total_root *= roots_8[7 - state_idx]

        elif deg >= 4:
            L_seq = context_length - 1
            
            # 1. Parse the empirical monomial into a Sequence State
            context_state = {}
            for idx, exp in part.items():
                type_, t, _ = parse_idx(idx)
                x, y = context_state.get(t, (0, 0))
                if type_ == 'X': x += exp
                else: y += exp
                context_state[t] = (x, y)
                
            # 2. Identify the Family Partition
            family_part = tuple(sorted([x + y for x, y in context_state.values() if (x + y) > 0], reverse=True))

            # 3. Extract the moments needed by the moment generator and query the cache.
            #    A covariance entry between two degree-`deg` monomials has total degree 2*deg,
            #    so we need moments mu_k up to 2*deg; anything less is silently zeroed inside
            #    the moment generator and corrupts the spectrum. The moments tuple is the
            #    @lru_cache key, so this keeps caching exact per (partition, L_seq, v, eps, distr).
            moments = tuple(float(np.mean(beta_d**k)) for k in range(1, 2 * deg + 1))

            _eigvals, _eigvecs, _basis_F, _trace, assignment = get_cached_higher_degree_roots(
                deg, family_part, L_seq, v, eps, moments)

            # 4. Symmetry-adapted assignment: map this monomial to the eigenvalue of the
            #    eigenfunction it most overlaps (symmetric mode for the canonical/lowest-position
            #    representative, antisymmetric otherwise), reproducing the deg 1-3 convention.
            total_root *= assign_higher_degree_eigval(context_state, assignment)

            # --- Importance flags for the odd-degree ICL terms (even context + query) ---
            # Only the canonical representative carries the importance flag (its symmetric mode
            # holds the eigencoefficient); shifted copies are antisymmetric with zero overlap.
            if total_deg in (5, 7) and (d, True) in partitions:
                # degree-5 terms (context degree 4)
                if context_state == {0: (0, 4)}:            # sum_i He_4(y_i) * X_q
                    is_impt = f"y4q{d}"
                elif context_state == {0: (1, 3)}:          # sum_i He_3(y_i) X_i * X_q
                    is_impt = f"y3Xq{d}"
                elif context_state == {0: (1, 1), 1: (1, 1)}:  # sum_{i!=j} y_i y_j X_i X_j * X_q
                    is_impt = f"X2y2q{d}"
                elif context_state == {0: (2, 0), 1: (1, 1)}:  # sum_{i!=j} y_i X_i X_j^2 * X_q
                    is_impt = f"yX3q{d}"
                # degree-7 terms (context degree 6)
                elif context_state == {0: (0, 6)}:          # sum_i He_6(y_i) * X_q
                    is_impt = f"y6q{d}"
                elif context_state == {0: (1, 5)}:          # sum_i He_5(y_i) X_i * X_q
                    is_impt = f"y5Xq{d}"
                elif context_state == {0: (1, 3), 1: (1, 1)}:  # sum_{i!=j} y_i^3 y_j X_i X_j * X_q
                    is_impt = f"y3yX2q{d}"
                elif context_state == {0: (2, 0), 1: (2, 0), 2: (1, 1)}:  # sum_{i!=j!=k} y_i X_i X_j^2 X_k^2 * X_q
                    is_impt = f"X5yq{d}"

    return total_root * math.factorial(total_deg), is_impt

def compute_kicl_eigval(data_eigvals, monomial, eval_level_coeff, context_length, dim, betas, noise_var=0.0):
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
        coupled_root, is_impt = correction_terms(monomial, data_eigvals, ensure_numpy(betas), noise_var, context_length, dim)
        return kicl_eigval * coupled_root, is_impt
    #fallback
    return kicl_eigval, is_impt


def _important_query_monomials(context_length, dim):
    """The odd-degree ICL terms (canonical, dim-0 representatives) that carry an eigencoefficient.

    Degree 5 (context degree 4):
      y4q   = sum_i He_4(y_i) X_q                      {y0^4, Xq}
      y3Xq  = sum_i He_3(y_i) X_i X_q                  {X0 y0^3, Xq}
      X2y2q = sum_{i!=j} y_i y_j X_i X_j X_q           {X0 y0, X1 y1, Xq}
      yX3q  = sum_{i!=j} y_i X_i X_j^2 X_q             {X0^2, X1 y1, Xq}
    Degree 7 (context degree 6):
      y6q    = sum_i He_6(y_i) X_q                     {y0^6, Xq}
      y5Xq   = sum_i He_5(y_i) X_i X_q                 {X0 y0^5, Xq}
      y3yX2q = sum_{i!=j} y_i^3 y_j X_i X_j X_q        {X0 y0^3, X1 y1, Xq}
      X5yq   = sum_{i!=j!=k} y_i X_i X_j^2 X_k^2 X_q   {X0^2, X1^2, X2 y2, Xq}
    Returns a list of Monomials."""
    def iX(t): return t * dim                       # dim-0 X of token t
    def iy(t): return (context_length - 1) * dim + t  # y of token t
    iq0 = (context_length - 1) * dim + (context_length - 1)  # dim-0 query feature
    return [
        # degree 5
        Monomial({iy(0): 4, iq0: 1}),
        Monomial({iX(0): 1, iy(0): 3, iq0: 1}),
        Monomial({iX(0): 1, iy(0): 1, iX(1): 1, iy(1): 1, iq0: 1}),
        Monomial({iX(0): 2, iX(1): 1, iy(1): 1, iq0: 1}),
        # degree 7
        Monomial({iy(0): 6, iq0: 1}),
        Monomial({iX(0): 1, iy(0): 5, iq0: 1}),
        Monomial({iX(0): 1, iy(0): 3, iX(1): 1, iy(1): 1, iq0: 1}),
        Monomial({iX(0): 2, iX(1): 2, iX(2): 1, iy(2): 1, iq0: 1}),
    ]


def compute_degree_masses(data_eigvals, eval_level_coeff, context_length, dim, betas,
                          noise_var, kmax, min_degree=4):
    """Total spectral mass (sum of all eigenvalues) at each degree from `min_degree` to `kmax`,
    obtained from family *traces* instead of enumerating every monomial.

    A degree-l eigenvalue factorizes as level_coeff(l) * l! * (context factor) * (query factor),
    and the query / context groups are independent, so the per-degree mass is a convolution of a
    per-degree context-mass sequence with the query sequence. The context mass at degree c is the
    sum over families of trace(C_weighted) -- no per-monomial work. For c <= 3 we instead sum the
    hand-path factors directly (cheap, and keeps the mass consistent with the explicitly enumerated
    low-degree eigenvalues, whose single-token-cubic assignment does not sum to its true trace).

    Exact for dim == 1; for dim > 1 it is an estimate (single context slot + dim-0 query).
    Returns {degree: mass}.
    """
    data_eigvals = ensure_numpy(data_eigvals)
    betas_np = ensure_numpy(betas)
    L = context_length - 1
    eps = noise_var
    v = float(data_eigvals[0])          # X context variance (dim 0)
    v_q = float(data_eigvals[-1])       # query variance
    beta_d = betas_np[:, 0]

    ctx = {0: 1.0}
    # c <= 3: enumerate context monomials (matches the hand-path per-monomial values exactly)
    ctx_feats = list(range(L * dim + L))           # X_ctx and y_ctx feature indices (no query)
    for c in range(1, min(3, kmax) + 1):
        tot = 0.0
        for combo in combinations_with_replacement(ctx_feats, c):
            m = {}
            for f in combo:
                m[f] = m.get(f, 0) + 1
            r, _ = correction_terms(Monomial(m), data_eigvals, betas_np, noise_var, context_length, dim)
            tot += abs(r) / math.factorial(c)      # strip l!, re-applied at total degree
        ctx[c] = tot
    # c >= 4: family traces (no enumeration)
    for c in range(4, kmax + 1):
        moments = tuple(float(np.mean(beta_d**k)) for k in range(1, 2 * c + 1))
        tot = 0.0
        for family in get_partitions(c):
            *_, trace, _ = get_cached_higher_degree_roots(c, family, L, v, eps, moments)
            tot += trace
        ctx[c] = tot

    masses = {}
    for ell in range(max(min_degree, 1), kmax + 1):
        s = sum(ctx[ell - qd] * (v_q ** qd) for qd in range(0, ell + 1))
        masses[ell] = float(eval_level_coeff(ell)) * math.factorial(ell) * s
    return masses


def generate_kicl_monomials(data_eigvals, num_monomials, eval_level_coeff, context_length=10, dim=1, betas=None, noise_var=0.0, kmax=10, aggregate_high_degree=False):
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
    
    first_kicl_eigval, is_impt = compute_kicl_eigval(data_eigvals, Monomial({}), eval_level_coeff, context_length, dim, betas=betas, noise_var=noise_var)
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
            kicl_eigval, is_impt = compute_kicl_eigval(data_eigvals, monomial, eval_level_coeff, context_length, dim, betas=betas, noise_var=noise_var)
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
                ev, ii = compute_kicl_eigval(data_eigvals, nm, eval_level_coeff, context_length, dim, betas=betas, noise_var=noise_var)
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
            ev, ii = compute_kicl_eigval(data_eigvals, nm, eval_level_coeff, context_length, dim, betas=betas, noise_var=noise_var)
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

