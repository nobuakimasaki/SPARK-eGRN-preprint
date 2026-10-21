import numpy as np
import math
from numba import njit, prange
from scipy.special import expit, logit, gammaln

def normal_logpdf(x, mean=0.0, sd=1.0):
    """
    Log density of N(mean, sd^2) at x, INCLUDING the normalizing constant
    -0.5*log(2*pi) - log(sd).  The constant does not cancel in
    dimension-changing (birth/death) acceptance ratios or in cross-chain
    swaps, where the number of active weights differs between the two
    states being compared, so it must be kept.
    Vectorized: x may be a scalar or ndarray.
    """
    var = sd * sd
    return (-0.5 * ((x - mean) ** 2) / var
            - 0.5 * math.log(2.0 * math.pi) - np.log(sd))

def gamma_logpdf(x, shape, rate):
    """
    Log PDF of Gamma(shape=a, rate=b) at x.
    Support: x > 0
    """
    a = shape
    b = rate
    return (
        a * math.log(b)
        - math.lgamma(a)
        + (a - 1.0) * np.log(x)
        - b * x
    )

def beta_logpdf(x, a, b):
    """Beta log-pdf. Any of x, a, b may be scalar or array-like (broadcast)."""
    x = np.asarray(x, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return (gammaln(a + b) - gammaln(a) - gammaln(b)
            + (a - 1.0) * np.log(x) + (b - 1.0) * np.log1p(-x))

def make_Q_partitioned(alpha, rho_logits, src_idx, *, diag_zero=True):
    """
    Construct Q where each directed edge (i,j) belongs to a source partition.

    Model (direct per-source edge probabilities):
        src_idx[i,j] = -1  => q_ij = 0                       (forbidden)
        src_idx[i,j] =  0  => q_ij = expit(alpha) = rho_0    (baseline)
        src_idx[i,j] =  k  => q_ij = expit(rho_logits[k-1])  = rho_k  (source k)

    Parameters
    ----------
    alpha : float
        logit(rho_0); the baseline edge log-odds.
    rho_logits : array-like, shape (S,)
        logit(rho_k) for k = 1..S (one entry per evidence source).
    src_idx : array-like, shape (n,n), integer
        Source-ID matrix. -1 = forbidden, 0 = baseline, k>=1 = evidence source k.
    diag_zero : bool
        If True, force Q[i,i]=0.

    Returns
    -------
    Q : ndarray, shape (n,n)
        Edge inclusion probabilities in [0,1], with optional zero diagonal.
    """
    src_idx = np.asarray(src_idx)
    if src_idx.ndim != 2 or src_idx.shape[0] != src_idx.shape[1]:
        raise ValueError(f"src_idx must be square (n,n). Got {src_idx.shape}")
    n = src_idx.shape[0]

    rho_logits = np.asarray(rho_logits, dtype=np.float64).ravel()
    S = rho_logits.size

    if not np.issubdtype(src_idx.dtype, np.integer):
        raise ValueError("src_idx must be integer array.")

    mn = int(src_idx.min())
    mx = int(src_idx.max())
    if mn < -1:
        raise ValueError(f"src_idx contains values < -1 (min={mn}).")
    if mx > S:
        raise ValueError(f"src_idx contains ID {mx} but rho_logits has length {S}.")

    # lookup table on logit scale: lookup[0]=alpha, lookup[k]=rho_logits[k-1]
    logit_lookup = np.empty(S + 1, dtype=np.float64)
    logit_lookup[0] = float(alpha)
    logit_lookup[1:] = rho_logits

    Q = np.zeros((n, n), dtype=np.float64)

    allowed = src_idx >= 0
    if np.any(allowed):
        Q[allowed] = expit(logit_lookup[src_idx[allowed]])

    if diag_zero:
        np.fill_diagonal(Q, 0.0)

    return Q

def log_prior_alpha(alpha, a, b):
    """
    Prior: rho0 ~ Beta(a, b).  Stored parameter: alpha = logit(rho0).

    Returns log p(rho0) = Beta(rho0; a, b) evaluated in the natural rho
    parameterization (rho0 = expit(alpha)), with NO logit Jacobian.  The sampler
    works in the rho parameterization of Supplement Section 3 (the state is
    Phi_A = {rho_0, ..., rho_K}, not their logits): the graph hyperparameters are
    tempered and Gibbs-resampled directly on the rho scale (Section 3.4), so the
    log-prior that feeds the untempered log-posterior must be the rho-scale Beta
    density.  alpha is only a storage transform of rho0; the change of variables
    leaves the rho-marginal identical, so dropping the Jacobian does not change
    the cold-chain target, only the (rho-space) tempered targets of the warm
    chains and the absolute value of the log-posterior trace.
    """
    rho0 = float(expit(alpha))
    return float(beta_logpdf(rho0, a, b))

def log_prior_hyper(alpha, rho_logits, prior_hyper):
    """
    Log joint prior on the graph hyperparameters (alpha, rho_logits).

    The baseline edge probability carries a Beta prior:

        rho_0  = expit(alpha)            ~ Beta(a, b)

    Each evidence-supported edge probability rho_k (k = 1..S) carries its own
    fixed Beta prior with user-specified shape parameters:

        rho_k  = expit(rho_logits[k-1])  ~ Beta(a_k, b_k)

    where (a_k, b_k) are length-S vectors supplied in prior_hyper["rho_k"].
    There is no hierarchical layer: mu_k and tau_k no longer exist.

    Returned log-density is the Beta prior evaluated in the natural rho
    parameterization (rho_0 = expit(alpha), rho_k = expit(rho_logits[k-1])),
    with NO logit Jacobians: the sampler tempers and Gibbs-updates rho on its
    own scale (Supplement Section 3.4), so this rho-scale density is the one
    that enters the untempered log-posterior. alpha/rho_logits are only a
    storage transform of rho.

    prior_hyper layout:
        prior_hyper["rho"]   = {"a": float, "b": float}            # rho_0
        prior_hyper["rho_k"] = {"a": (S,) array, "b": (S,) array}  # rho_1..rho_S
    """
    a = prior_hyper["rho"]["a"]
    b = prior_hyper["rho"]["b"]

    lp = log_prior_alpha(alpha, a, b)   # rho_0 ~ Beta(a, b)

    rho_logits = np.asarray(rho_logits, dtype=np.float64).ravel()
    S = rho_logits.size
    if S == 0:
        return float(lp)

    a_k = np.asarray(prior_hyper["rho_k"]["a"], dtype=np.float64).ravel()
    b_k = np.asarray(prior_hyper["rho_k"]["b"], dtype=np.float64).ravel()
    if a_k.size != S or b_k.size != S:
        raise ValueError(
            f"prior_hyper['rho_k']['a'] and ['b'] must have length S={S}; "
            f"got {a_k.size} and {b_k.size}."
        )

    rho = expit(rho_logits)

    # rho_k ~ Beta(a_k, b_k), evaluated on the rho scale (no logit Jacobian).
    lp_rho = beta_logpdf(rho, a_k, b_k).sum()

    lp += float(lp_rho)
    return float(lp)

def log_prior_graph_and_hyper(A, alpha, rho_logits,
                              src_idx, prior_hyper, *, diag_zero=True):
    """
    Log joint prior  log p(A, alpha, rho_logits).

    Factorises as:
        log p(A | alpha, rho_logits, src_idx)
        + log p(alpha, rho_logits)

    The graph prior is an independent Bernoulli model over allowed edges:
        A[i,j] | Q[i,j] ~ Bernoulli(Q[i,j])
    where Q is constructed via make_Q_partitioned and
        q_ij = rho_{src_idx[i,j]} for src_idx >= 0,  q_ij = 0 for forbidden.

    Forbidden edges (src_idx == -1) must be absent; if any are present returns -inf.

    Parameters
    ----------
    A : np.ndarray, shape (n, n)
        Binary adjacency matrix of the DAG.
    alpha : float
        logit(rho_0); logit-scale baseline edge probability.
    rho_logits : array-like, shape (S,)
        logit(rho_k) for k = 1..S.
    src_idx : np.ndarray, shape (n, n), dtype int
        Source-index matrix: -1=forbidden, 0=baseline, s>=1=evidence source s.
    prior_hyper : dict
        Must contain "rho": {"a","b"} (Beta prior on rho_0) and
        "rho_k": {"a","b"} with length-S shape vectors (Beta prior on rho_k).
    diag_zero : bool
        If True, enforce zero diagonal on Q (default True).

    Returns
    -------
    float
        log p(A, alpha, rho_logits), or -inf if the graph violates hard
        constraints.
    """
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square (n,n), got shape {A.shape}")
    n = A.shape[0]

    if np.any(np.diag(A) != 0):
        raise ValueError("A must have zeros on the diagonal (no self-edges).")

    src_idx = np.asarray(src_idx)
    if src_idx.shape != A.shape:
        raise ValueError(f"src_idx shape {src_idx.shape} does not match A shape {A.shape}")

    lp_params = log_prior_hyper(alpha, rho_logits, prior_hyper)
    if not np.isfinite(lp_params):
        return -np.inf

    forbidden = (src_idx == -1)
    if np.any(A[forbidden] != 0):
        return -np.inf

    Q = make_Q_partitioned(alpha, rho_logits, src_idx, diag_zero=diag_zero)
    if diag_zero and np.any(np.diag(Q) != 0):
        raise ValueError("Q must have zeros on the diagonal (no self-edges).")

    offdiag = ~np.eye(n, dtype=bool)
    allowed = offdiag & (src_idx >= 0)

    Qm = Q[allowed]
    Am = A[allowed].astype(bool)

    logQm   = np.log(Qm)
    log1mQm = np.log1p(-Qm)

    lp_A = logQm[Am].sum() + log1mQm[~Am].sum()

    return float(lp_A + lp_params)

def log_prior_theta(b, w, log_psi, A, hyper):
    """
    Log p(theta | A) for:
      b_j    ~ Normal(0, sd_b^2)
      w_ij   ~ Normal(0, sd_w^2)   only where A[i, j] != 0
      psi_j  ~ Gamma(a_psi, b_psi),  psi_j = exp(log_psi_j) with Jacobian

    IMPORTANT: w[i, j] may exist even when A[i, j] == 0;
               we do NOT evaluate the prior there.
    """
    b = np.asarray(b)
    w = np.asarray(w)
    log_psi = np.asarray(log_psi)

    sd_b = hyper["b"]["sd"]
    sd_w = hyper["w"]["sd"]
    a_psi = hyper["psi"]["a"]
    b_psi = hyper["psi"]["b"]

    total_lp = 0.0

    # ---- intercepts
    total_lp += normal_logpdf(b, mean=0, sd=sd_b).sum()

    # ---- dispersion (psi = exp(log_psi)) + Jacobian
    psi = np.exp(log_psi)
    total_lp += gamma_logpdf(psi, shape=a_psi, rate=b_psi).sum()
    total_lp += log_psi.sum()

    # ---- weights: ONLY where A != 0
    try:
        # scipy sparse
        A_coo = A.tocoo()
        ii, jj = A_coo.row, A_coo.col
    except AttributeError:
        # dense
        ii, jj = np.nonzero(A)

    total_lp += normal_logpdf(w[ii, jj], mean=0, sd=sd_w).sum()

    return float(total_lp)

@njit(cache=False)
def nb_loglik_2d(Y, ETA, psi_vec):
    """
    Y:   (N, K) float64 (or int64 cast to float64)
    ETA: (N, K) float64
    psi_vec: (K,) float64

    returns ll_per_node: (K,) float64
    """
    eta_min = math.log(1e-12)
    N, K = Y.shape
    out = np.zeros(K, dtype=np.float64)

    for j in range(K):
        psi = psi_vec[j]
        ll = 0.0

        if psi <= 1e-12:
            # Poisson limit
            for t in range(N):
                eta = ETA[t, j]
                if eta < eta_min:
                    eta = eta_min
                mu = math.exp(eta)
                y = Y[t, j]
                ll += y * eta - mu - math.lgamma(y + 1.0)
            out[j] = ll
            continue

        r = 1.0 / psi
        logpsi = math.log(psi)
        lgamma_r = math.lgamma(r)

        for t in range(N):
            eta = ETA[t, j]
            if eta < eta_min:
                eta = eta_min
            mu = math.exp(eta)

            log1p_psimu = math.log1p(psi * mu)

            y = Y[t, j]

            ll += (
                math.lgamma(y + r)
                - lgamma_r
                - math.lgamma(y + 1.0)
                - r * log1p_psimu
                + y * (logpsi + eta - log1p_psimu)
            )

        out[j] = ll

    return out

@njit(cache=False)
def nb_loglik_node(y, eta, psi):
    """
    Negative-binomial log-likelihood for a SINGLE node's N observations.

    y   : (N,) float64 counts for the node
    eta : (N,) float64 linear predictor for the node (X @ (W[:,j]*A[:,j]) + b_j)
    psi : float64 dispersion for the node

    Returns the scalar log-likelihood. The arithmetic is identical to column j
    of nb_loglik_2d, so a full sweep of per-node deltas reproduces the joint
    log-likelihood exactly (up to floating-point rounding).
    """
    eta_min = math.log(1e-12)
    N = y.shape[0]
    ll = 0.0

    if psi <= 1e-12:
        # Poisson limit
        for t in range(N):
            e = eta[t]
            if e < eta_min:
                e = eta_min
            mu = math.exp(e)
            ll += y[t] * e - mu - math.lgamma(y[t] + 1.0)
        return ll

    r = 1.0 / psi
    logpsi = math.log(psi)
    lgamma_r = math.lgamma(r)

    for t in range(N):
        e = eta[t]
        if e < eta_min:
            e = eta_min
        mu = math.exp(e)
        log1p_psimu = math.log1p(psi * mu)
        yt = y[t]
        ll += (
            math.lgamma(yt + r)
            - lgamma_r
            - math.lgamma(yt + 1.0)
            - r * log1p_psimu
            + yt * (logpsi + e - log1p_psimu)
        )
    return ll

def loglik_nb_adjmat(df, A, b, W, log_psi):
    """
    df: (N, K) ndarray
    A:  (K, K) adjacency mask (0/1) where A[i,j]=1 if i->j
    b:  (K,) intercepts
    W:  (K, K) weights where W[i,j] is i->j
    log_psi: (K,) log-dispersion

    Returns per-node ll and total.
    """
    X = np.asarray(df, dtype=np.float64)
    Y = X  
    A = np.asarray(A, dtype=np.float64)   # mask
    b = np.asarray(b, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    psi_vec = np.exp(np.asarray(log_psi, dtype=np.float64))

    # BLAS-accelerated eta computation
    B = W * A
    ETA = X @ B
    ETA += b  # broadcast (N,K) += (K,)

    ll_per_node = nb_loglik_2d(Y, ETA, psi_vec)
    total = float(ll_per_node.sum())
    return ll_per_node, total

def log_posterior(df, A, b, W, log_psi,
                  alpha, rho_logits, src_idx,
                  hyper, prior_hyper,
                  return_parts=True):
    """
    Compute posterior log-density:

      log p(A, theta, alpha, rho_logits | Y)
        = log p(Y | A, theta)
        + log p(theta | A)
        + log p(A | alpha, rho_logits, src_idx)
        + log p(alpha, rho_logits)

    IMPORTANT: does not enforce DAG constraint. Enforce acyclicity in proposals/checks.
    """
    ll_per_node, ll = loglik_nb_adjmat(df, A, b, W, log_psi)

    lp_theta = log_prior_theta(b, W, log_psi, A, hyper)

    lp_G_and_hyper = log_prior_graph_and_hyper(
        A, alpha, rho_logits, src_idx, prior_hyper
    )

    post = ll + lp_theta + lp_G_and_hyper

    if return_parts:
        return {
            "logpost": float(post),
            "loglik": float(ll),
            "logprior_theta": float(lp_theta),
            "logprior_graph_plus_hyper": float(lp_G_and_hyper),
            "ll_per_node": ll_per_node,
        }
    return float(post)