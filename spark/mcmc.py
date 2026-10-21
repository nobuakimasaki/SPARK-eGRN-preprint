"""
mcmc.py — Parallel Tempering MCMC for Bayesian DAG inference.

High-level overview
-------------------
This module implements replica-exchange (parallel tempering) MCMC to sample
from the joint posterior over DAG structures G and parameters theta = (b, W, psi)
given observations X.

Multiple chains run at different inverse temperatures beta_r = factor^(-r).
Chain 0 (beta=1) targets the true posterior; hotter chains (beta<1) explore
more freely and periodically swap states with cooler neighbours to improve mixing.

Each iteration is either:
  - A swap step  : propose exchanging the full state of two adjacent chains.
  - A local step : for every chain independently apply
      (1) a graph move       (birth / death / swap of an edge),
      (2) a theta update      (per-node Metropolis-within-Gibbs on b, W, log_psi),
      (3) an edge-prob update (Gibbs resample of rho_0..rho_K from their
                               tempered Beta full conditionals; Supplement 3.4).

Entry point
-----------
Run directly as a script to fit the model on data provided via CLI flags
(--x_path, --q_path, --out_path, --seed).  Results are serialised to a
pickle file containing posterior samples and diagnostics.
"""

import argparse
import math
import numpy as np
import pandas as pd
from copy import deepcopy
from dataclasses import dataclass, field
from dist import log_posterior, log_prior_graph_and_hyper, log_prior_hyper, nb_loglik_node
from graph_fast import propose_graph_move_with_probs, has_cycle
from tqdm.auto import trange
import pickle
import os
import sys
import time
from scipy.special import logit
from concurrent.futures import ProcessPoolExecutor


@dataclass
class ChainState:
    """Full state of one parallel-tempering chain."""
    A:              np.ndarray
    b:              np.ndarray
    W:              np.ndarray
    log_psi:        np.ndarray
    alpha:          float        # logit(rho_0)
    rho_logits:     np.ndarray   # logit(rho_k) for k = 1..S
    lp_graph_hyper: float
    logpi:          float


@dataclass
class SampleStore:
    """
    Accumulates posterior samples from the cold chain (chain 0).

    Graphs and edge weights are stored sparsely: graphs[k] is an (E_k, 2)
    int16 array of (parent, child) edges, and W[k] is the matching (E_k,)
    float64 array of weights on those edges.  Off-edge entries of state.W are
    not kept — they never enter the likelihood, which uses W * A.  Reconstruct
    dense matrices with graph.edges_to_adj(edges, p[, values]).
    """
    graphs:     list = field(default_factory=list)
    b:          list = field(default_factory=list)
    W:          list = field(default_factory=list)
    log_psi:    list = field(default_factory=list)
    alpha:      list = field(default_factory=list)
    rho_logits: list = field(default_factory=list)

    def record(self, state: "ChainState"):
        edges = np.argwhere(state.A != 0).astype(np.int16)
        self.graphs.append(edges)
        self.b.append(deepcopy(state.b))
        self.W.append(state.W[edges[:, 0], edges[:, 1]].astype(np.float64))
        self.log_psi.append(deepcopy(state.log_psi))
        self.alpha.append(float(state.alpha))
        self.rho_logits.append(deepcopy(state.rho_logits))


@dataclass
class DiagAccumulator:
    """Accumulates per-chain and swap acceptance statistics during MCMC."""
    n_chains: int
    k_list:   np.ndarray
    betas:    np.ndarray

    acc_theta:       np.ndarray = field(init=False)
    att_theta:       np.ndarray = field(init=False)
    acc_graph:       np.ndarray = field(init=False)
    att_graph:       np.ndarray = field(init=False)
    acc_swap_pair: np.ndarray = field(init=False)
    att_swap_pair: np.ndarray = field(init=False)

    def __post_init__(self):
        n = self.n_chains
        self.acc_theta    = np.zeros(n,     dtype=int)
        self.att_theta    = np.zeros(n,     dtype=int)
        self.acc_graph    = np.zeros(n,     dtype=int)
        self.att_graph    = np.zeros(n,     dtype=int)
        self.acc_swap_pair = np.zeros(n - 1, dtype=int)
        self.att_swap_pair = np.zeros(n - 1, dtype=int)

    def record_local(self, r, acc_g, att_g, acc_t, att_t):
        self.acc_graph[r] += acc_g;  self.att_graph[r] += att_g
        self.acc_theta[r] += acc_t;  self.att_theta[r] += att_t

    def record_swap(self, i, accepted):
        self.att_swap_pair[i] += 1
        if accepted:
            self.acc_swap_pair[i] += 1

    def build(self) -> dict:
        """Return a serialisable diagnostics dict (copies all arrays)."""
        n_att = int(self.att_swap_pair.sum())
        n_acc = int(self.acc_swap_pair.sum())
        return {
            "theta": {
                "attempts": self.att_theta.copy(),
                "accepts":  self.acc_theta.copy(),
                "rates":    (self.acc_theta / np.maximum(1, self.att_theta)).copy(),
            },
            "graph": {
                "attempts": self.att_graph.copy(),
                "accepts":  self.acc_graph.copy(),
                "rates":    (self.acc_graph / np.maximum(1, self.att_graph)).copy(),
            },
            # No "hyper" block: the edge-probability update is a Gibbs draw from
            # the exact Beta full conditional (Supplement 3.4), so it always
            # "accepts" and has no acceptance rate to report.
            "swap": {
                "overall": {
                    "attempts": n_att,
                    "accepts":  n_acc,
                    "rate":     float(n_acc / max(1, n_att)),
                },
                "adjacent_pairs": {
                    "attempts": self.att_swap_pair.copy(),
                    "accepts":  self.acc_swap_pair.copy(),
                    "rates":    (self.acc_swap_pair / np.maximum(1, self.att_swap_pair)).copy(),
                },
            },
            "temperatures": {
                "k_list": np.asarray(self.k_list, dtype=float).copy(),
                "betas":  self.betas.copy(),
            },
        }

    def print_summary(self):
        print("Parallel tempering diagnostics:")
        for r in range(self.n_chains):
            g_rate = self.acc_graph[r] / max(1, self.att_graph[r])
            t_rate = self.acc_theta[r] / max(1, self.att_theta[r])
            print(
                f"  Chain {r} (k={self.k_list[r]:.3g}, beta={self.betas[r]:.3f}) "
                f"Graph acc: {g_rate:.3f} ({self.acc_graph[r]}/{self.att_graph[r]}), "
                f"Theta acc: {t_rate:.3f} ({self.acc_theta[r]}/{self.att_theta[r]})"
            )
        n_att = int(self.att_swap_pair.sum())
        n_acc = int(self.acc_swap_pair.sum())
        if n_att > 0:
            print(f"  Swap acceptance (overall): {n_acc / n_att:.3f} ({n_acc}/{n_att})")
        else:
            print("  No swap attempts.")
        for i in range(self.n_chains - 1):
            rate = self.acc_swap_pair[i] / max(1, self.att_swap_pair[i])
            print(
                f"  Swap acc pair ({i},{i+1}) "
                f"(k={self.k_list[i]:.3g}↔{self.k_list[i+1]:.3g}): "
                f"{rate:.3f} ({self.acc_swap_pair[i]}/{self.att_swap_pair[i]})"
            )


# ---------------------------------------------------------------------------
# Worker infrastructure for parallel chain updates
# ---------------------------------------------------------------------------

def normal_logpdf_full(x, mean, sd):
    """Return the log-density of N(mean, sd^2) evaluated at x (scalar)."""
    z = (x - mean) / sd
    return -0.5 * z * z - math.log(sd) - 0.5 * math.log(2.0 * math.pi)

# Per-worker-process storage for read-only shared data (set by initialiser).
_worker_globals: dict = {}


def _init_chain_worker(X, src_idx, hyper, prior_hyper,
                       allowed_mask,
                       sigma_b, sigma_w, sigma_logpsi):
    """
    ProcessPoolExecutor initialiser: called once per worker process.
    Stores large read-only arrays and fixed hyperparameters so they are not
    re-pickled on every task submission.
    """
    # Ensure the module directory and project root are importable in workers
    # (needed on spawn-based systems such as macOS).
    worker_dir   = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(worker_dir, "..", ".."))
    for p in [worker_dir, project_root]:
        if p not in sys.path:
            sys.path.insert(0, p)

    _worker_globals.update(
        X=X, src_idx=src_idx,
        hyper=hyper, prior_hyper=prior_hyper,
        allowed_mask=allowed_mask,
        sigma_b=sigma_b, sigma_w=sigma_w, sigma_logpsi=sigma_logpsi,
    )


def _edge_logit(src_idx, alpha, rho_logits, i, j):
    """
    logit(Q[i, j]) for an allowed edge under make_Q_partitioned:
        src_idx[i,j] == 0 -> alpha            (baseline rho_0)
        src_idx[i,j] == k -> rho_logits[k-1]  (source-k rho_k)
    For a Bernoulli graph prior, toggling edge (i,j) from absent to present
    changes log p(A) by exactly log Q - log(1-Q) = logit(Q) = this value.
    Only ever called on allowed edges (src_idx >= 0), so -1 never reaches here.
    """
    s = int(src_idx[i, j])
    return float(alpha) if s == 0 else float(rho_logits[s - 1])


def _node_loglik(X, A_col, W_col, b_j, log_psi_j, j):
    """
    NB log-likelihood of node j given its own parent column of (A, W).

    eta_j is rebuilt from scratch with the SAME per-node arithmetic used in
    `_theta_mh_step_pernode` and evaluated with the SAME `nb_loglik_node`
    kernel, so a difference of two calls reproduces column j of the full
    `log_posterior` likelihood up to floating-point rounding.
    """
    parents = np.nonzero(A_col)[0]
    yj = np.ascontiguousarray(X[:, j])
    if parents.size:
        eta = X[:, parents] @ W_col[parents] + b_j
    else:
        eta = np.full(X.shape[0], b_j)
    return nb_loglik_node(yj, np.ascontiguousarray(eta), math.exp(log_psi_j))


def _graph_mh_step(A_curr, W_curr, b_curr, log_psi_curr,
                   alpha, rho_logits,
                   logpi_curr, lp_gh_curr, beta_r, rng,
                   X, src_idx, hyper, prior_hyper,
                   allowed_mask):
    """
    One MH step updating the graph structure and the weight of the affected edge.

    The untempered log-posterior change `d_logpi = logpi_prop - logpi_curr` is
    computed INCREMENTALLY from only the affected node(s) and edge(s) rather than
    by a full `log_posterior` recompute:

        d_logpi = Δ loglik(affected nodes)   # birth/death: child j; swap: i and j
                + Δ weight-prior             # the added/removed Normal(0, sd_w) term
                + Δ graph-prior              # ± logit(Q) of the toggled edge(s)

    This is mathematically identical to the old full-recompute acceptance ratio
    (the proposal terms log_q_fwd/bwd and the `log_alpha` formula are unchanged),
    but avoids the dense N×K×K `X @ (W*A)` matmul and the full Bernoulli
    graph-prior sum on every proposal. The RNG draw sequence is byte-for-byte
    identical to the old version (same proposal, same single accept uniform).

    Returns
    -------
    A_curr, W_curr, logpi_curr, lp_gh_curr, acc_graph, att_graph
    """
    acc_graph = 0
    att_graph = 0

    A_prop, move_type, move_detail, log_q_fwd, log_q_bwd = \
        propose_graph_move_with_probs(
            A_curr, rng, allowed_mask=allowed_mask)

    if move_type == "none":
        return A_curr, W_curr, logpi_curr, lp_gh_curr, acc_graph, att_graph

    log_q_fwd_total = log_q_fwd
    log_q_bwd_total = log_q_bwd
    w_mean = 0.0
    w_sd   = hyper["w"]["sd"]

    if move_type == "birth":
        i, j = move_detail["parent"], move_detail["child"]
        w_new = rng.normal(w_mean, w_sd)
        W_prop = W_curr.copy()
        W_prop[i, j] = w_new
        log_q_fwd_total = log_q_fwd + normal_logpdf_full(w_new, w_mean, w_sd)

        ll_old = _node_loglik(X, A_curr[:, j], W_curr[:, j], b_curr[j], log_psi_curr[j], j)
        ll_new = _node_loglik(X, A_prop[:, j], W_prop[:, j], b_curr[j], log_psi_curr[j], j)
        d_ll = ll_new - ll_old
        d_theta = normal_logpdf_full(w_new, 0.0, w_sd)
        d_lpA = _edge_logit(src_idx, alpha, rho_logits, i, j)

    elif move_type == "death":
        i, j = move_detail["parent"], move_detail["child"]
        w_old = W_curr[i, j]
        W_prop = W_curr.copy()
        W_prop[i, j] = 0.0
        log_q_bwd_total = log_q_bwd + normal_logpdf_full(w_old, w_mean, w_sd)

        ll_old = _node_loglik(X, A_curr[:, j], W_curr[:, j], b_curr[j], log_psi_curr[j], j)
        ll_new = _node_loglik(X, A_prop[:, j], W_prop[:, j], b_curr[j], log_psi_curr[j], j)
        d_ll = ll_new - ll_old
        d_theta = -normal_logpdf_full(w_old, 0.0, w_sd)
        d_lpA = -_edge_logit(src_idx, alpha, rho_logits, i, j)

    else:  # swap:  edge i->j is reversed to j->i
        i, j = move_detail["old_parent"], move_detail["old_child"]
        w_old = W_curr[i, j]
        w_new = rng.normal(w_mean, w_sd)
        W_prop = W_curr.copy()
        W_prop[i, j] = 0.0
        W_prop[j, i] = w_new
        log_q_fwd_total = log_q_fwd + normal_logpdf_full(w_new, w_mean, w_sd)
        log_q_bwd_total = log_q_bwd + normal_logpdf_full(w_old, w_mean, w_sd)

        # node j loses parent i; node i gains parent j -> both columns change.
        ll_old_j = _node_loglik(X, A_curr[:, j], W_curr[:, j], b_curr[j], log_psi_curr[j], j)
        ll_new_j = _node_loglik(X, A_prop[:, j], W_prop[:, j], b_curr[j], log_psi_curr[j], j)
        ll_old_i = _node_loglik(X, A_curr[:, i], W_curr[:, i], b_curr[i], log_psi_curr[i], i)
        ll_new_i = _node_loglik(X, A_prop[:, i], W_prop[:, i], b_curr[i], log_psi_curr[i], i)
        d_ll = (ll_new_j - ll_old_j) + (ll_new_i - ll_old_i)
        d_theta = (normal_logpdf_full(w_new, 0.0, w_sd)
                   - normal_logpdf_full(w_old, 0.0, w_sd))
        d_lpA = (_edge_logit(src_idx, alpha, rho_logits, j, i)
                 - _edge_logit(src_idx, alpha, rho_logits, i, j))

    d_logpi = d_ll + d_theta + d_lpA

    att_graph += 1
    log_alpha = beta_r * d_logpi + (log_q_bwd_total - log_q_fwd_total)
    if np.log(rng.uniform()) < log_alpha:
        A_curr, W_curr = A_prop, W_prop
        logpi_curr = logpi_curr + d_logpi
        # Re-anchor the graph+hyper log-prior with a full recompute (cheap: only
        # on acceptance). This keeps lp_gh_curr exact for the downstream hyper
        # step's baseline, matching the old code term-for-term.
        lp_gh_curr = log_prior_graph_and_hyper(
            A_curr, alpha, rho_logits,
            src_idx, prior_hyper)
        acc_graph += 1

    return A_curr, W_curr, logpi_curr, lp_gh_curr, acc_graph, att_graph


def _node_theta_logprior(bj, wj, lpj, sd_b, sd_w, a_psi, b_psi):
    """
    Log prior contribution of one node's theta = (b_j, W[:,j] on active edges,
    log_psi_j). Matches the per-node terms of dist.log_prior_theta:
      b_j    ~ Normal(0, sd_b^2)                 (normalized)
      w_ij   ~ Normal(0, sd_w^2)  for active i   (normalized)
      psi_j  ~ Gamma(a_psi, b_psi), psi=exp(lpj), plus log-Jacobian lpj.
    All densities are fully normalized so the running logpi equals a full
    recompute; the per-weight constant is load-bearing in the birth/death
    RJ ratio and in cross-chain swaps.
    """
    _half_log_2pi = 0.5 * math.log(2.0 * math.pi)
    psi = math.exp(lpj)
    lp = -0.5 * (bj * bj) / (sd_b * sd_b) - _half_log_2pi - math.log(sd_b)
    if wj.size:
        lp += (-0.5 * float(np.dot(wj, wj)) / (sd_w * sd_w)
               - wj.size * (_half_log_2pi + math.log(sd_w)))
    lp += (a_psi * math.log(b_psi) - math.lgamma(a_psi)
           + (a_psi - 1.0) * math.log(psi) - b_psi * psi)
    lp += lpj                                    # Jacobian d psi / d log_psi
    return lp


def _theta_mh_step_pernode(A_curr, W_curr, b_curr, log_psi_curr,
                           logpi_curr, beta_r, rng,
                           X, hyper,
                           sigma_b, sigma_w, sigma_logpsi):
    """
    Metropolis-within-Gibbs sweep over nodes for theta = (b, W, log_psi).

    For each node j, jointly propose (b_j, W[:,j] on its active parent edges,
    log_psi_j) with a symmetric random walk and accept/reject using ONLY node
    j's likelihood term plus node j's theta prior. This is exact: conditional on
    the graph A, the posterior over theta factorises over nodes (each node's
    NB-regression likelihood depends only on its own parameters), so the terms
    for every other node cancel in the acceptance ratio.

    The untempered running log-posterior `logpi_curr` is updated incrementally by
    each accepted node's exact delta.

    Returns
    -------
    b_curr, W_curr, log_psi_curr, logpi_curr, acc_theta, att_theta
        acc_theta/att_theta count accepted/attempted *node* updates.
    """
    N, K = X.shape
    sd_b = hyper["b"]["sd"]
    sd_w = hyper["w"]["sd"]
    a_psi = hyper["psi"]["a"]
    b_psi = hyper["psi"]["b"]

    y = np.ascontiguousarray(X)
    acc_theta = 0

    # Work on copies so the caller's arrays are never mutated in place.
    b_curr = b_curr.copy()
    W_curr = W_curr.copy()
    log_psi_curr = log_psi_curr.copy()

    for j in range(K):
        parents = np.nonzero(A_curr[:, j])[0]
        yj = np.ascontiguousarray(y[:, j])
        Xp = X[:, parents] if parents.size else None

        bj = b_curr[j]
        lpj = log_psi_curr[j]
        wj = W_curr[parents, j] if parents.size else np.empty(0)

        eta_old = (Xp @ wj + bj) if parents.size else np.full(N, bj)
        ll_old = nb_loglik_node(yj, np.ascontiguousarray(eta_old), math.exp(lpj))
        lp_old = _node_theta_logprior(bj, wj, lpj, sd_b, sd_w, a_psi, b_psi)

        bj_new = bj + rng.normal(0.0, sigma_b)
        lpj_new = lpj + rng.normal(0.0, sigma_logpsi)
        if parents.size:
            wj_new = wj + rng.normal(0.0, sigma_w, size=parents.size)
            eta_new = Xp @ wj_new + bj_new
        else:
            wj_new = wj
            eta_new = np.full(N, bj_new)

        ll_new = nb_loglik_node(yj, np.ascontiguousarray(eta_new), math.exp(lpj_new))
        lp_new = _node_theta_logprior(bj_new, wj_new, lpj_new, sd_b, sd_w, a_psi, b_psi)

        d_untemp = (ll_new - ll_old) + (lp_new - lp_old)   # exact logpi delta
        if math.log(rng.uniform()) < beta_r * d_untemp:
            b_curr[j] = bj_new
            log_psi_curr[j] = lpj_new
            if parents.size:
                W_curr[parents, j] = wj_new
            logpi_curr += d_untemp
            acc_theta += 1

    return b_curr, W_curr, log_psi_curr, logpi_curr, acc_theta, K


def _graph_prior_counts(A_curr, src_idx):
    """
    Per-source (present, absent) edge counts m_s, mbar_s over allowed edges.
    These are the sufficient statistics of the Bernoulli graph prior: they drive
    the O(S) cached prior evaluation (`_lp_graph_hyper_cached`) and the Beta
    full conditionals of the edge-probability Gibbs step (`_edge_prob_gibbs_step`).

    An edge (i, j) is allowed iff i != j and src_idx[i, j] >= 0. Its source id
    s = src_idx[i, j] indexes a single shared probability rho_s (s=0 -> baseline
    rho_0, s=k -> rho_k), so the full per-edge Bernoulli prior collapses to
        sum_s [ present[s] * log rho_s + absent[s] * log(1 - rho_s) ].
    `present[s]` / `absent[s]` depend only on A (fixed across the sweep).

    Returns float64 arrays of length S+1 (index 0 = baseline).
    """
    n = A_curr.shape[0]
    src_idx = np.asarray(src_idx)
    offdiag = ~np.eye(n, dtype=bool)
    allowed = offdiag & (src_idx >= 0)
    s_allowed = src_idx[allowed].astype(np.int64)
    A_allowed = (A_curr[allowed] != 0).astype(np.float64)
    S = int(src_idx.max())
    total   = np.bincount(s_allowed, minlength=S + 1).astype(np.float64)
    present = np.bincount(s_allowed, weights=A_allowed, minlength=S + 1)
    absent  = total - present
    return present, absent


def _lp_graph_hyper_cached(alpha, rho_logits, present, absent, prior_hyper):
    """
    Cached O(S) evaluation of log p(A, alpha, rho_logits) for a FIXED graph A,
    given the per-source present/absent counts from _graph_prior_counts.

    Reproduces `log_prior_graph_and_hyper` up to floating-point rounding:
    the graph term is regrouped by source (log rho_s and log(1-rho_s) computed
    on the logit scale via -softplus), and the hyper term reuses the exact same
    `log_prior_hyper` call.
    """
    x = np.empty(present.shape[0], dtype=np.float64)
    x[0] = alpha
    x[1:] = np.asarray(rho_logits, dtype=np.float64)
    log_rho   = -np.logaddexp(0.0, -x)      # = log(expit(x))    = log rho_s
    log_1mrho = -np.logaddexp(0.0,  x)      # = log(1 - expit(x)) = log(1 - rho_s)
    lp_A = float(present @ log_rho + absent @ log_1mrho)
    lp_params = log_prior_hyper(alpha, rho_logits, prior_hyper)
    return lp_A + lp_params


def _edge_prob_gibbs_step(A_curr, alpha, rho_logits,
                          lp_gh_curr, logpi_curr, beta_r, rng,
                          src_idx, prior_hyper):
    """
    Gibbs resample of the edge probabilities Phi_A = (rho_0, rho_1, ..., rho_K)
    from their Beta full conditionals (Supplement Section 3.4, "Edge-probability
    update"). Replaces the old random-walk Metropolis sweep over (alpha,
    rho_logits): each rho_s has a closed-form full conditional, so the whole
    block is resampled in one shot and the step always "accepts".

    Holding the graph A (and the effect parameters theta) fixed, the graph prior
    depends on Phi_A only through per-source edge counts. Grouping the allowed
    ordered pairs (i, j) by source s = src_idx[i, j], let m_s and mbar_s be the
    number of PRESENT and ABSENT allowed edges of source s (s = 0 baseline,
    s = 1..K evidence). With an independent Beta(a_s, b_s) hyper-prior on rho_s,
    Beta-Bernoulli conjugacy gives the untempered (cold-chain) full conditional

        rho_s | A  ~  Beta(a_s + m_s, b_s + mbar_s).

    Chain r targets the tempered posterior pi_r(x) ∝ P(x)^{beta_r}, whose rho_s
    full conditional is again Beta but with shape parameters scaled and
    re-centred by beta_r (Supplement eq. for the tempered draw),

        rho_s | A  ~  Beta( beta_r (a_s + m_s - 1) + 1,
                            beta_r (b_s + mbar_s - 1) + 1 ),

    which reduces to the line above at beta_r = 1. Each rho_s is drawn
    independently from this Beta at the chain's own inverse temperature beta_r.
    (This is exact only if the sampler tempers rho on its natural scale, i.e. the
    rho prior enters logpi WITHOUT a logit Jacobian -- see dist.log_prior_hyper.)

    rho is stored on the logit scale for the rest of the pipeline
    (alpha = logit(rho_0), rho_logits[k-1] = logit(rho_k)); the freshly drawn rho
    is converted back to logits here. The cached graph+hyper log-prior lp_gh_curr
    and the untempered log-posterior logpi_curr are refreshed to the resampled
    rho on return.

    Returns
    -------
    alpha, rho_logits, lp_gh_curr, logpi_curr
    """
    # Per-source (present, absent) = (m_s, mbar_s) counts over allowed edges;
    # these depend only on the fixed graph A. Index 0 is the baseline source.
    present, absent = _graph_prior_counts(A_curr, src_idx)      # length K + 1

    # Beta hyper-prior shape vectors (a_s, b_s), index 0 = baseline rho_0.
    a_prior = np.empty(present.shape[0], dtype=np.float64)
    b_prior = np.empty(present.shape[0], dtype=np.float64)
    a_prior[0]  = prior_hyper["rho"]["a"]
    b_prior[0]  = prior_hyper["rho"]["b"]
    a_prior[1:] = np.asarray(prior_hyper["rho_k"]["a"], dtype=np.float64)
    b_prior[1:] = np.asarray(prior_hyper["rho_k"]["b"], dtype=np.float64)

    # Re-anchor logpi's graph+hyper term to a fresh evaluation at the CURRENT rho
    # (corrects any floating-point drift in the incremental logpi), then apply the
    # resampled rho as an exact delta, so the returned logpi_curr stays consistent
    # with lp_gh_curr -- both feed the replica-exchange acceptance ratio.
    lp_gh_old = _lp_graph_hyper_cached(alpha, rho_logits, present, absent, prior_hyper)
    logpi_curr += lp_gh_old - lp_gh_curr

    # Tempered Beta full conditional (Supplement 3.4). For the priors used here
    # (a_s, b_s >= 1) both shapes stay >= 1, so the Beta is proper for every beta_r.
    shape1 = beta_r * (a_prior + present - 1.0) + 1.0
    shape2 = beta_r * (b_prior + absent  - 1.0) + 1.0
    rho = rng.beta(shape1, shape2)

    # Store on the logit scale; clip away from {0, 1} so the logit stays finite
    # (the Beta draw is a.s. interior for shapes >= 1; the clip is a guard only).
    rho = np.clip(rho, 1e-12, 1.0 - 1e-12)
    logits = np.log(rho) - np.log1p(-rho)
    alpha = float(logits[0])
    rho_logits = np.ascontiguousarray(logits[1:])

    lp_gh_new = _lp_graph_hyper_cached(alpha, rho_logits, present, absent, prior_hyper)
    logpi_curr += lp_gh_new - lp_gh_old
    lp_gh_curr = lp_gh_new

    return alpha, rho_logits, float(lp_gh_curr), float(logpi_curr)


def _swap_step(states, i, j, betas, rng):
    """
    Propose a replica-exchange swap between adjacent chains i and j.

    The MH acceptance ratio for swapping states x_i and x_j is:
        log α = (beta_i - beta_j) * (log π(x_j) - log π(x_i))
    where log π is the untempered log-posterior.

    Parameters
    ----------
    states : list of ChainState
        Current states of all chains (mutated in-place on acceptance).
    i, j : int
        Indices of the two adjacent chains to consider swapping (j = i + 1).
    betas : np.ndarray
        Inverse temperatures for all chains.
    rng : np.random.Generator

    Returns
    -------
    accepted : bool
    """
    log_alpha = (betas[i] - betas[j]) * (states[j].logpi - states[i].logpi)
    if np.log(rng.uniform()) < log_alpha:
        states[i], states[j] = states[j], states[i]
        return True
    return False


def _chain_local_step(args):
    """
    Execute one full local update (graph + theta + edge-prob) for a single chain.
    Runs in a worker process; reads shared data from _worker_globals.

    Parameters
    ----------
    args : tuple
        (A, b, W, log_psi,
         alpha, rho_logits,
         lp_graph_hyper_r, logpi_r, beta_r, rng_seed)

    Returns
    -------
    tuple : updated state + per-step acceptance/attempt counts
        (A, b, W, log_psi, alpha, rho_logits,
         lp_graph_hyper, logpi,
         acc_graph, att_graph, acc_theta, att_theta)
    """
    (A_curr, b_curr, W_curr, log_psi_curr,
     alpha, rho_logits,
     lp_gh_curr, logpi_curr, beta_r, rng_seed) = args

    g             = _worker_globals
    X             = g['X']
    src_idx       = g['src_idx']
    hyper         = g['hyper']
    prior_hyper   = g['prior_hyper']
    allowed_mask  = g['allowed_mask']
    sigma_b       = g['sigma_b']
    sigma_w       = g['sigma_w']
    sigma_logpsi  = g['sigma_logpsi']

    rng = np.random.default_rng(rng_seed)

    A_curr, W_curr, logpi_curr, lp_gh_curr, acc_graph, att_graph = _graph_mh_step(
        A_curr, W_curr, b_curr, log_psi_curr,
        alpha, rho_logits,
        logpi_curr, lp_gh_curr, beta_r, rng,
        X, src_idx, hyper, prior_hyper, allowed_mask,
    )

    b_curr, W_curr, log_psi_curr, logpi_curr, acc_theta, att_theta = _theta_mh_step_pernode(
        A_curr, W_curr, b_curr, log_psi_curr,
        logpi_curr, beta_r, rng,
        X, hyper, sigma_b, sigma_w, sigma_logpsi,
    )

    (alpha, rho_logits,
     lp_gh_curr, logpi_curr) = _edge_prob_gibbs_step(
        A_curr, alpha, rho_logits,
        lp_gh_curr, logpi_curr, beta_r, rng,
        src_idx, prior_hyper,
    )

    return (A_curr, b_curr, W_curr, log_psi_curr,
            float(alpha), rho_logits,
            float(lp_gh_curr), float(logpi_curr),
            acc_graph, att_graph, acc_theta, att_theta)


def _save_pickle(obj, path, compress=False, compress_level=3):
    """Atomically pickle `obj` to `path`, optionally zstd-compressed."""
    tmp = path + ".tmp"
    if compress:
        try:
            import zstandard as zstd
        except ImportError as e:
            raise ImportError("--compress requires `pip install zstandard`") from e
        cctx = zstd.ZstdCompressor(level=int(compress_level), threads=-1)
        with open(tmp, "wb") as f, cctx.stream_writer(f) as w:
            pickle.dump(obj, w)
    else:
        with open(tmp, "wb") as f:
            pickle.dump(obj, f)
    os.replace(tmp, path)


def load_warm_start(init_path, col_names, src_idx, *, dag_only=False,
                    b_default=None, W_default=None, log_psi_default=None):
    """
    Load a warm-start initial state produced by warm_start.py.

    Accepts either an .npz (with A and optionally b, W, log_psi, col_names) or a
    plain adjacency CSV (rows -> columns, node-named). The loaded graph is
    validated against the current problem: node ordering must match col_names,
    every edge must be prior-allowed (src_idx != -1), the diagonal must be zero,
    and the graph must be acyclic.

    Parameters
    ----------
    init_path : str
        Path to the .npz or .csv warm-start file.
    col_names : list of str
        Node names (X columns) in the order the sampler uses.
    src_idx : np.ndarray, shape (n, n), int
        Source-index matrix; src_idx == -1 marks forbidden edges.
    dag_only : bool
        If True, return only the adjacency and keep the provided defaults for
        b, W, log_psi (ignore any theta stored in the file).
    b_default, W_default, log_psi_default : np.ndarray or None
        Fallback theta used when the file has no theta or dag_only is True.

    Returns
    -------
    A : (n, n) int8
    b, W, log_psi : np.ndarray   (from the file when available, else defaults)
    """
    n = len(col_names)
    b = None if b_default is None else np.array(b_default, dtype=np.float64)
    W = None if W_default is None else np.array(W_default, dtype=np.float64)
    log_psi = None if log_psi_default is None else np.array(log_psi_default, dtype=np.float64)

    ext = os.path.splitext(init_path)[1].lower()
    if ext == ".npz":
        data = np.load(init_path, allow_pickle=True)
        if "col_names" in data:
            file_cols = [str(c) for c in data["col_names"].tolist()]
            if file_cols != list(col_names):
                raise ValueError(
                    "Warm-start col_names do not match X columns in the same order.\n"
                    f"  X columns    : {list(col_names)}\n"
                    f"  warm-start   : {file_cols}"
                )
        A = np.asarray(data["A"]).astype(np.int8)
        if not dag_only:
            if "b" in data:       b = np.asarray(data["b"], dtype=np.float64)
            if "W" in data:       W = np.asarray(data["W"], dtype=np.float64)
            if "log_psi" in data: log_psi = np.asarray(data["log_psi"], dtype=np.float64)
    elif ext == ".csv":
        df = pd.read_csv(init_path, index_col=0)
        if list(df.index) != list(col_names) or list(df.columns) != list(col_names):
            raise ValueError(
                "Warm-start CSV row/column names must match X columns in the same order."
            )
        A = df.to_numpy().astype(np.int8)
    else:
        raise ValueError(f"--init_path must be .npz or .csv; got {init_path!r}")

    # ---- Validate the starting graph ----
    if A.shape != (n, n):
        raise ValueError(f"Warm-start A has shape {A.shape}, expected {(n, n)}.")
    if not np.all(np.isin(A, (0, 1))):
        raise ValueError("Warm-start A must be binary (0/1).")
    if np.any(np.diag(A) != 0):
        raise ValueError("Warm-start A must have a zero diagonal (no self-edges).")
    forbidden = np.asarray(src_idx) == -1
    if np.any(A[forbidden] != 0):
        n_bad = int(np.count_nonzero(A[forbidden]))
        raise ValueError(
            f"Warm-start A contains {n_bad} forbidden edge(s) (src_idx == -1). "
            "The starting graph must respect the prior."
        )
    if has_cycle(A):
        raise ValueError(
            "Warm-start A is not a DAG (contains a cycle). Regenerate it with "
            "warm_start.py, which resolves cycles before saving."
        )

    return A, b, W, log_psi


# ---------------------------------------------------------------------------
# Main MCMC sampler
# ---------------------------------------------------------------------------

def run_mcmc_parallel_tempering(
    X,
    A_init,
    b_init,
    W_init,
    log_psi_init,
    src_idx,
    alpha_init,
    rho_logits_init,
    hyper,
    prior_hyper,
    rng,
    n_iter=150000,
    n_chains=10,
    factor=2,
    sigma_b=0.01,
    sigma_w=0.01,
    sigma_logpsi=0.01,
    thin=1,
    p_exchange_total=0.2,
    parallel_chains=False,
    checkpoint_path=None,
    checkpoint_interval=10000,
    col_names=None,
    compress=False,
    compress_level=3,
):
    """
    Parallel tempering (replica exchange) MCMC for joint inference of (G, theta).

    Chains r = 0,...,n_chains-1 target the tempered posterior
        Pi_r(theta, G | D)  ∝  Pi(theta, G | D)^(beta_r)
    where beta_r = factor^(-r).  Chain 0 has beta_0 = 1 and targets the true
    posterior; subsequent chains are progressively hotter (beta < 1) to aid
    mixing across graph structures.

    Parameters
    ----------
    X : np.ndarray, shape (n_obs, n_nodes)
        Observed data matrix.
    A_init : np.ndarray, shape (n_nodes, n_nodes), dtype int8
        Initial adjacency matrix (DAG).  Off-diagonal 0/1 entries; diagonal is 0.
    b_init : np.ndarray, shape (n_nodes,)
        Initial intercept vector.
    W_init : np.ndarray, shape (n_nodes, n_nodes)
        Initial edge-weight matrix.  Entries where A_init=0 are zeroed on entry.
    log_psi_init : np.ndarray, shape (n_nodes,)
        Initial log-dispersion parameters.
    src_idx : np.ndarray, shape (n_nodes, n_nodes), dtype int64
        Source-index matrix encoding prior edge beliefs.
        -1 = forbidden edge, 0 = baseline (no evidence), >=1 = evidence level.
    alpha_init : float
        Initial value of alpha = logit(rho_0) (baseline edge log-odds).
    rho_logits_init : array-like, shape (n_sources,)
        Initial values of logit(rho_k) for each evidence source k = 1..S.
    hyper : dict
        Hyperparameters for the likelihood/prior on theta:
          - "b"   : {"sd": float}
          - "w"   : {"sd": float}
          - "psi" : {"a": float, "b": float}
    prior_hyper : dict
        Hyperparameters for the prior on (alpha, rho_logits):
          - "rho"   : {"a": float, "b": float}   (Beta prior on rho_0)
          - "rho_k" : {"a": (S,) array, "b": (S,) array}
                      (per-source Beta prior on rho_k)
    rng : np.random.Generator
        NumPy random generator (use np.random.default_rng(seed)).
        Graph-move proposals are restricted to edges the prior permits
        (src_idx != -1), derived internally as `allowed_mask`; forbidden
        entries are never proposed.
    n_iter : int
        Total number of MCMC iterations (swap + local combined).
    n_chains : int
        Number of parallel chains (replicas).
    factor : float  (> 1)
        Geometric spacing factor for inverse temperatures:
        beta_r = factor^(-r).  Larger values widen the temperature ladder.
    sigma_b, sigma_w, sigma_logpsi : float
        Random-walk step sizes for b, W, and log_psi respectively.
    thin : int
        Thinning interval; save every `thin`-th sample.
    p_exchange_total : float in [0, 1]
        Probability that a given iteration is a swap step rather than a
        local-update step.
    checkpoint_path : str or None
        If provided, write a checkpoint pickle to this path every
        `checkpoint_interval` iterations.
    checkpoint_interval : int
        Number of iterations between checkpoints.
    col_names : list of str or None
        Node names to store in checkpoint files for downstream use.

    Returns
    -------
    samples : SampleStore
        Posterior samples from the cold chain (graphs, b, W, log_psi,
        alpha, rho_logits).
    logpost_trace_cold : np.ndarray, shape (n_iter,)
        Untempered log-posterior trace for the cold chain (chain 0).
    logpost_traces_all : np.ndarray, shape (n_chains, n_iter)
        Untempered log-posterior traces for all chains.
    edge_counts_all : np.ndarray, shape (n_chains, n_iter)
        Edge count A.sum() for every chain at each iteration (diagnoses
        whether hotter chains drift into denser graphs).
    diagnostics : dict
        Acceptance rates and swap statistics; keys:
        "theta", "graph", "swap", "temperatures". (The edge-probability update
        is an exact Gibbs draw, so it has no acceptance rate and no diag block.)
    """
    W_init = W_init * A_init

    if n_chains == 1:
        parallel_chains = False

    if factor <= 1:
        raise ValueError("factor must be > 1 for geometric spacing.")

    betas = factor ** (-np.arange(n_chains))

    if np.any(betas <= 0) or np.any(betas > 1):
        raise ValueError("All betas must lie in (0,1].")

    k_list = 1.0 / betas

    # Graph-move proposals are restricted to prior-permitted edges: an edge
    # i -> j may be proposed iff it is not forbidden (src_idx != -1) and i != j.
    # Forbidden edges carry an -inf graph prior, so excluding them here changes
    # nothing about the target distribution while skipping the ~half of
    # candidate pairs that would be rejected on sight.
    src_idx_arr = np.asarray(src_idx)
    allowed_mask = (src_idx_arr != -1) & ~np.eye(src_idx_arr.shape[0], dtype=bool)

    # Populate worker globals in the main process (used by sequential path
    # and also serves as the initialiser argument for worker processes).
    _init_chain_worker(
        np.asarray(X, dtype=np.float64), src_idx,
        hyper, prior_hyper,
        allowed_mask,
        sigma_b, sigma_w, sigma_logpsi,
    )

    # Pre-compile the Numba kernels in this parent process *before* any worker
    # is spawned. Otherwise every worker -- and, in a SLURM array job, every
    # array task sharing this directory -- races to JIT-compile the same
    # functions at once. With on-disk caching that corrupts Numba's cache on a
    # shared network filesystem (os.replace of the cache index fails). The
    # kernels are compiled with cache=False (nothing is written to disk), so
    # this pre-warm makes startup race-free: with fork the workers inherit the
    # already-compiled code; with spawn each compiles independently in memory.
    from graph_fast import _reachability, _has_cycle
    from dist import nb_loglik_2d
    _reachability(np.zeros((2, 2), dtype=np.int8))
    _has_cycle(np.zeros((2, 2), dtype=np.int8))
    _warm = np.zeros(1, dtype=np.float64)
    nb_loglik_node(_warm, _warm.copy(), 1.0)
    nb_loglik_2d(np.zeros((1, 1)), np.zeros((1, 1)), np.ones(1))

    # Spin up a process pool for parallel chain updates (if requested).
    # The pool lives for the full run; workers are initialised once with the
    # large read-only arrays so they are not re-pickled every iteration.
    executor = None
    if parallel_chains and n_chains > 1:
        executor = ProcessPoolExecutor(
            max_workers=n_chains,
            initializer=_init_chain_worker,
            initargs=(
                np.asarray(X, dtype=np.float64), src_idx,
                hyper, prior_hyper,
                allowed_mask,
                sigma_b, sigma_w, sigma_logpsi,
            ),
        )

    # Initialise one ChainState per replica (all start from the same point).
    states: list[ChainState] = []
    for r in range(n_chains):
        A_r          = deepcopy(A_init)
        b_r          = deepcopy(b_init)
        W_r          = deepcopy(W_init)
        lp_r         = deepcopy(log_psi_init)
        alpha_r      = float(alpha_init)
        rho_logits_r = deepcopy(np.asarray(rho_logits_init, dtype=np.float64))

        lp_gh_r = log_prior_graph_and_hyper(
            A_r, alpha_r, rho_logits_r,
            src_idx, prior_hyper)
        logpi_r = log_posterior(
            X, A_r, b_r, W_r, lp_r,
            alpha_r, rho_logits_r,
            src_idx, hyper, prior_hyper, return_parts=False,
        )
        states.append(ChainState(
            A_r, b_r, W_r, lp_r,
            alpha_r, rho_logits_r,
            lp_gh_r, logpi_r))

    # Traces of untempered log-posterior
    logpost_trace_cold = np.empty(n_iter)
    logpost_traces_all = np.empty((n_chains, n_iter))

    # Per-chain edge counts (A.sum()) at every iteration, to diagnose whether
    # hotter chains drift into denser graphs.
    edge_counts_all = np.empty((n_chains, n_iter), dtype=np.int32)

    diag = DiagAccumulator(n_chains, k_list, betas)

    samples = SampleStore()

    # Time the sampling loop, accumulating the periodic checkpoint writes
    # separately so they can be subtracted. What is left is the sampler alone,
    # which is what dag0.R reports as its SAMPLER TIME for mcmc_1sampleDAG0
    # (benchmarking/dag0/dag0.R): that sampler does no I/O inside its loop, so
    # subtracting ours is what makes the two numbers comparable.
    t_start = time.perf_counter()
    checkpoint_seconds = 0.0

    for t in trange(n_iter, desc="Parallel tempering MCMC"):
        # ---- Swap step ----
        if n_chains > 1 and rng.uniform() < p_exchange_total:
            i = int(rng.integers(n_chains - 1))
            j = i + 1
            diag.record_swap(i, _swap_step(states, i, j, betas, rng))

        # ---- Local updates: graph + theta + edge-prob for each chain ----
        else:
            if executor is not None:
                seeds = rng.integers(0, 2**31, size=n_chains)
                chain_args = [
                    (states[r].A, states[r].b, states[r].W, states[r].log_psi,
                     states[r].alpha, states[r].rho_logits,
                     states[r].lp_graph_hyper, states[r].logpi,
                     betas[r], int(seeds[r]))
                    for r in range(n_chains)
                ]
                results = list(executor.map(_chain_local_step, chain_args))
            else:
                results = []
                for r in range(n_chains):
                    seed_r = int(rng.integers(0, 2**31))
                    results.append(_chain_local_step((
                        states[r].A, states[r].b, states[r].W, states[r].log_psi,
                        states[r].alpha, states[r].rho_logits,
                        states[r].lp_graph_hyper, states[r].logpi,
                        betas[r], seed_r,
                    )))

            for r, res in enumerate(results):
                (A, b, W, lp, alpha, rho_logits,
                 lp_gh, lpi,
                 acc_g, att_g, acc_t, att_t) = res
                states[r] = ChainState(
                    A, b, W, lp,
                    alpha, rho_logits,
                    lp_gh, lpi)
                diag.record_local(r, acc_g, att_g, acc_t, att_t)

        # ---- Record traces ----
        for r in range(n_chains):
            logpost_traces_all[r, t] = states[r].logpi
            edge_counts_all[r, t] = int(states[r].A.sum())
        logpost_trace_cold[t] = states[0].logpi

        # ---- Collect cold-chain samples (thinning; burn-in applied post-hoc) ----
        if t % thin == 0:
            samples.record(states[0])

        # ---- Periodic checkpoint ----
        if checkpoint_path is not None and (t + 1) % checkpoint_interval == 0:
            ckpt = {
                "graphs_samples":      samples.graphs,
                "b_samples":           samples.b,
                "W_samples":           samples.W,
                "log_psi_samples":     samples.log_psi,
                "alpha_samples":       samples.alpha,
                "rho_logits_samples":  samples.rho_logits,
                "logpost_trace":       logpost_trace_cold[:t + 1].copy(),
                "logpost_trace_all":   logpost_traces_all[:, :t + 1].copy(),
                "edge_counts_all":     edge_counts_all[:, :t + 1].copy(),
                "iteration":           t + 1,
                "n_nodes":             int(states[0].A.shape[0]),
                "col_names":           col_names,
                "diagnostics": diag.build(),
            }
            t_ckpt = time.perf_counter()
            _save_pickle(ckpt, checkpoint_path, compress=compress, compress_level=compress_level)
            checkpoint_seconds += time.perf_counter() - t_ckpt
            print(f"  [Checkpoint saved at iteration {t + 1}]")
            diag.print_summary()

    # ---- Final diagnostics ----
    sampler_seconds = time.perf_counter() - t_start - checkpoint_seconds

    diagnostics = diag.build()
    diagnostics["sampler_seconds"] = sampler_seconds
    diagnostics["checkpoint_seconds"] = checkpoint_seconds
    diag.print_summary()

    print(f"\nSAMPLER TIME  {sampler_seconds:.1f} s  ({sampler_seconds / 3600:.2f} h)"
          f"  for {n_iter} iterations x {n_chains} chains")
    print(f"SAMPLER RATE  {sampler_seconds / n_iter:.4f} s/iteration"
          f"  ({n_iter / sampler_seconds:.1f} iterations/s)")
    print(f"CHECKPOINT IO {checkpoint_seconds:.1f} s  (excluded from SAMPLER TIME)")

    if executor is not None:
        executor.shutdown(wait=True)

    return (
        samples,
        logpost_trace_cold,
        logpost_traces_all,
        edge_counts_all,
        diagnostics,
    )

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Loads data (X, Q), builds the prior structure, runs parallel
    # tempering MCMC, and writes results to a pickle file.

    parser = argparse.ArgumentParser(
        description="Run Parallel Tempering MCMC for Bayesian DAG inference."
    )

    parser.add_argument(
        "--x_path",
        type=str,
        required=True,
        help="Path to X.csv file"
    )

    parser.add_argument(
        "--q_path",
        type=str,
        required=True,
        help="Path to Q matrix CSV (required)"
    )

    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Output path for .pkl results"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=31,
        help="Random seed"
    )

    parser.add_argument(
        "--n_iter",
        type=int,
        default=150000,
        help="Number of MCMC iterations"
    )

    parser.add_argument(
        "--n_chains",
        type=int,
        default=10,
        help="Number of parallel tempering chains"
    )

    parser.add_argument(
        "--thin",
        type=int,
        default=1,
        help="Thinning interval; save every k-th sample (burn-in applied post-hoc)"
    )

    parser.add_argument(
        "--factor",
        type=float,
        default=2 ** (1/9),
        help="Geometric spacing factor for inverse temperatures: beta_r = factor^(-r). "
             "Smaller values bring chains closer together, improving swap acceptance rates."
    )

    parser.add_argument(
        "--sigma_b",
        type=float,
        default=2e-4,
        help="Random-walk step size for b"
    )

    parser.add_argument(
        "--sigma_w",
        type=float,
        default=2e-4,
        help="Random-walk step size for W"
    )

    parser.add_argument(
        "--sigma_logpsi",
        type=float,
        default=2e-4,
        help="Random-walk step size for log_psi"
    )

    parser.add_argument(
        "--prior_w_sd",
        type=float,
        default=0.05,
        help="SD of the Normal(0, sd^2) prior on each edge effect w_ij "
             "(the prior, not the proposal step size --sigma_w)"
    )

    parser.add_argument(
        "--prior_b_sd",
        type=float,
        default=2.0,
        help="SD of the Normal(0, sd^2) prior on each node intercept b_j"
    )

    parser.add_argument(
        "--prior_psi_a",
        type=float,
        default=1.0,
        help="Shape of the Gamma(a, b) prior on each overdispersion psi_j"
    )

    parser.add_argument(
        "--prior_psi_b",
        type=float,
        default=2.0,
        help="Rate of the Gamma(a, b) prior on each overdispersion psi_j"
    )

    parser.add_argument(
        "--prior_rho_a",
        type=float,
        default=1.0,
        help="Shape parameter a of the Beta prior on baseline edge probability rho_0 "
             "(prior_hyper['rho']['a']). Default 1.0 with b=1.0 gives a flat "
             "Beta(1,1)=Uniform(0,1) prior, letting the data drive the baseline rate "
             "(matches run.slurm)."
    )

    parser.add_argument(
        "--prior_rho_b",
        type=float,
        default=1.0,
        help="Shape parameter b of the Beta prior on baseline edge probability rho_0 "
             "(prior_hyper['rho']['b']). Default 1.0 with a=1.0 gives a flat "
             "Beta(1,1)=Uniform(0,1) prior on rho_0."
    )

    parser.add_argument(
        "--prior_rho_k_param",
        type=str,
        default="ab",
        choices=["ab", "mean_conc"],
        help="Parameterization for the per-source Beta prior on rho_k (k=1..K). "
             "'ab' (default): read the shape parameters directly from "
             "--prior_rho_k_a / --prior_rho_k_b. 'mean_conc': read a prior mean "
             "and concentration from --prior_rho_k_mean / --prior_rho_k_conc and "
             "convert via a_k = mean_k * conc_k, b_k = (1 - mean_k) * conc_k."
    )

    parser.add_argument(
        "--prior_rho_k_a",
        type=str,
        default="1.0",
        help="Shape parameter a of the Beta prior on each source edge "
             "probability rho_k (k=1..K), as a comma-separated list of K "
             "values (prior_hyper['rho_k']['a']). A single value is broadcast "
             "to all K sources. Used only when --prior_rho_k_param=ab. Default "
             "'1.0' gives Beta(1,1)=Uniform(0,1) on every rho_k."
    )

    parser.add_argument(
        "--prior_rho_k_b",
        type=str,
        default="1.0",
        help="Shape parameter b of the Beta prior on each source edge "
             "probability rho_k (k=1..K), as a comma-separated list of K "
             "values (prior_hyper['rho_k']['b']). A single value is broadcast "
             "to all K sources. Used only when --prior_rho_k_param=ab. Default "
             "'1.0' gives Beta(1,1)=Uniform(0,1) on every rho_k."
    )

    parser.add_argument(
        "--prior_rho_k_mean",
        type=str,
        default="0.5",
        help="Prior mean of each source edge probability rho_k (k=1..K), as a "
             "comma-separated list of K values in (0,1). A single value is "
             "broadcast to all K sources. Used only when "
             "--prior_rho_k_param=mean_conc; converted to shape parameters via "
             "a_k = mean_k * conc_k, b_k = (1 - mean_k) * conc_k."
    )

    parser.add_argument(
        "--prior_rho_k_conc",
        type=str,
        default="2.0",
        help="Prior concentration conc_k = a_k + b_k (> 0) of each source edge "
             "probability rho_k (k=1..K), as a comma-separated list of K values. "
             "A single value is broadcast to all K sources. Used only when "
             "--prior_rho_k_param=mean_conc. Default mean=0.5, conc=2.0 gives "
             "Beta(1,1)=Uniform(0,1) on every rho_k."
    )

    parser.add_argument(
        "--compress",
        action="store_true",
        help="Zstd-compress the output pkl (and checkpoint). Requires the "
             "`zstandard` Python package. If set and --out_path does not already "
             "end in '.zst', '.zst' is appended automatically."
    )

    parser.add_argument(
        "--compress_level",
        type=int,
        default=3,
        help="Zstd compression level (1-22) when --compress is set. Default 3."
    )

    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=10000,
        help="Number of iterations between checkpoint writes. Each write "
             "serializes the WHOLE accumulated sample store, so the total "
             "checkpoint cost grows quadratically in n_iter at a fixed "
             "interval: halving the number of writes halves that cost. Scale "
             "this with --n_iter (e.g. 250000 for a 10,000,000-iteration run, "
             "which keeps the total checkpoint I/O comparable to the 200 writes "
             "of the 2,000,000-iteration runs)."
    )

    parser.add_argument(
        "--init_path",
        type=str,
        default=None,
        help="Optional warm-start file produced by warm_start.py (an .npz with A, "
             "b, W, log_psi, col_names, or a plain adjacency CSV). If given, the "
             "sampler starts from that DAG instead of the empty graph. An .npz also "
             "seeds b/W/log_psi unless --init_dag_only is set. The starting graph "
             "must be a DAG that respects the prior (no forbidden edges)."
    )

    parser.add_argument(
        "--init_dag_only",
        action="store_true",
        help="With --init_path pointing to an .npz, use only its adjacency A and "
             "keep the default b/W/log_psi initial values (ignore the file's theta)."
    )

    args = parser.parse_args()

    if args.compress and not args.out_path.endswith(".zst"):
        args.out_path = args.out_path + ".zst"

    def parse_shape_vector(s, length, name, lo=0.0, hi=None):
        """Parse a comma-separated list of floats into a length-`length` vector.
        A single value is broadcast to all `length` entries. Each value must lie
        in the open interval (lo, hi); hi=None means (lo, +inf)."""
        vals = np.array(
            [float(p) for p in str(s).split(",") if p.strip() != ""],
            dtype=np.float64,
        )
        if vals.size == 1:
            vals = np.full(length, vals[0])
        elif vals.size != length:
            raise ValueError(
                f"{name} must have 1 or {length} comma-separated values; "
                f"got {vals.size} ({s!r})."
            )
        if np.any(vals <= lo) or (hi is not None and np.any(vals >= hi)):
            bound = f"in ({lo}, {hi})" if hi is not None else f"> {lo}"
            raise ValueError(f"{name} must be {bound}; got {vals}.")
        return vals

    rng = np.random.default_rng(args.seed)

    # ---- Load observations ----
    X_df = pd.read_csv(args.x_path, index_col=0)
    col_names = X_df.columns.tolist()
    X = X_df.to_numpy()
    print(X)
    print(X.shape)

    n = X.shape[1]

    # Bipartite split: TFs (no chr prefix) vs enhancers (chr prefix)
    set1 = [i for i, c in enumerate(col_names) if not c.startswith("chr")]  # TFs
    set2 = [i for i, c in enumerate(col_names) if c.startswith("chr")]      # enhancers
    print(f"set1 (TFs, n={len(set1)}): indices {set1[:5]}...")
    print(f"set2 (enhancers, n={len(set2)}): indices {set2[:5]}...")

    # ---- Initial parameter values ----
    b        = np.full(n, 0.5)
    W        = np.zeros((n, n))
    log_psi  = np.full(n, np.log(0.5))
    alpha    = logit(0.1)             # rho_0 = 0.1 (neutral start under the flat
                                      # Beta(1,1) prior; not forcing baseline sparsity)

    # ---- Hyperparameters ----
    # Priors on theta (intercepts, weights, dispersion).
    # Note on hyper["w"]["sd"]: this is BOTH the prior sd on each edge weight
    # AND the proposal sd used when proposing a new edge weight inside a graph
    # birth move (see _graph_mh_step). Keeping the two tied gives a clean
    # cancellation in the MH acceptance ratio (the prior-as-proposal RJ form).
    # All three are set from the CLI (--prior_w_sd / --prior_b_sd /
    # --prior_psi_a / --prior_psi_b) so a run's priors are visible in its config
    # rather than hard-coded here; the defaults reproduce the previous values.
    #
    # The default sd_w = 0.05 is a weakly informative scale for regulatory effects
    # on the raw-X linear predictor, avoiding the severe Occam over-penalty that
    # sd_w = 0.3 imposed (~4.6 nats/edge), which was collapsing the real-data graph
    # to a near-empty DAG. It is calibrated to the real data, NOT to the simulation
    # study, whose data are generated with --coef-sd 0.3: a run on simulated data
    # should pass a --prior_w_sd on that scale, or w_ij comes back shrunk toward 0
    # with badly under-covering credible intervals.
    hyper = {
        "b":   {"sd": args.prior_b_sd},
        "w":   {"sd": args.prior_w_sd},
        "psi": {"a": args.prior_psi_a, "b": args.prior_psi_b},
    }

    # Priors on graph hyperparameters. rho_0 carries a Beta(a, b) prior; each
    # source edge probability rho_k carries its own fixed Beta(a_k, b_k) prior.
    # The per-source shape vectors are attached below once n_sources is known.
    prior_hyper = {
        "rho": {"a": args.prior_rho_a, "b": args.prior_rho_b},
    }

    # ---- Load evidence matrix Q ----
    # Q must be an integer-coded square matrix with row and column names.
    # Encoding: -1 = forbidden (same-type pair), 0 = no evidence, k>=1 = evidence level k.
    Q_df = pd.read_csv(args.q_path, index_col=0)
    if list(Q_df.index) != col_names or list(Q_df.columns) != col_names:
        raise ValueError(
            "Row and column names of Q must match the column names of X in the same order.\n"
            f"  X columns : {col_names}\n"
            f"  Q rows    : {list(Q_df.index)}\n"
            f"  Q columns : {list(Q_df.columns)}"
        )
    Q = Q_df.to_numpy()
    print("Q:")
    print(Q)
    print(Q.shape)

    # Q must be integer-coded: -1=forbidden, 0=no evidence, k>=1=evidence level k.
    if not np.all(Q == Q.astype(np.int64)):
        raise ValueError("Q must be integer-coded (-1, 0, 1, 2, ...). Got float values.")
    src_idx = Q.astype(np.int64)
    n_sources = int(Q.max())

    # Attach the per-source Beta shape vectors for rho_k now that K is known.
    # Each input may be a single value (broadcast to all K sources) or a list of
    # K. Under the "mean_conc" parameterization the (mean, concentration) inputs
    # are converted to shape parameters: a_k = mean_k * conc_k,
    # b_k = (1 - mean_k) * conc_k.
    if args.prior_rho_k_param == "mean_conc":
        prior_rho_k_mean = parse_shape_vector(
            args.prior_rho_k_mean, n_sources, "--prior_rho_k_mean", lo=0.0, hi=1.0)
        prior_rho_k_conc = parse_shape_vector(
            args.prior_rho_k_conc, n_sources, "--prior_rho_k_conc", lo=0.0)
        prior_rho_k_a = prior_rho_k_mean * prior_rho_k_conc
        prior_rho_k_b = (1.0 - prior_rho_k_mean) * prior_rho_k_conc
    else:
        prior_rho_k_a = parse_shape_vector(args.prior_rho_k_a, n_sources, "--prior_rho_k_a")
        prior_rho_k_b = parse_shape_vector(args.prior_rho_k_b, n_sources, "--prior_rho_k_b")
    prior_hyper["rho_k"] = {"a": prior_rho_k_a, "b": prior_rho_k_b}

    # Starting DAG. By default start empty — the prior on edges is expressed
    # through src_idx and the rho parameters, not the initial graph. If a warm
    # start is supplied (see warm_start.py), load and validate it instead; an
    # .npz warm start also seeds b/W/log_psi unless --init_dag_only is set.
    if args.init_path is None:
        A = np.zeros((n, n), dtype=np.int8)
    else:
        A, b_ws, W_ws, log_psi_ws = load_warm_start(
            args.init_path, col_names, src_idx,
            dag_only=args.init_dag_only,
            b_default=b, W_default=W, log_psi_default=log_psi,
        )
        if b_ws is not None:       b = b_ws
        if W_ws is not None:       W = W_ws
        if log_psi_ws is not None: log_psi = log_psi_ws
        print(f"Loaded warm-start DAG from {args.init_path}: "
              f"{int(A.sum())} edges"
              f"{' (adjacency only)' if args.init_dag_only else ' (+ b/W/log_psi)'}.")

    # Initial value for rho_k (matches old initial q_ij for source edges).
    rho_init   = 0.1
    rho_logits = np.full(n_sources, logit(rho_init))

    print("src_idx:")
    print(src_idx)
    print(f"n_sources: {n_sources}")
    print(f"rho_logits  (length {len(rho_logits)}): {rho_logits}")
    print(f"prior rho_k parameterization: {args.prior_rho_k_param}")
    if args.prior_rho_k_param == "mean_conc":
        print(f"prior rho_k mean (length {len(prior_rho_k_mean)}): {prior_rho_k_mean}")
        print(f"prior rho_k conc (length {len(prior_rho_k_conc)}): {prior_rho_k_conc}")
    print(f"prior rho_k a (length {len(prior_rho_k_a)}): {prior_rho_k_a}")
    print(f"prior rho_k b (length {len(prior_rho_k_b)}): {prior_rho_k_b}")
    print(f"prior rho_k implied mean (a/(a+b)): {prior_rho_k_a / (prior_rho_k_a + prior_rho_k_b)}")

    # ---- Run MCMC ----
    checkpoint_path = args.out_path.replace(".pkl", "_checkpoint.pkl")
    (samples, logpost_cold, logpost_all,
     edge_counts_all, diagnostics) = run_mcmc_parallel_tempering(
            X,
            A,
            b,
            W,
            log_psi,
            src_idx,
            alpha,
            rho_logits,
            hyper,
            prior_hyper,
            rng,
            n_iter=args.n_iter,
            n_chains=args.n_chains,
            factor=args.factor,
            sigma_b=args.sigma_b,
            sigma_w=args.sigma_w,
            sigma_logpsi=args.sigma_logpsi,
            thin=args.thin,
            p_exchange_total=0.2,
            parallel_chains=True,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=args.checkpoint_interval,
            col_names=col_names,
            compress=args.compress,
            compress_level=args.compress_level,
        )

    print("Collected", len(samples.graphs), "posterior samples from the cold chain.")

    # ---- Timing row -------------------------------------------------------
    # One row per run in the same header-less five-column format dag0.R writes
    # (run, nodes, iterations, chains, seconds), so the two methods' costs
    # aggregate the same way. `seconds` is the sampling loop alone: python
    # startup, reading X and Q, the checkpoint writes, and the final pickle are
    # all excluded, matching what dag0.R times.
    # --compress has already appended ".zst" to args.out_path by this point, so
    # strip both suffixes to get the run's base name.
    out_base = args.out_path
    for suffix in (".zst", ".pkl"):
        if out_base.endswith(suffix):
            out_base = out_base[: -len(suffix)]
    timing_path = out_base + "_timing.csv"
    os.makedirs(os.path.dirname(os.path.abspath(timing_path)), exist_ok=True)
    with open(timing_path, "w") as fh:
        fh.write("%s,%d,%d,%d,%.3f\n" % (
            os.path.basename(out_base),
            int(A.shape[0]),
            args.n_iter,
            args.n_chains,
            diagnostics["sampler_seconds"],
        ))
    print(f"Wrote timing to {timing_path}")

    # ---- Save results ----
    out = {
        "graphs_samples":      samples.graphs,
        "b_samples":           samples.b,
        "W_samples":           samples.W,
        "log_psi_samples":     samples.log_psi,
        "alpha_samples":       samples.alpha,
        "rho_logits_samples":  samples.rho_logits,
        "logpost_trace":       logpost_cold,
        "logpost_trace_all":   logpost_all,
        "edge_counts_all":     edge_counts_all,
        "diagnostics":         diagnostics,
        "n_nodes":             int(A.shape[0]),
        "col_names":           col_names,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    _save_pickle(out, args.out_path,
                 compress=args.compress, compress_level=args.compress_level)