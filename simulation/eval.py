"""
eval.py
=======
Posterior summaries of the simulation study. For every prior setting and ground-truth DAG
it loads the sampler output (config.MCMC_DIR/<setting>_DAG<i>.pkl.zst), discards the first
config.BURN_IN iterations, and writes to config.EVAL_BASE/<setting>/:

  posterior_dir_sim<i>.npy : (n, n) PIPs, the share of retained draws containing each edge
  rho_samples_sim<i>.npy   : (S, K+1) retained draws [rho_0, rho_1, ..., rho_K]
  edge_recovery.csv        : one row per (DAG, true edge): generating coefficient true_w,
                             PIP, and the posterior median and 95% CrI of the coefficient
                             CONDITIONAL on the edge being present (W is stored masked to
                             the sampled graph, so the marginal would be biased toward 0)
  b_recovery.csv           : one row per (DAG, node): generating intercept, posterior
                             median and 95% CrI
  psi_recovery.csv         : one row per (DAG, node): generating overdispersion, posterior
                             median and 95% CrI
  performance.csv          : one row per DAG: retained draws, true edges, prevalence of true
                             edges among the permitted pairs, and PR-AUC

PR-AUC scores the permitted pairs (off-diagonal and not forbidden by the prior) by their
PIP. Forbidden pairs are fixed to absent by the prior, so they are excluded.

Sample storage: the sampler stores each draw's graph as an (E_s, 2) array of
(parent, child) edges and its coefficients as the matching (E_s,) array, so PIPs and
coefficient draws are gathered from these edge lists directly.

Usage
-----
  python eval.py
"""

import os
import gc
import pickle

import numpy as np
import pandas as pd
from scipy.special import expit

import config as cfg

# Minimum number of draws in which a true edge must be present for its conditional
# coefficient posterior to be summarized; fewer -> NaN.
W_MIN_PRESENT_DRAWS = 10


def average_precision(y_true, y_score):
    """Area under the precision-recall curve (average precision), matching
    sklearn.metrics.average_precision_score (step interpolation, tie-aware)."""
    y_true  = np.asarray(y_true,  dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)

    order = np.argsort(-y_score, kind="mergesort")
    y_true  = y_true[order]
    y_score = y_score[order]

    # Last index of each group of tied scores (where precision/recall are read).
    end = np.r_[np.where(np.diff(y_score))[0], y_true.size - 1]
    tp  = np.cumsum(y_true)[end]
    fp  = (end + 1) - tp

    n_pos = tp[-1]
    if n_pos == 0:
        return float("nan")
    precision = tp / (tp + fp)
    recall    = tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def median_ci(x, cred=0.95):
    """Posterior median and central credible interval of a 1-D sample, as
    (median, lower, upper). NaNs for an empty sample."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    lo_q = 100.0 * (1.0 - cred) / 2.0
    return (float(np.percentile(x, 50)),
            float(np.percentile(x, lo_q)),
            float(np.percentile(x, 100.0 - lo_q)))


def load_chain(path):
    import zstandard as zstd
    with open(path, "rb") as f:
        return pickle.load(zstd.ZstdDecompressor().stream_reader(f))


def evaluate_run(sim, path, out_dir, Q, W_true, b_true):
    """Summarize one run; save its per-DAG arrays and return (row, edge_df, b_df, psi_df)."""
    print(f"{os.path.basename(path)}", flush=True)
    res = load_chain(path)
    edge_lists = res["graphs_samples"][cfg.BURN_IN:]
    w_lists    = res["W_samples"][cfg.BURN_IN:]
    S = len(edge_lists)
    n = int(res["n_nodes"])
    A_true = np.loadtxt(f"{cfg.SIM_DIR}/A_{sim}.csv", delimiter=",").astype(np.int8)

    # --- PIPs: one bincount over every draw's edges, by flat index i*n + j ---
    flat_all = [e[:, 0].astype(np.int64) * n + e[:, 1] for e in edge_lists if e.size]
    counts = (np.bincount(np.concatenate(flat_all), minlength=n * n)
              if flat_all else np.zeros(n * n, dtype=np.int64))
    post = (counts.reshape(n, n) / S).astype(np.float64)
    np.save(f"{out_dir}/posterior_dir_sim{sim}.npy", post)

    # --- PR-AUC over the permitted pairs ---
    cand = ~np.eye(n, dtype=bool) & (Q != -1)
    y_true  = A_true[cand].ravel().astype(int)
    pr_auc = average_precision(y_true, post[cand].ravel())

    # --- Class edge probabilities: rho_0 = expit(alpha), rho_k = expit(rho_logits) ---
    rho_logits = np.asarray(res["rho_logits_samples"][cfg.BURN_IN:], dtype=np.float64)
    alpha      = np.asarray(res["alpha_samples"][cfg.BURN_IN:], dtype=np.float64)
    np.save(f"{out_dir}/rho_samples_sim{sim}.npy",
            np.hstack([expit(alpha)[:, None], expit(rho_logits)]))

    # --- Edge coefficients at the true edges, conditional on the edge being present ---
    # Each draw stores coefficients only for its own edges, so they are scattered into a
    # reusable flat buffer, gathered at the true edges, and the touched slots cleared.
    src, dst = np.where(A_true == 1)
    n_edges_true = src.size
    true_flat = src.astype(np.int64) * n + dst
    w_draws = np.zeros((S, n_edges_true), dtype=np.float64)
    present = np.zeros((S, n_edges_true), dtype=bool)
    w_buf    = np.zeros(n * n, dtype=np.float64)
    seen_buf = np.zeros(n * n, dtype=bool)
    for si in range(S):
        e = edge_lists[si]
        if e.size == 0:
            continue
        f = e[:, 0].astype(np.int64) * n + e[:, 1]
        w_buf[f]    = np.asarray(w_lists[si], dtype=np.float64)
        seen_buf[f] = True
        w_draws[si] = w_buf[true_flat]
        present[si] = seen_buf[true_flat]
        w_buf[f]    = 0.0
        seen_buf[f] = False
    n_present = present.sum(axis=0).astype(int)
    w_med = np.full(n_edges_true, np.nan)
    w_lo  = np.full(n_edges_true, np.nan)
    w_hi  = np.full(n_edges_true, np.nan)
    for e in range(n_edges_true):
        if n_present[e] >= W_MIN_PRESENT_DRAWS:
            w_med[e], w_lo[e], w_hi[e] = median_ci(w_draws[present[:, e], e])
    edge_df = pd.DataFrame({
        "sim": sim, "parent": src, "child": dst, "true_w": W_true[src, dst],
        "post_prob": post[src, dst], "w_post_median": w_med,
        "w_post_lower95": w_lo, "w_post_upper95": w_hi,
    })

    # --- Node intercepts and overdispersions ---
    b_draws   = np.asarray(res["b_samples"][cfg.BURN_IN:], dtype=np.float64)
    psi_draws = np.exp(np.asarray(res["log_psi_samples"][cfg.BURN_IN:], dtype=np.float64))
    b_rows, psi_rows = [], []
    for node in range(n):
        bm, blo, bhi = median_ci(b_draws[:, node])
        b_rows.append({"sim": sim, "node": node, "b_true": float(b_true[node]),
                       "b_post_median": bm, "b_post_lower95": blo, "b_post_upper95": bhi})
        pm, plo, phi = median_ci(psi_draws[:, node])
        psi_rows.append({"sim": sim, "node": node, "psi_true": cfg.TRUE_PSI,
                         "psi_post_median": pm, "psi_post_lower95": plo,
                         "psi_post_upper95": phi})

    del res, edge_lists, w_lists, w_draws, present, b_draws, psi_draws
    gc.collect()
    row = {"sim": sim, "n_post_draws": S, "n_true_edges": int(y_true.sum()),
           "prevalence": float(y_true.mean()), "pr_auc": pr_auc}
    return row, edge_df, pd.DataFrame(b_rows), pd.DataFrame(psi_rows)


def main():
    Q = cfg.load_prior_codes()
    W_true = np.loadtxt(f"{cfg.SIM_DIR}/W.csv", delimiter=",")
    b_true = np.loadtxt(f"{cfg.SIM_DIR}/b.csv", delimiter=",").ravel()
    for setting in cfg.SETTINGS:
        out_dir = f"{cfg.EVAL_BASE}/{setting}"
        os.makedirs(out_dir, exist_ok=True)
        results = [evaluate_run(sim, f"{cfg.MCMC_DIR}/{setting}_DAG{sim}.pkl.zst",
                                out_dir, Q, W_true, b_true)
                   for sim in range(cfg.N_DAGS)]
        pd.DataFrame([r for r, _, _, _ in results]).to_csv(
            f"{out_dir}/performance.csv", index=False)
        pd.concat([e for _, e, _, _ in results], ignore_index=True).to_csv(
            f"{out_dir}/edge_recovery.csv", index=False)
        pd.concat([b for _, _, b, _ in results], ignore_index=True).to_csv(
            f"{out_dir}/b_recovery.csv", index=False)
        pd.concat([p for _, _, _, p in results], ignore_index=True).to_csv(
            f"{out_dir}/psi_recovery.csv", index=False)
        print(f"{setting}: mean PR-AUC "
              f"{np.mean([r['pr_auc'] for r, _, _, _ in results]):.3f}", flush=True)


if __name__ == "__main__":
    main()
