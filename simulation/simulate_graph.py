#!/usr/bin/env python3
"""
simulate_graph.py
=================
Simulates the datasets of the simulation study: ground-truth DAGs drawn from the
topological prior, and count data from a negative binomial structural equation model on
each. The defaults are the study's settings.

Reproducing data/simulation/ exactly (byte for byte), from the simulation/ directory:

    python build_chr2_subset.py   # writes data/simulation/prior/ (prior_matrix.csv, node_order.csv)
    python simulate_graph.py      # reads that prior; writes data/simulation/datasets/ and
                                  # data/simulation/prior/prior_matrix_nopartition.csv

The order matters: simulate_graph.py reads the prior build_chr2_subset.py writes. Run
both without arguments; the defaults (seed 7, 50 observations, generating edge
probabilities 0.286 / 0.606 / 0.05) are the study's settings.

1. Map each evidence code of the topological prior (data/simulation/prior/prior_matrix.csv)
   to a generating edge probability: -1 (forbidden) -> 0, 0 (baseline) -> --baseline-prob,
   and the evidence classes via --prob-map (1 JASPAR2024 CORE -> 0.286, 2 PCHi-C -> 0.606,
   the expert-elicited prior means).
2. Draw DAGs: every entry A[i, j] ~ Bernoulli(Q[i, j]) independently, diagonal zeroed,
   cyclic graphs rejected, until --max-dags acyclic graphs are found.
3. Draw the regression parameters shared by every DAG: intercepts a_i ~ N(mu_log_mean,
   mu_log_sd^2) and coefficients B[i, j] ~ N(0, coef_sd^2).
4. For each DAG in turn, draw --n-samples observations, nodes in topological order:
       log(mu_i) = a_i + sum_{j -> i} B[j, i] * X_j,   X_i ~ NegBin(mu_i, theta)
   (overdispersion psi = 1 / theta). A DAG whose mean exceeds 1e7 at any node is skipped;
   the first --n-dags that simulate are kept.

Writes to data/simulation/datasets/:
  X_<i>.csv : counts for DAG i (rows = observations, columns = nodes)
  A_<i>.csv : adjacency matrix of DAG i (A[i, j] = 1 means i -> j)
  Q.csv     : generating edge-probability matrix
  b.csv     : intercepts a_i
  W.csv     : coefficient matrix B, before masking by each DAG
  X.csv     : every dataset, concatenated
and data/simulation/prior/prior_matrix_nopartition.csv, the prior with every evidence
class merged into the baseline class (the "no evidence classes" setting).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(REPO, "spark"))
from graph import has_cycle  # noqa: E402

PRIOR_DIR = os.path.join(REPO, "data", "simulation", "prior")


def parse_args():
    p = argparse.ArgumentParser(
        description="Simulate the DAGs and count data of the simulation study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", default=os.path.join(PRIOR_DIR, "prior_matrix.csv"),
                   help="Integer-coded topological prior.")
    p.add_argument("--output-dir", default=os.path.join(REPO, "data", "simulation", "datasets"))
    p.add_argument("--n-dags", type=int, default=10, help="Number of datasets to keep.")
    p.add_argument("--max-dags", type=int, default=100, help="Candidate DAGs to draw.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-samples", type=int, default=50, help="Observations per dataset.")
    p.add_argument("--theta", type=float, default=10.0, help="NB dispersion (psi = 1/theta).")
    p.add_argument("--mu-log-mean", type=float, default=np.log(2))
    p.add_argument("--mu-log-sd", type=float, default=1.0)
    p.add_argument("--coef-sd", type=float, default=0.3)
    p.add_argument("--baseline-prob", type=float, default=0.05,
                   help="Generating edge probability of the baseline class (code 0).")
    p.add_argument("--prob-map", default="1:0.286,2:0.606",
                   help="Generating edge probability of each evidence class, 'code:prob,...'.")
    return p.parse_args()


def prob_matrix(codes, baseline_prob, prob_map):
    """Map every integer code to its generating edge probability."""
    probs = {-1: 0.0, 0: baseline_prob}
    for item in prob_map.split(","):
        code, prob = item.split(":")
        probs[int(code)] = float(prob)
    unknown = set(np.unique(codes.values)) - set(probs)
    if unknown:
        raise ValueError(f"no probability for code(s) {sorted(unknown)}; add them to --prob-map")
    return codes.replace(probs).astype(float)


def generate_dags(Q, k, seed, max_tries=1_000_000):
    """Draw adjacency matrices A[i, j] ~ Bernoulli(Q[i, j]) until k acyclic ones are found."""
    rng = np.random.default_rng(seed)
    P = np.asarray(Q, dtype=float)
    dags, tries = [], 0
    while len(dags) < k and tries < max_tries:
        tries += 1
        A = (rng.random(P.shape) < P).astype(np.int8)
        np.fill_diagonal(A, 0)
        if not has_cycle(A):
            dags.append(A)
    if len(dags) < k:
        raise RuntimeError(f"only {len(dags)} acyclic graph(s) in {tries} tries")
    return dags


def topological_order(A):
    """Node indices in topological order (Kahn's algorithm); A[i, j] = 1 means i -> j."""
    A = np.asarray(A, dtype=int)
    in_degree = A.sum(axis=0).astype(int).copy()
    queue = [i for i in range(A.shape[0]) if in_degree[i] == 0]
    order = []
    while queue:
        node = queue.pop()
        order.append(node)
        for child in np.where(A[node, :] == 1)[0]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
    if len(order) != A.shape[0]:
        raise ValueError("graph contains a cycle")
    return order


def simulate_counts(A, a, B, n_samples, theta, rng):
    """Draw n_samples observations from the NB SEM on DAG A with intercepts a and
    coefficients B (already masked to A's edges), nodes in topological order."""
    A = np.asarray(A, dtype=int)
    X = np.zeros((n_samples, A.shape[0]), dtype=int)
    for i in topological_order(A):
        parents = np.where(A[:, i] == 1)[0]
        log_mu = np.full(n_samples, a[i]) + (X[:, parents] @ B[parents, i] if parents.size > 0 else 0.0)
        mu = np.exp(log_mu)
        if np.any(mu > 1e7):
            raise RuntimeError(f"mean exceeded 1e7 at node {i} (max {mu.max():.2f})")
        X[:, i] = rng.negative_binomial(theta, theta / (theta + mu))
    return X


def main(args):
    codes = pd.read_csv(args.input, index_col=0)
    Q = prob_matrix(codes, args.baseline_prob, args.prob_map)
    nodes = list(Q.index)

    candidates = generate_dags(Q, args.max_dags, args.seed)

    # Shared parameters: every dataset differs from the others in its graph only.
    rng = np.random.default_rng(args.seed)
    a = rng.normal(args.mu_log_mean, args.mu_log_sd, size=len(nodes))
    B = rng.normal(0.0, args.coef_sd, size=(len(nodes), len(nodes)))
    np.fill_diagonal(B, 0.0)

    dags, datasets = [], []
    for g, A in enumerate(candidates):
        try:
            X = simulate_counts(A, a, B * A, args.n_samples, args.theta, rng)
        except RuntimeError as e:
            print(f"skipping DAG {g}: {e}")
            continue
        dags.append(A)
        datasets.append(pd.DataFrame(X, columns=nodes))
        if len(dags) == args.n_dags:
            break
    if len(dags) < args.n_dags:
        raise RuntimeError(f"only {len(dags)} of {args.n_dags} simulations succeeded")

    out = args.output_dir
    os.makedirs(out, exist_ok=True)
    for g, (A, df) in enumerate(zip(dags, datasets)):
        df.to_csv(os.path.join(out, f"X_{g}.csv"), index=True)
        pd.DataFrame(A).to_csv(os.path.join(out, f"A_{g}.csv"), index=False, header=False)
    Q.to_csv(os.path.join(out, "Q.csv"), index=False, header=False)
    pd.Series(a).to_csv(os.path.join(out, "b.csv"), index=False, header=False)
    pd.DataFrame(B).to_csv(os.path.join(out, "W.csv"), index=False, header=False)
    pd.concat(datasets, ignore_index=True).to_csv(os.path.join(out, "X.csv"), index=True)

    nopart = codes.copy()
    nopart[nopart > 0] = 0
    nopart.to_csv(os.path.join(os.path.dirname(args.input), "prior_matrix_nopartition.csv"))
    print(f"{len(dags)} datasets of {args.n_samples} observations x {len(nodes)} nodes "
          f"written to {out}")


if __name__ == "__main__":
    main(parse_args())
