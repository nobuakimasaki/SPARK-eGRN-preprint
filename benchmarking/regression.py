#!/usr/bin/env python3
"""
Regularized-regression comparator for the simulation study (paper Section
"Benchmarking against other methods"): CellOracle's network-inference regression,
with the topological prior in place of CellOracle's base GRN.

Each step follows the CellOracle source:
  transform   log2(1 + y)                      (oracle_core.import_anndata_as_raw_count)
  scaling     divide each predictor by its SD, without centring
              (utility.standard: StandardScaler(with_mean=False))
  design      predictors scaled, response not  (regression_models.get_bayesian_ridge_coefs)
  estimator   BayesianRidge() with all defaults
  score       p = 2 * Phi(0; |coef mean|, coef sd), scored by -log p (natural log)
              (net_core._get_stats_df_from_bayesian_ridge)

Every node is regressed on its permitted regulators: the rows i with T[i, j] != -1 are
the permitted parents of node j (T[i, j] refers to the edge i -> j, as A_<i>.csv does).
The score is computed through norm.logcdf, so it stays finite where CellOracle's
-np.log(norm.cdf(...) * 2) overflows to inf; it is identical wherever CellOracle's is
finite.

Writes output/regression/reg_DAG<i>.npy: an (n, n) matrix of -log p for every pair i -> j,
0 for the pairs T forbids.

Usage
-----
  python regression.py          # all ten simulated datasets
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(os.path.dirname(_HERE), "data", "simulation")
T_PATH = os.path.join(SIM, "prior", "prior_matrix.csv")
OUT_DIR = os.path.join(_HERE, "output", "regression")
N_DAGS = 10


def regression_scores(X, T):
    """(n, n) matrix of -log p from one BayesianRidge fit per node."""
    nodes = list(X.columns)
    if list(T.index) != nodes or list(T.columns) != nodes:
        raise ValueError("X and T must list the nodes in the same order")
    n = len(nodes)
    Tv = T.to_numpy().astype(int)

    # BayesianRidge(fit_intercept=True) centres internally, so with_mean=False changes
    # no coefficient; it is kept to stay literally identical to CellOracle's standard().
    Y = np.log2(1.0 + X.to_numpy(dtype=np.float64))
    Y_scaled = StandardScaler(with_mean=False).fit_transform(Y)

    score = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        parents = np.flatnonzero((Tv[:, j] != -1) & (np.arange(n) != j))
        if parents.size == 0:
            continue
        model = BayesianRidge()
        model.fit(Y_scaled[:, parents], Y[:, j])
        beta_bar = model.coef_                          # posterior mean
        s = np.sqrt(np.diag(model.sigma_))              # posterior sd
        # -log p = -(log 2 + logcdf(-|beta_bar| / s))
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(s > 0, np.abs(beta_bar) / s, np.inf)
        score[parents, j] = -(np.log(2.0) + norm.logcdf(-z))
    return score


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    T = pd.read_csv(T_PATH, index_col=0)
    for i in range(N_DAGS):
        X = pd.read_csv(os.path.join(SIM, "datasets", f"X_{i}.csv"), index_col=0)
        np.save(os.path.join(OUT_DIR, f"reg_DAG{i}.npy"), regression_scores(X, T))
    print(f"wrote {N_DAGS} score matrices to {OUT_DIR}")


if __name__ == "__main__":
    main()
