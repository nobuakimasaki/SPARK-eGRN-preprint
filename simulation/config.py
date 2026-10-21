"""
config.py
=========
Paths, settings, colours and loaders shared by the simulation-study scripts.

Node ordering
-------------
The matrices on disk (prior_matrix.csv, A_<i>.csv, posterior_dir_sim<i>.npy) are all in
the node order of node_order.csv. The heatmaps display them instead in DISPLAY order: TFs,
then downstream genes, then enhancers, each block sorted by genomic position, which puts
the TF -> enhancer block and the enhancer -> gene block in opposite corners.

Evidence classes of the chr2 prior: 0 baseline, 1 JASPAR2024 CORE (TF -> enhancer),
2 PCHi-C (enhancer -> gene); -1 is a forbidden pair.
"""

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO  = os.path.dirname(_HERE)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SIM_DATA   = os.path.join(REPO, "data", "simulation")
SIM_DIR    = os.path.join(SIM_DATA, "datasets")     # X_<i>, A_<i>, W, b (simulate_graph.py)
PRIOR_DIR  = os.path.join(SIM_DATA, "prior")        # prior_matrix, node_order
MCMC_DIR   = os.path.join(SIM_DATA, "mcmc")         # <setting>_DAG<i>.pkl.zst (spark/run_sim.slurm)
EVAL_BASE  = os.path.join(_HERE, "output", "eval")  # written by eval.py
TABLE_DIR  = os.path.join(_HERE, "output", "tables")
FIG_DIR    = os.path.join(REPO, "paper", "figures")
# Predictions of the two comparator methods (benchmarking/dag0.R, benchmarking/regression.py).
DAG0_DIR   = os.path.join(REPO, "benchmarking", "output", "dag0")
REG_DIR    = os.path.join(REPO, "benchmarking", "output", "regression")

PRIOR_PATH      = os.path.join(PRIOR_DIR, "prior_matrix.csv")
NODE_ORDER_PATH = os.path.join(PRIOR_DIR, "node_order.csv")

# ---------------------------------------------------------------------------
# Study settings
# ---------------------------------------------------------------------------
N_DAGS = 10
# Burn-in in iterations (the runs store every iteration).
BURN_IN = 150_000
# PIP at which a pair counts as an inferred edge.
CALL_THRESHOLD = 0.5

# The four prior settings of the paper, in the order of Figure 2E and the tables.
SETTINGS = ["expert_elicited", "noninfo", "wrong_extreme", "no_partition"]
SETTING_LABELS = {
    "noninfo":         "Non-informative",
    "expert_elicited": "Expert-elicited",
    "wrong_extreme":   "Strongly informative,\nmisspecified",
    "no_partition":    "Single class",
}
SETTING_LABELS_FLAT = {k: v.replace("\n", " ") for k, v in SETTING_LABELS.items()}

# The comparator methods, after SPARK's settings in Figure 2E and Table S2.
COMPARATORS = ["dag0", "regression"]
METHOD_SHORT_LABELS = {
    "expert_elicited": "Expert-elicited",
    "noninfo":         "Non-informative",
    "wrong_extreme":   "Strongly informative,\nmisspecified",
    "no_partition":    "Single class",
    "dag0":            "DAG0",
    "regression":      "Regularized\nregression",
}
# The regression calls a pair at p < 0.001, the default cutoff of CellOracle.
REG_CALL_P = 0.001

# Generating edge probabilities of the classes (simulate_graph.py's defaults).
TRUE_RHO = {0: 0.05, 1: 0.286, 2: 0.606}
# Generating overdispersion: counts are NB with dispersion theta = 10, so psi_j = 0.1.
TRUE_PSI = 0.1

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
FORBIDDEN_COLOR = "#FFFFFF"     # code -1, a pair no edge may occupy
BASELINE_COLOR  = "#DDE3EA"     # code  0, permitted but unsupported
# Background for a permitted pair that is not an edge, in the panels that draw realized
# edges on top: the same grey, so a permitted pair reads the same in every panel.
CANDIDATE_COLOR = BASELINE_COLOR
EVIDENCE_CLASSES = {
    0: {"name": "No evidence",      "color": BASELINE_COLOR},
    1: {"name": "JASPAR2024 motif", "color": "#E8871A"},
    2: {"name": "PCHi-C contact",   "color": "#7B52D8"},
}
# A realized edge in the baseline class, darker than the baseline background.
BASELINE_EDGE_COLOR = "#4A5568"
# Node kinds, as in Figure 1 and the real-data figures.
KIND_COLOR = {"TF": "#DC143C", "gene": "#2CA02C", "enhancer": "#3B7DD8"}
# Parameter recovery plots: one colour for every mark, the reference line and the CrIs.
POINT_COLOR = "#1F4E9C"
REF_COLOR   = "#555555"
CI_COLOR    = "#cccccc"

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_nodes():
    """node_order.csv as a DataFrame with columns [node, kind, pos], in the row and
    column order of every matrix on disk."""
    return pd.read_csv(NODE_ORDER_PATH)


def display_order(nodes=None):
    """Permutation grouping the nodes by kind, TFs then downstream genes then enhancers,
    each block sorted by genomic position. Returns (order, n_tf, n_reg): the on-disk
    indices, the index at which the gene block starts, and the index at which the
    enhancer block starts."""
    if nodes is None:
        nodes = load_nodes()
    idx = np.arange(len(nodes))
    pos = nodes["pos"].to_numpy()
    kind = nodes["kind"].to_numpy()
    blocks = []
    for k in ("TF", "gene", "enhancer"):
        sel = idx[kind == k]
        blocks.append(sel[np.argsort(pos[sel], kind="mergesort")])
    n_tf, n_gene = len(blocks[0]), len(blocks[1])
    return np.concatenate(blocks), n_tf, n_tf + n_gene


def reorder(M, order):
    """Apply a display permutation to both axes of a square matrix."""
    return np.asarray(M)[np.ix_(order, order)]


def load_prior_codes():
    """The chr2 topological prior as an (n, n) integer matrix, in on-disk node order."""
    return pd.read_csv(PRIOR_PATH, index_col=0).to_numpy().astype(int)


def load_true_adjacency(sim):
    """Ground-truth adjacency of DAG `sim` as an (n, n) 0/1 integer matrix."""
    return np.loadtxt(f"{SIM_DIR}/A_{sim}.csv", delimiter=",").astype(int)


def load_posterior(setting, sim):
    """PIPs of DAG `sim` under `setting` (eval.py), in on-disk node order."""
    return np.load(f"{EVAL_BASE}/{setting}/posterior_dir_sim{sim}.npy").astype(float)


def load_comparator(method, sim):
    """Prediction of a comparator method for DAG `sim`, in on-disk node order: DAG0's
    PIPs, or the regression's -log p."""
    if method == "regression":
        return np.load(f"{REG_DIR}/reg_DAG{sim}.npy").astype(np.float64)
    df = pd.read_csv(f"{DAG0_DIR}/dag0_DAG{sim}_post.csv", index_col=0)
    # R writes the node names sanitized (chr2-28391547-... -> chr2.28391547...), so check
    # the order after the same substitution: the matrices are aligned by position.
    expected = [re.sub(r"[^0-9A-Za-z_.]", ".", s) for s in load_nodes()["node"]]
    if list(df.index) != expected or list(df.columns) != expected:
        raise ValueError(f"dag0_DAG{sim}_post.csv is not in the node order of the prior")
    return df.to_numpy(dtype=np.float64)


def load_prediction(method, sim):
    """Score of every ordered pair for DAG `sim`: SPARK's PIPs under a prior setting, or a
    comparator's prediction."""
    return load_comparator(method, sim) if method in COMPARATORS else load_posterior(method, sim)


def load_rho(setting, sim):
    """Retained draws of the class edge probabilities (S, K+1) [rho_0, ..., rho_K]."""
    return np.load(f"{EVAL_BASE}/{setting}/rho_samples_sim{sim}.npy")


def savefig(fig, name, dpi=300):
    """Write `fig` to FIG_DIR/`name` and close it."""
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
    return path
