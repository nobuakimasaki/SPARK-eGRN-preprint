"""
config.py
=========
Paths and analysis settings shared by the real-data post-processing scripts.

All paths are relative to the repository, so the scripts run from any working directory.
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO  = os.path.dirname(_HERE)

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(REPO, "data", "real_data")

# MCMC output of spark/run.slurm: the expert-elicited run on the M-CSF 16h dataset.
CHAIN_PATH = os.path.join(DATA_DIR, "mcmc", "expert_elicited_chain.pkl.zst")

# Topological prior (377 x 377, integer codes; see CODE_LOOKUP) and node annotation table
# (kind, is_tf, is_gwas_anchor, is_gene_near_anchor, chr, pos, ...).
PRIOR_PATH = os.path.join(DATA_DIR, "vm_m16h_b1_prior_matrix.csv")
NODES_PATH = os.path.join(DATA_DIR, "vm_m16h_b1_nodes.csv")

# Immune disease GWAS annotation of the immGWAS enhancers: rp_id -> study
# (gwas_variants_immune.csv) -> trait_id / trait_label (study_trait_map.csv), keeping only
# trait_ids flagged as immune diseases (efo_trait_sets.csv).
GWAS_PATH        = os.path.join(DATA_DIR, "gwas", "gwas_variants_immune.csv")
STUDY_TRAIT_PATH = os.path.join(DATA_DIR, "gwas", "study_trait_map.csv")
EFO_TRAIT_PATH   = os.path.join(DATA_DIR, "gwas", "efo_trait_sets.csv")

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
OUT_DIR  = os.path.join(_HERE, "output")
EVAL_DIR = os.path.join(OUT_DIR, "eval")      # posterior summaries written by eval.py
FIG_DIR  = os.path.join(REPO, "paper", "figures")   # figures included by the paper

# ---------------------------------------------------------------------------
# Analysis window (stored draws; the run stores every iteration). The paper discards the
# first 3,000,000 iterations as burn-in and keeps every 10th cold-chain draw of the
# remaining 2,250,000, giving S = 225,000 draws.
# ---------------------------------------------------------------------------
BURN_IN  = 3_000_000
WIN_END  = 5_250_000
THIN     = 10

# Posterior inclusion probability at which an edge is called (Results, figures, tables).
PIP_CUT = 0.5

# Prior evidence code -> source name. 1 and 2 support TF -> enhancer pairs, 3 and 4
# enhancer -> gene pairs; 0 is the baseline class and -1 a forbidden pair.
CODE_LOOKUP = {1: "ChIP-Atlas", 2: "JASPAR2024_CORE",
               3: "Linear proximity", 4: "PCHi-C_high_confidence"}
