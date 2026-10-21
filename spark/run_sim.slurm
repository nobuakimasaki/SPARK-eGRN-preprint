#!/bin/bash
#SBATCH --job-name=mcmc_sim_chr2
#SBATCH --output=logs/mcmc_%A_%a.out
#SBATCH --error=logs/mcmc_%A_%a.err
#SBATCH --partition=hmem
#SBATCH --time=3-0:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60GB
#SBATCH --array=0-39

# NOTE: create the log dir before submitting (SLURM opens the --output/--error
# files before this script runs):   mkdir -p logs
set -euo pipefail

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-NA}"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"

# Activate conda environment
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate mcmc

# Pin BLAS/threading libraries to a single thread per process. Chains are
# parallelized across processes (max_workers=n_chains=10 on cpus-per-task=10),
# so each worker must stay single-threaded -- otherwise a worker whose matmul
# trips the BLAS multi-threading cutoff spawns many BLAS threads, and 10 workers
# x N threads thrash the 10 allocated cores, slowing every iteration ~10-50x.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "Using python: $(which python)"
python --version

# ===========================================================================
# Simulation study: 10 ground-truth DAGs x 4 prior settings on rho_k, on the
# chr2 subset (51 nodes). Array spans 0..39:
#     DAG index   = TASK % 10     (simulated dataset X_0 .. X_9)
#     setting idx = TASK / 10     (prior setting 0 .. 3)
# The sampler starts from the empty graph. Settings are those of the
# supplement's table "Sampler settings for the simulation study".
# ===========================================================================
N_DAGS=10
TASK=${SLURM_ARRAY_TASK_ID}
IDX=$(( TASK % N_DAGS ))       # which simulated dataset (ground-truth DAG)
S=$(( TASK / N_DAGS ))         # which prior setting

# Inputs, written by simulation/build_chr2_subset.py and simulation/simulate_graph.py:
#   X_PATH   : simulated count matrix, 50 cells x 51 nodes.
#   Q_ORIG   : topological prior with evidence classes 1 (JASPAR2024 CORE) and
#              2 (PCHi-C), -1 forbidden, 0 baseline.
#   Q_NOPART : the same prior with every evidence class merged into the baseline
#              class (the no_partition setting).
SIM=../data/simulation
X_PATH=$SIM/datasets/X_${IDX}.csv
Q_ORIG=$SIM/prior/prior_matrix.csv
Q_NOPART=$SIM/prior/prior_matrix_nopartition.csv

# ---- MCMC run parameters --------------------------------------------------
N_ITER=250000
N_CHAINS=10
THIN=1
SEED=31
# 10 chains spanning beta in [0.40, 1]: factor^9 = 1/0.40 gives beta_9 = 0.40.
FACTOR=$(python3 -c "print((1/0.40)**(1/9))")
# Per-node Metropolis-within-Gibbs step sizes for (b, W, log_psi).
SIGMA_B=0.25
SIGMA_W=0.075
SIGMA_LOGPSI=0.25

# rho_0 (baseline edge probability) prior: flat Beta(1,1) in every setting.
PRIOR_RHO_A=1.0
PRIOR_RHO_B=1.0

# ---- Regression parameter priors ------------------------------------------
# w_ij ~ N(0, 0.2^2): same order as the generating scale (0.3) but deliberately not
# equal to it. b_j ~ N(0, 2^2). psi_j ~ Gamma(1, 2) (shape, rate).
PRIOR_W_SD=0.2
PRIOR_B_SD=2.0
PRIOR_PSI_A=1.0
PRIOR_PSI_B=2.0

# ===========================================================================
# Beta prior on each rho_k. The misspecified setting uses the mean/concentration
# parameterization a_k = mu_k * tau_k, b_k = (1 - mu_k) * tau_k with tau_k = n_k,
# the number of pairs in class k (21 JASPAR2024, 13 PCHi-C), so each prior is worth
# as much as the data on its class.
# ===========================================================================
make_tau () {  # $1 = Q path: n_k for every class k >= 1, comma-separated
  python3 -c '
import sys, numpy as np, pandas as pd
Q = pd.read_csv(sys.argv[1], index_col=0).to_numpy()
od = ~np.eye(Q.shape[0], dtype=bool)
print(",".join(f"{float((Q[od] == k).sum()):g}" for k in range(1, int(Q.max()) + 1)))
' "$1"
}
TAU=$(make_tau "$Q_ORIG")
# Strongly informative but misspecified means: JASPAR2024 (true 0.286) at 0.80,
# PCHi-C (true 0.606) at 0.25.
MU_WRONG="0.8,0.25"
# Expert-elicited Beta shape parameters (supplement table "prior elicitation") of
# JASPAR2024 and PCHi-C; their means are the generating probabilities.
ELICITED_A="2.2042,5.6799"
ELICITED_B="5.5046,3.6995"

# ---- The 4 prior settings (index S = 0..3) --------------------------------
#   0 noninfo         : Beta(1,1) on every rho_k
#   1 expert_elicited : the elicited Beta(a_k, b_k)
#   2 wrong_extreme   : means (0.8, 0.25), tau_k = n_k
#   3 no_partition    : Q_NOPART, so no rho_k; rho_0 keeps its Beta(1,1) prior
NAMES=(  noninfo   expert_elicited  wrong_extreme  no_partition )
PARAMS=( ab        ab               mean_conc      ab           )
A_VALS=( "1.0"     "$ELICITED_A"    ""             "1.0"        )
B_VALS=( "1.0"     "$ELICITED_B"    ""             "1.0"        )
MEANS=(  ""        ""               "$MU_WRONG"    ""           )
CONCS=(  ""        ""               "$TAU"         ""           )
Q_VALS=( "$Q_ORIG" "$Q_ORIG"        "$Q_ORIG"      "$Q_NOPART"  )

NAME=${NAMES[$S]}
Q_PATH=${Q_VALS[$S]}
if [ "${PARAMS[$S]}" = "ab" ]; then
  PRIOR_K_ARGS=( --prior_rho_k_param ab \
                 --prior_rho_k_a "${A_VALS[$S]}" --prior_rho_k_b "${B_VALS[$S]}" )
else
  PRIOR_K_ARGS=( --prior_rho_k_param mean_conc \
                 --prior_rho_k_mean "${MEANS[$S]}" --prior_rho_k_conc "${CONCS[$S]}" )
fi

OUT_DIR=${OUT_DIR:-$SIM/mcmc}
mkdir -p "$OUT_DIR"
OUT_PATH=$OUT_DIR/${NAME}_DAG${IDX}.pkl

echo "Setting $S = $NAME  (DAG $IDX)"
echo "X_PATH=$X_PATH"
echo "Q_PATH=$Q_PATH"
echo "OUT_PATH=$OUT_PATH"

python mcmc.py \
  --x_path "$X_PATH" \
  --q_path "$Q_PATH" \
  --out_path "$OUT_PATH" \
  --n_iter "$N_ITER" \
  --n_chains "$N_CHAINS" \
  --thin "$THIN" \
  --seed "$SEED" \
  --factor "$FACTOR" \
  --sigma_b "$SIGMA_B" \
  --sigma_w "$SIGMA_W" \
  --sigma_logpsi "$SIGMA_LOGPSI" \
  --prior_w_sd "$PRIOR_W_SD" \
  --prior_b_sd "$PRIOR_B_SD" \
  --prior_psi_a "$PRIOR_PSI_A" \
  --prior_psi_b "$PRIOR_PSI_B" \
  --prior_rho_a "$PRIOR_RHO_A" \
  --prior_rho_b "$PRIOR_RHO_B" \
  "${PRIOR_K_ARGS[@]}" \
  --compress --compress_level 9

echo "Job finished at $(date)"
