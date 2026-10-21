#!/bin/bash
#SBATCH --job-name=mcmc_real
#SBATCH --output=logs/mcmc_%j.out
#SBATCH --error=logs/mcmc_%j.err
#SBATCH --partition=hmem
#SBATCH --time=7-0:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=750GB

set -euo pipefail

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"

# Activate conda environment
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate mcmc

# Pin BLAS/threading libraries to a single thread per process. Chains are
# parallelized across processes (max_workers=n_chains=20 on cpus-per-task=20),
# so each worker must stay single-threaded -- otherwise a worker whose matmul
# trips the BLAS multi-threading cutoff spawns ~10 BLAS threads, and 20 workers
# x ~10 threads thrash the 20 allocated cores, slowing the whole iteration by
# ~10-50x.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "Using python: $(which python)"
python --version

# ===========================================================================
# Real-data MCMC: the M-CSF 16h depth-normalized count matrix under the
# expert-elicited prior on rho_k, warm-started from a pre-built DAG (.npz) --
# see the warm-start block below.
#
# These are the settings of the run reported in the paper (supplement table
# "Sampler settings for the real data analysis"): 20 chains, ladder with
# beta_19 = 0.6, swap probability 0.2, a flat Beta(1,1) prior on rho_0, and
# 5,250,000 iterations, of which the paper discards the first 3,000,000 as
# burn-in and keeps every 10th cold-chain draw.
#
# RUNTIME EXPECTATION -- READ BEFORE SUBMITTING
# ---------------------------------------------
# The real-data problem (377 nodes, 83 cells, 20 chains) runs at roughly 5-9
# iterations/s, so 5,250,000 iterations can approach or exceed the 7-day wall
# limit. If the job is killed, nothing is lost up to the last checkpoint: the
# checkpoint file is a strict truncation of the final file -- mcmc.py writes the
# same keys and slices every trace to [:t+1] -- so the analysis scripts read a
# checkpoint exactly as they read a completed run. There is no resume path
# (--init_path takes a warm-start .npz, not a checkpoint).
# ===========================================================================

# ---- Inputs: depth-normalized count matrix + topological prior ------------
# Both are provided in ../data/real_data/ (the M-CSF 16h dataset: 83 metacells
# x 377 features).
MAT_NAME=vm_m16h_b1
DATA_DIR=../data/real_data
X_PATH=$DATA_DIR/${MAT_NAME}.csv

# Directed topological prior matrix, rows and columns in the same node order as
# X. Integer codes: 1 ChIP-Atlas, 2 JASPAR2024_CORE, 3 linear proximity,
# 4 PCHi-C_high; 0 = baseline class; -1 = forbidden.
Q_PATH=$DATA_DIR/${MAT_NAME}_prior_matrix.csv

# ---- MCMC run parameters --------------------------------------------------
N_ITER=5250000
N_CHAINS=20
THIN=1
SEED=31

# Checkpoint cadence. Every write serializes the WHOLE accumulated sample store,
# so at a fixed interval the total checkpoint cost grows quadratically in the
# iteration count. The 2M runs used the old hardcoded 10,000 (200 writes); the
# same interval at 10M would be 1,000 writes of a store growing to ~223 GB --
# days of wall clock spent on I/O alone. 250,000 keeps the total comparable to
# the 2M runs while still bounding loss at the wall limit to <=250k iterations
# (~12 h at the observed rate).
CHECKPOINT_INTERVAL=250000

# Temperature ladder: 20 chains at beta_r = FACTOR^-r with beta_0=1 (cold chain).
# The hottest chain (r=19) is pinned at beta_19 = 0.6, NOT the 0.35 of the
# completed 2M runs. This is the one setting deliberately kept from the abandoned
# 5M runs, on the diagnostic evidence in results_real_data/output/diag/:
#
#   beta_19 = 0.35  ->  swap acceptance 0.018-0.090 across the 19 pairs (~0.05)
#   beta_19 = 0.60  ->  swap acceptance 0.21 -0.42  across the 19 pairs (~0.33)
#
# The 0.35 ladder spaces the rungs too widely (per-rung inverse-temperature gap
# 1 - 0.35^(1/19) = 0.054, against 0.027 at 0.6), so states almost never traverse
# it and the tempering does little for the cold chain. This matters because the
# swap step attempts only ONE adjacent pair per iteration (mcmc.py), making the
# expected iterations per ladder round trip ~ R^3 / (acc * p_ex): ~800,000 at
# acc=0.05 versus ~121,000 at acc=0.33, i.e. ~12 round trips over 10M iterations
# versus ~80.
#
# Do NOT instead keep beta_19 = 0.35 and add chains to close the spacing. With
# one pair swapped per iteration the round-trip cost grows as R^3, so going to 40
# chains would make mixing worse despite the 64 cores hmem offers.
#
# Cold-chain LOCAL acceptance is unchanged between the two ladders (graph 0.178
# vs 0.179, theta 0.136 vs 0.134), so the step sizes below still apply as tuned,
# and both ladders plateau at the same cold-chain log-posterior (~ -98,150).
FACTOR=$(python3 -c "print((1/0.6)**(1/19))")
FACTOR_STR=$(printf "%.4f" "$FACTOR")
# Theta step sizes tuned for the per-node (Metropolis-within-Gibbs) update.
# The sampler updates (b_j, W[:,j], log_psi_j) one node at a time, so each
# accept/reject is low-dimensional and tolerates far larger steps than the old
# joint update (which collapsed to ~0 acceptance on the dense warm-start graph).
# rho_0..rho_K are not random-walk parameters: the edge-probability update is a
# Gibbs draw from the exact Beta full conditional (Supplement 3.4).
SIGMA_B=0.1
SIGMA_W=0.03
SIGMA_LOGPSI=0.1

# ---- Expert-elicited rho_k priors (4 evidence classes) --------------------
# Beta shape parameters (a_k, b_k) fitted to the domain expert's elicited 10th
# and 90th percentiles for each source (supplement table "prior elicitation"):
# 1 ChIP-Atlas, 2 JASPAR2024_CORE, 3 linear proximity, 4 PCHi-C_high_confidence.
NAME=expert_elicited
A_VAL="2.0645,2.2042,5.5046,5.6799"
B_VAL="2.0645,5.5046,2.2042,3.6995"

# Flat Beta(1,1) prior on the baseline edge probability rho_0.
PRIOR_RHO_A=1.0
PRIOR_RHO_B=1.0

INIT_TAG=warm

# Filename tag encoding the setting + a/b vectors (commas -> dashes).
RHO_TAG="${NAME}_a${A_VAL//,/-}_b${B_VAL//,/-}"

mkdir -p output/real_data/mcmc
OUT_PATH=output/real_data/mcmc/${MAT_NAME}_init_${INIT_TAG}_${RHO_TAG}_pra_${PRIOR_RHO_A}_prb_${PRIOR_RHO_B}_it_${N_ITER}_f_${FACTOR_STR}_sb_${SIGMA_B}_sw_${SIGMA_W}_slp_${SIGMA_LOGPSI}_nc_${N_CHAINS}_th_${THIN}_sd_${SEED}.pkl

echo "Matrix = $MAT_NAME  Setting = $NAME  Init = $INIT_TAG"
echo "X_PATH=$X_PATH"
echo "Q_PATH=$Q_PATH"
echo "OUT_PATH=$OUT_PATH"
echo "N_ITER=$N_ITER  CHECKPOINT_INTERVAL=$CHECKPOINT_INTERVAL  FACTOR=$FACTOR_STR"
echo "rho_0 prior = Beta($PRIOR_RHO_A, $PRIOR_RHO_B)"

# ---- Warm-start DAG (pre-built) -------------------------------------------
# Every chain starts from the warm start used in the paper (Supplementary
# Note 4), provided as a pre-built .npz. mcmc.py loads it via --init_path,
# validates it (acyclic, no forbidden edges), and seeds b/W/log_psi from it.
INIT_DIR=../data/real_data/warm_start
INIT_PATH=$INIT_DIR/${MAT_NAME}.npz
if [ ! -f "$INIT_PATH" ]; then
  echo "ERROR: warm-start file not found: $INIT_PATH" >&2
  exit 1
fi
echo "Using pre-built warm start -> $INIT_PATH"
INIT_ARG=(--init_path "$INIT_PATH")

python mcmc.py \
  --x_path "$X_PATH" \
  --q_path "$Q_PATH" \
  --out_path "$OUT_PATH" \
  ${INIT_ARG[@]+"${INIT_ARG[@]}"} \
  --n_iter "$N_ITER" \
  --n_chains "$N_CHAINS" \
  --thin "$THIN" \
  --seed "$SEED" \
  --factor "$FACTOR" \
  --sigma_b "$SIGMA_B" \
  --sigma_w "$SIGMA_W" \
  --sigma_logpsi "$SIGMA_LOGPSI" \
  --prior_rho_a "$PRIOR_RHO_A" \
  --prior_rho_b "$PRIOR_RHO_B" \
  --prior_rho_k_param ab \
  --prior_rho_k_a "$A_VAL" \
  --prior_rho_k_b "$B_VAL" \
  --checkpoint_interval "$CHECKPOINT_INTERVAL" \
  --compress --compress_level 9

echo "Job finished at $(date)"
