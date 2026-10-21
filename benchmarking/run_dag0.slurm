#!/bin/bash
#SBATCH --job-name=bench_dag0
#SBATCH --output=logs/dag0_%A_%a.out
#SBATCH --error=logs/dag0_%A_%a.err
#SBATCH --partition=hmem
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64GB
#SBATCH --array=0-9

# DAG0 on the simulation study: one array task per ground-truth DAG. Each run takes
# roughly 1.5 days (~0.56 s/iteration). mcmc_1sampleDAG0 allocates every sample
# store at n_sample up front, so a run peaks at ~33 GB.
#
# NOTE: create the log dir before submitting (SLURM opens the --output/--error
# files before this script runs):   mkdir -p logs
set -euo pipefail

echo "Job started on $(hostname) at $(date)"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-NA}"

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate r_env

# DAG0 parallelizes its 10 tempered chains across processes, so each worker must
# stay single-threaded; otherwise 10 workers x N BLAS threads thrash the 10 cores.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

IDX=${SLURM_ARRAY_TASK_ID}
X_PATH=../data/simulation/datasets/X_${IDX}.csv
OUT_DIR=output/dag0
mkdir -p "$OUT_DIR"

Rscript dag0.R "$X_PATH" "$OUT_DIR/dag0_DAG${IDX}"

echo "Job finished at $(date)"
