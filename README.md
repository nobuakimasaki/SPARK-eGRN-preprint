# SPARK

Code and data for the preprint *SPARK: Bayesian Structure Learning of Enhancer-based Gene
Regulatory Networks from Single-cell Multiomics Data and Prior Knowledge*.

## Layout

```
spark/                  the SPARK sampler (mcmc.py) and the SLURM scripts of the paper's runs
simulation/             simulation study: data generation, evaluation, figures, tables
benchmarking/           the two comparator methods (DAG0 and the CellOracle-style regression)
real_data/              real data analysis: evaluation, diagnostics, figures, tables
data/
  simulation/
    prior/              the chr2 topological prior (51 nodes)
    datasets/           the 10 ground-truth DAGs and their simulated counts
    mcmc/               SPARK's output, 4 prior settings x 10 DAGs (4 GB, not yet in the repository)
  real_data/
    vm_m16h_b1.csv                      M-CSF 16h dataset: 83 metacells x 377 features
    vm_m16h_b1_prior_matrix.csv         the topological prior for the above dataset
    vm_m16h_b1_nodes.csv                annotation of the 377 features
    warm_start/                         the starting DAG of every chain
    gwas/                               immune disease GWAS annotation of the immGWAS enhancers
    supplementary_data_1_reference_peaks_hg38.bed   reference ATAC peak set (Supplementary Data 1)
    mcmc/                               SPARK's output (6.5 GB, not yet in the repository)
paper/figures/          TikZ sources of Figure 1A-F and the figures written by the scripts
```

## Environments

- `environment.yml`: the SPARK sampler (`spark/`), with the versions the paper's runs used.
- `environment-analysis.yml`: everything else. These versions reproduce the paper's
  figures and tables.
- DAG0 needs R with the [BayesDAG0](https://github.com/junsoukchoi/BayesDAG0) package.
- Figure 1G (`simulation/fig1_panelG.py`) needs a LaTeX installation.

```
conda env create -f environment.yml
conda env create -f environment-analysis.yml
```

## Reproducing the paper

Run the Python scripts from the directory they are in.

**1. Simulated data** (`simulation/`, regenerates `data/simulation/prior/` and `datasets/`):

```
python build_chr2_subset.py
python simulate_graph.py
```

**2. SPARK runs** (`spark/`, SLURM array jobs; the `module`/`conda` lines are specific to our
cluster):

```
sbatch run_sim.slurm    # simulation: 4 prior settings x 10 DAGs -> data/simulation/mcmc/
sbatch run.slurm        # real data, expert-elicited prior   -> data/real_data/mcmc/
```

**3. Comparators** (`benchmarking/`):

```
sbatch run_dag0.slurm   # DAG0, one task per DAG          -> output/dag0/
python regression.py    # regression, all ten DAGs        -> output/regression/
```

**4. Simulation results** (`simulation/`):

```
python eval.py          # posterior summaries             -> output/eval/
python tables.py        # Tables S2, S3, Results numbers  -> output/tables/
python figures.py       # Figure 2, Figure S1             -> paper/figures/
python fig1_panelG.py   # Figure 1G                       -> paper/figures/
```

**5. Real data results** (`real_data/`):

```
./run_analysis.sh               # Figure 3, Figure S6, Table S4, Results numbers
./run_analysis.sh --from-chain  # also recompute output/eval/ and Figures S2-S5 from the chain
```

## Where each result comes from

| Result | Script |
|---|---|
| Figure 1A-F | `paper/figures/fig1_panel*.tex` (TikZ) |
| Figure 1G | `simulation/fig1_panelG.py` |
| Figure 2 | `simulation/figures.py` |
| Figure 3 | `real_data/gwas_panel.py` |
| Figure S1 (parameter recovery) | `simulation/figures.py` |
| Figures S2-S5 (MCMC diagnostics, real data) | `real_data/diag.py` |
| Figure S6 (IL1RN unit) | `real_data/unit.py` |
| Tables S2, S3 | `simulation/tables.py` |
| Table S4 (TF-enhancer signs) | `real_data/tables.py` |
| Simulation numbers in the Results | `simulation/tables.py` (`output/tables/results.txt`) |
| Real data numbers in the Results | `real_data/tables.py` (`output/tables/results.txt`) |
