#!/usr/bin/env bash
# run_analysis.sh -- regenerate every real-data result in the paper.
#
#   ./run_analysis.sh               figures and tables from the posterior summaries in
#                                   output/eval/ (minutes)
#   ./run_analysis.sh --from-chain  first recompute output/eval/ and the MCMC diagnostic
#                                   figures from the sampler output
#                                   (data/real_data/mcmc/expert_elicited_chain.pkl.zst;
#                                   about 15 minutes, up to ~16 GB of RAM)
#
# Figures are written to ../paper/figures/ under the names the paper includes; tables and
# the matrices behind Figure 3 to output/.
set -euo pipefail
cd "$(dirname "$0")"
export MPLBACKEND=Agg

if [[ "${1:-}" == "--from-chain" ]]; then
  # Posterior summaries over the paper's window: draws 3,000,000 to 5,250,000, every 10th.
  python eval.py
  # Supplementary figures: MCMC traces, acceptance and swap rates.
  python diag.py
fi

# Table S4 and every count, share and rho_k summary quoted in the Results.
python tables.py
# Figure 3.
python gwas_panel.py
# Supplementary figure: the cis-regulatory unit of IL1RN.
python unit.py --gene IL1RN --out ../paper/figures/supp_unit_IL1RN.png
