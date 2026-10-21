#!/usr/bin/env python3
"""Build the topological prior of the simulation study: the chr2 subset of the real-data
(M-CSF 16h) topological prior.

Nodes: every node on chr2 (enhancers are named chr2-...; TFs and genes use the `chr`
column of the node table), ordered by genomic position.

Evidence recoding. The real-data prior codes its sources as
    {ChIP-Atlas: 1, JASPAR2024_CORE: 2, linear proximity: 3, PCHi-C_high: 4}.
The chr2 subset contains no ChIP-Atlas pairs and is recoded onto two classes:
    2 (JASPAR2024 CORE)  -> 1
    4 (PCHi-C high)      -> 2
    3 (linear proximity) -> 0 (2 pairs, demoted to the baseline class)
with -1 (forbidden) and 0 (baseline) unchanged.

Writes data/simulation/prior/prior_matrix.csv and node_order.csv. Run this first, then
simulate_graph.py (see its docstring for reproducing data/simulation/ exactly).
"""
import os

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRIOR = os.path.join(REPO, "data", "real_data", "vm_m16h_b1_prior_matrix.csv")
NODES = os.path.join(REPO, "data", "real_data", "vm_m16h_b1_nodes.csv")
OUT_DIR = os.path.join(REPO, "data", "simulation", "prior")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    P = pd.read_csv(PRIOR, index_col=0)
    nodes = pd.read_csv(NODES).set_index("node")

    def is_enh(n):
        return n.startswith("chr")

    def chrom(n):
        return n.split("-")[0] if is_enh(n) else str(nodes.loc[n, "chr"])

    def pos(n):
        """Enhancer start (chr2-START-END) or gene TSS."""
        return int(n.split("-")[1]) if is_enh(n) else int(nodes.loc[n, "pos"])

    def kind(n):
        return "enhancer" if is_enh(n) else ("TF" if nodes.loc[n, "is_tf"] else "gene")

    ordered = sorted((n for n in P.index if chrom(n) == "chr2"), key=pos)
    pd.DataFrame({"node": ordered,
                  "kind": [kind(n) for n in ordered],
                  "pos":  [pos(n) for n in ordered]}).to_csv(
        os.path.join(OUT_DIR, "node_order.csv"), index=False)

    sub = P.loc[ordered, ordered].replace({2: 1, 4: 2, 3: 0})
    sub.to_csv(os.path.join(OUT_DIR, "prior_matrix.csv"))

    vals, cts = np.unique(sub.values, return_counts=True)
    print(f"{len(ordered)} chr2 nodes; code counts {dict(zip(vals.tolist(), cts.tolist()))}")


if __name__ == "__main__":
    main()
