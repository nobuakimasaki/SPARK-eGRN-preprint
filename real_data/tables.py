"""
tables.py
=========
The numbers the paper reports from the real-data posterior, recomputed from eval.py's
output so none of them rests on a hand calculation.

Writes to config.OUT_DIR/tables/:

  tf_signs.csv     : Table S4 -- per TF with >= 1 edge at PIP >= config.PIP_CUT into an
                     enhancer: edges, negative (posterior median coefficient < 0) and
                     supported (prior evidence code > 0), with percentages
  tf_signs.tex     : the same as the LaTeX body of Table S4 (two columns of TFs)
  results.txt      : every count, share and rho_k summary quoted in the Results

Usage
-----
  python tables.py
"""

import os

import numpy as np
import pandas as pd

import config

TABLE_DIR = os.path.join(config.OUT_DIR, "tables")
SOURCE_NAMES = {0: "baseline", 1: "ChIP-Atlas", 2: "JASPAR2024 CORE",
                3: "linear proximity", 4: "PCHi-C"}


def _is_enh(s):
    return s.str.startswith("chr")


def _pct(k, n):
    return 100.0 * k / n if n else float("nan")


def tf_signs(tfenh):
    """Table S4: one row per TF, sorted by number of edges (descending), then by name."""
    rows = []
    for tf, t in tfenh.groupby("parent"):
        n, neg, sup = len(t), int((t["w_post_median"] < 0).sum()), int((t["code"] > 0).sum())
        rows.append(dict(TF=tf, edges=n, negative=neg, negative_pct=_pct(neg, n),
                         supported=sup, supported_pct=_pct(sup, n)))
    df = pd.DataFrame(rows).sort_values(["edges", "TF"], ascending=[False, True])
    return df.reset_index(drop=True)


def tf_signs_latex(df, total):
    """Table S4's body: the TFs in two side-by-side columns, then the all-TFs total."""
    cell = lambda r: (f"{r.TF} & {r.edges} & {r.negative} ({r.negative_pct:.1f}) & "
                      f"{r.supported} ({r.supported_pct:.1f})")
    half = (len(df) + 1) // 2
    left, right = df.iloc[:half], df.iloc[half:]
    lines = []
    for i in range(half):
        l = cell(left.iloc[i])
        r = cell(right.iloc[i]) if i < len(right) else "& & &"
        lines.append(f"{l} & {r} \\\\")
    lines.append("\\midrule")
    lines.append(f"All TFs & {total['edges']} & {total['negative']} "
                 f"({total['negative_pct']:.1f}) & {total['supported']} "
                 f"({total['supported_pct']:.1f}) & & & & \\\\")
    return "\n".join(lines) + "\n"


def main():
    os.makedirs(TABLE_DIR, exist_ok=True)
    Q = pd.read_csv(config.PRIOR_PATH, index_col=0)
    nodes = pd.read_csv(config.NODES_PATH).set_index("node")
    edges = pd.read_csv(os.path.join(config.EVAL_DIR, "edge_recovery.csv"))
    rho = np.load(os.path.join(config.EVAL_DIR, "rho_samples.npy"))

    out = []
    say = out.append

    # --- Topological prior ---
    q = Q.to_numpy()
    off = ~np.eye(q.shape[0], dtype=bool)
    say("Topological prior")
    say(f"  ordered pairs: {int(off.sum()):,}")
    for c in (-1, 0, 1, 2, 3, 4):
        say(f"  label {c:>2} ({'forbidden' if c == -1 else SOURCE_NAMES[c]}): "
            f"{int(((q == c) & off).sum()):,}")
    permitted = int(((q >= 0) & off).sum())
    supported = int(((q > 0) & off).sum())
    say(f"  permitted pairs: {permitted:,}; supported by a prior source: {supported:,} "
        f"({_pct(supported, permitted):.0f}%)")
    chip_tfs = sorted(Q.index[(Q == 1).any(axis=1)])
    say(f"  TFs with ChIP-Atlas pairs: {', '.join(chip_tfs)}; "
        + ", ".join(f"{t} {int((Q.loc[t] == 1).sum())}" for t in chip_tfs))

    # --- Inferred edges at PIP >= cut ---
    edges["code"] = [int(Q.loc[p, c]) for p, c in zip(edges["parent"], edges["child"])]
    hi = edges[edges["post_prob"] >= config.PIP_CUT].copy()
    tfenh = hi[~_is_enh(hi["parent"]) & _is_enh(hi["child"])]
    enhg  = hi[_is_enh(hi["parent"]) & ~_is_enh(hi["child"])]
    say("")
    say(f"Inferred network (PIP >= {config.PIP_CUT})")
    say(f"  edges: {len(hi)}; supported by a prior source: {int((hi['code'] > 0).sum())} "
        f"({_pct(int((hi['code'] > 0).sum()), len(hi)):.0f}%)")
    say(f"  TF -> enhancer: {len(tfenh)} over {tfenh['parent'].nunique()} TFs; negative "
        f"{int((tfenh['w_post_median'] < 0).sum())} "
        f"({_pct(int((tfenh['w_post_median'] < 0).sum()), len(tfenh)):.1f}%)")
    say(f"  enhancer -> gene: {len(enhg)}; positive {int((enhg['w_post_median'] > 0).sum())}")
    sup, uns = tfenh[tfenh["code"] > 0], tfenh[tfenh["code"] == 0]
    say(f"  share negative, TF -> enhancer supported {_pct((sup['w_post_median'] < 0).sum(), len(sup)):.1f}% "
        f"vs unsupported {_pct((uns['w_post_median'] < 0).sum(), len(uns)):.1f}%")
    top = tfenh.groupby("parent").size().sort_values(ascending=False).head(3)
    say(f"  top 3 TFs by outgoing edges: "
        + ", ".join(f"{t} {n}" for t, n in top.items())
        + f" ({_pct(top.sum(), len(tfenh)):.1f}% of TF -> enhancer edges)")
    for tf in ("AFF3", "NFIA", "PPARG", "MEF2A", "EGR1"):
        t = tfenh[tfenh["parent"] == tf]
        say(f"  {tf}: {len(t)} edges, {int((t['code'] > 0).sum())} supported "
            f"({_pct(int((t['code'] > 0).sum()), len(t)):.1f}%), "
            f"{_pct(int((t['w_post_median'] < 0).sum()), len(t)):.1f}% negative")

    # --- Class edge probabilities ---
    say("")
    say(f"Class edge probabilities (posterior median [95% CrI], S = {rho.shape[0]:,} draws)")
    for k in range(rho.shape[1]):
        med, lo, up = np.percentile(rho[:, k], [50, 2.5, 97.5])
        say(f"  rho_{k} ({SOURCE_NAMES[k]}): {med:.2f} [{lo:.2f}, {up:.2f}]")

    # --- immGWAS enhancers ---
    gw = nodes.index[nodes["is_gwas_anchor"].astype(bool)].tolist()
    tg = tfenh[tfenh["child"].isin(gw)]
    by_tf = tg.groupby("parent")["child"].nunique().sort_values(ascending=False)
    ge = hi[hi["parent"].isin(gw) & ~_is_enh(hi["child"])]
    say("")
    say(f"immGWAS enhancers ({len(gw)})")
    say(f"  TFs regulating >= 1 immGWAS enhancer: {tg['parent'].nunique()} of "
        f"{int(nodes['is_tf'].astype(bool).sum())}; "
        + ", ".join(f"{t} {n}" for t, n in by_tf.head(3).items()))
    say(f"  immGWAS enhancer -> target edges: {len(ge)} "
        f"(targets: {', '.join(ge['child'])})")

    with open(os.path.join(TABLE_DIR, "results.txt"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))

    # --- Table S4 ---
    df = tf_signs(tfenh)
    df.to_csv(os.path.join(TABLE_DIR, "tf_signs.csv"), index=False, float_format="%.1f")
    n, neg, sup = len(tfenh), int((tfenh["w_post_median"] < 0).sum()), int((tfenh["code"] > 0).sum())
    total = dict(edges=n, negative=neg, negative_pct=_pct(neg, n),
                 supported=sup, supported_pct=_pct(sup, n))
    with open(os.path.join(TABLE_DIR, "tf_signs.tex"), "w") as f:
        f.write(tf_signs_latex(df, total))
    print(f"\nTable S4: {len(df)} TFs; written to {TABLE_DIR}")


if __name__ == "__main__":
    main()
