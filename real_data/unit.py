"""
unit.py
=======
The inferred cis-regulatory unit of one target gene, drawn as a small layered network
(Figure 3D for IL1A, drawn by gwas_panel.py, and the supplementary figure for IL1RN).

Starting from the target gene, the unit is grown outwards along the inferred edges, two
steps:

  1. the enhancers e with P(e -> target) >= THRESHOLD, and
  2. the transcription factors t with P(t -> e) >= THRESHOLD for one of those enhancers.

Only the edges selected in those two steps are drawn. Each edge carries three things beyond
its direction: the posterior median regulatory effect w conditional on the edge being
present, written above the edge and reinforced by the edge colour (red up-regulation, blue
down, grey where the 95% CrI for w includes 0); the posterior edge probability, as the edge
width; and whether the topological prior had any evidence for that ordered pair, as a solid
rather than dashed line, with the supporting source named under the effect.

Inputs
------
config.EVAL_DIR/edge_recovery.csv (from eval.py), the topological prior and the node table.

Outputs
-------
--out <path>.png (+ .pdf alongside), and config.OUT_DIR/unit/unit_<GENE>.csv: every drawn
edge with its posterior summaries and prior code.

Usage
-----
  python unit.py --gene IL1RN --out ../paper/figures/supp_unit_IL1RN.png
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

import config
import network

# Posterior edge probability an ordered pair must reach to be drawn, and so to pull its
# parent into the unit.
THRESHOLD = config.PIP_CUT

# ---------------------------------------------------------------------------
# Geometry. Three rows: TFs on top, enhancers in the middle, the target gene below. The
# TF -> enhancer band is the busy one and is given the taller gap, so its effect labels
# have room to stagger.
# ---------------------------------------------------------------------------
Y_TF, Y_ENH, Y_GENE = 2.30, 1.00, 0.00

# Marker areas in points^2, and the outline width of an ordinary node.
NODE_SIZE     = 620
NODE_EDGE_LW  = 1.0
# immGWAS enhancers get a heavier outline, so a unit's anchor is identifiable even in
# greyscale, where its disease colour is not.
ANCHOR_EDGE_LW = 2.6

# Type sizes (points).
NODE_LABEL_FS  = 13   # TF and target-gene names
ENH_LABEL_FS   = 11   # 'chr10:62.64 Mb'
# Horizontal offset (points) of an enhancer's name from its node, to the side away from the
# target gene, so the name clears that enhancer's own edge down to the gene.
ENH_LABEL_DX   = 16
TRAIT_LABEL_FS = 11   # the disease abbreviation under an anchor enhancer
EDGE_LABEL_FS  = 10   # the median effect w
SRC_LABEL_FS   = 8.5  # the prior evidence source, under the effect
LEGEND_FS      = 10
# Column counts of the node and edge legends. gwas_panel.py takes its layout from this
# value, so the supplementary unit and Figure 3D read the same way.
LEGEND_NCOLS   = (1, 2)

# Edge width encodes P(edge), on a ramp stretched over [cut, 1]: a near-certain edge is
# drawn at exactly twice the width of one at the cut.
EDGE_W_MIN = network.EDGE_FLAT_WIDTH
EDGE_W_MAX = 2.0 * EDGE_W_MIN

# Colour of an edge whose 95% CrI for w includes 0.
EDGE_UNSIGNED_COLOR = "#8C8C8C"


def _edge_width(p, lo):
    """Line width of an edge of posterior probability `p`, on a ramp running from EDGE_W_MIN
    at the cut `lo` to EDGE_W_MAX at 1."""
    f = 0.0 if p <= lo else min(1.0, (float(p) - lo) / max(1e-9, 1.0 - lo))
    return EDGE_W_MIN + (EDGE_W_MAX - EDGE_W_MIN) * f


# Distance in points by which an edge stops short of the node markers at either end, so the
# arrowhead sits against the target node rather than under it.
EDGE_SHRINK = 15.0
ARROW_SCALE = 16.0

# Where along an edge its effect label sits, as a fraction of the way from parent to child.
# Labels are staggered through this cycle in order of their edge's midpoint x, so two edges
# running side by side do not put their labels in the same place.
LABEL_FRACTIONS     = (0.32, 0.46, 0.60, 0.74)
GENE_LABEL_FRACTION = 0.62

# Backing box for the labels that sit on top of edges.
LABEL_BBOX = dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none",
                  alpha=0.82)

# Evidence source names as they appear under an edge. Code 0 is the baseline class and is
# left unnamed: the dashed line already says there is no supporting source.
EVIDENCE_SHORT = {1: "ChIP-Atlas", 2: "JASPAR2024", 3: "Linear proximity", 4: "PCHi-C"}


def load_nodes():
    """The node table (kind, is_tf, is_gwas_anchor, is_gene_near_anchor, chr, pos, ...),
    indexed by node name."""
    return pd.read_csv(config.NODES_PATH).set_index("node")


def _fmt_w(w):
    """Posterior median effect as it is written above an edge: two significant figures,
    with the sign always shown."""
    return f"{w:+#.2g}"


# ---------------------------------------------------------------------------
# Growing the unit
# ---------------------------------------------------------------------------

def build_unit(edges, Q, target, threshold=THRESHOLD):
    """Grow the cis-regulatory unit of `target` outwards along the inferred edges.

    `edges` is edge_recovery.csv and `Q` the integer prior matrix. Returns
    (enh_edges, tf_edges): the enhancer -> target edges at P >= `threshold`, and the
    TF -> enhancer edges into those enhancers at P >= `threshold`. Both frames carry
    edge_recovery.csv's columns plus `prior_code` and `prior_source`.

    The parent of a gene can only ever be an enhancer and the parent of an enhancer only a
    TF -- the prior forbids every other pair -- so no type filter is applied beyond
    following the edges."""
    hi = edges[edges["post_prob"] >= threshold]

    enh_edges = hi[hi["child"] == target].copy()
    enhancers = enh_edges["parent"].tolist()
    tf_edges  = hi[hi["child"].isin(enhancers)].copy()

    for df in (enh_edges, tf_edges):
        df["prior_code"] = [int(Q.loc[p, c]) for p, c in zip(df["parent"], df["child"])]
        df["prior_source"] = [network.CODE_LOOKUP.get(k, "") for k in df["prior_code"]]
    return enh_edges, tf_edges


def _enhancer_order(enhancers):
    """Enhancers left to right along the genome (chromosome, start)."""
    def key(e):
        chrom, start, _ = e.split("-", 2)
        return (network._chr_key(chrom), int(start))
    return sorted(enhancers, key=key)


def _tf_order(tf_edges, enh_x):
    """TFs left to right by the mean x of the enhancers they regulate (the barycentre
    heuristic for layered drawings), so the edges of the TF -> enhancer band cross as
    little as the layering allows. Ties break on the TF name."""
    bary = tf_edges.groupby("parent")["child"].apply(
        lambda cs: float(np.mean([enh_x[c] for c in cs])))
    return sorted(bary.index, key=lambda t: (bary[t], t))


def _row_x(n):
    """x coordinates of `n` nodes spread across the row, in [0, 1]. A single node sits in
    the middle rather than at the left edge."""
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.array([0.5])
    return np.linspace(0.0, 1.0, n)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _sign_determined(row):
    """Whether the edge's 95% credible interval for w excludes 0. Bounds are NaN for an
    edge present in too few draws to summarize; it then counts as undetermined."""
    return bool((row["w_post_lower95"] > 0) or (row["w_post_upper95"] < 0))


def _edge_colour(row):
    """Red for a credibly positive effect, blue for a credibly negative one, grey when the
    95% CrI includes 0."""
    if not _sign_determined(row):
        return EDGE_UNSIGNED_COLOR
    return network.EDGE_POS_COLOR if row["w_post_median"] > 0 else network.EDGE_NEG_COLOR


def _draw_edges(ax, rows, pos, fractions, lo, s=1.0):
    """Draw `rows` (an edge frame) as arrows between the node positions in `pos`, and write
    each one's effect above it. `fractions` is the cycle of label positions along the edge;
    edges are taken in order of their midpoint x. `lo` is the probability cut the edges
    passed, which anchors the width ramp; `s` scales every size to the panel."""
    rows = rows.assign(_mx=[(pos[p][0] + pos[c][0]) / 2
                            for p, c in zip(rows["parent"], rows["child"])])
    for k, (_, r) in enumerate(rows.sort_values("_mx").iterrows()):
        a, b = np.array(pos[r["parent"]]), np.array(pos[r["child"]])
        colour = _edge_colour(r)
        supported = r["prior_code"] > 0
        lw = _edge_width(r["post_prob"], lo)
        # The arrowhead grows with the line.
        ax.add_patch(FancyArrowPatch(
            a, b, arrowstyle="-|>", mutation_scale=(ARROW_SCALE + 1.1 * lw) * s,
            shrinkA=EDGE_SHRINK * s, shrinkB=EDGE_SHRINK * s,
            linewidth=lw * s, color=colour,
            linestyle="solid" if supported else (0, (4.0, 2.4)), zorder=1))

        f = fractions[k % len(fractions)]
        x, y = a + f * (b - a)
        ax.text(x, y, _fmt_w(r["w_post_median"]), ha="center", va="center",
                fontsize=EDGE_LABEL_FS * s, color=colour, fontweight="bold",
                bbox=LABEL_BBOX, zorder=4)
        # The supporting source goes under the effect. Nothing is written for the baseline
        # class: the dashed line says it.
        if supported:
            ax.annotate(EVIDENCE_SHORT.get(r["prior_code"], r["prior_source"]),
                        (x, y), textcoords="offset points", xytext=(0, -11 * s),
                        ha="center", va="center", fontsize=SRC_LABEL_FS * s, color="0.25",
                        bbox=LABEL_BBOX, zorder=4)


def _draw_nodes(ax, xs, y, *, colours, marker, sizes, lws):
    for x, colour, size, lw in zip(xs, colours, sizes, lws):
        ax.scatter([x], [y], s=size, marker=marker, facecolor=colour,
                   edgecolor="black", linewidth=lw, zorder=3)


def _legends(fig, traits_present, has_plain_enh, has_supported, has_baseline,
             has_unsigned, lo, fs=LEGEND_FS, ncols=(1, 1)):
    """The two legends, side by side under the drawing: what a node is, and what an edge
    says. Every entry is conditional on the unit containing what it describes. `fs` is the
    entry size, for a host figure set at a different type size (gwas_panel.py), and `ncols`
    the column counts of the node and edge legends."""
    node_handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=11,
               markerfacecolor=network.NODE_TF_COLOR, markeredgecolor="black",
               label="Transcription factor"),
    ]
    if has_plain_enh:
        node_handles.append(
            Line2D([], [], marker=network.NODE_ENH_SHAPE, linestyle="none", markersize=11,
                   markerfacecolor=network.NODE_ENH_COLOR, markeredgecolor="black",
                   label="Enhancer"))
    for trait, colour in traits_present:
        node_handles.append(
            Line2D([], [], marker=network.NODE_ENH_SHAPE, linestyle="none", markersize=11,
                   markerfacecolor=colour, markeredgecolor="black",
                   markeredgewidth=ANCHOR_EDGE_LW / 2,
                   label=f"immGWAS enhancer ({network._trait_display(trait)})"))
    node_handles.append(
        Line2D([], [], marker="o", linestyle="none", markersize=11,
               markerfacecolor=network.NODE_GENE_COLOR, markeredgecolor="black",
               label="Target gene"))

    edge_handles = [
        Line2D([], [], color=network.EDGE_POS_COLOR, lw=2.6, label="Up-regulation ($w > 0$)"),
        Line2D([], [], color=network.EDGE_NEG_COLOR, lw=2.6, label="Down-regulation ($w < 0$)"),
    ]
    if has_unsigned:
        edge_handles.append(
            Line2D([], [], color=EDGE_UNSIGNED_COLOR, lw=2.6,
                   label="95% CrI for $w$ includes 0"))
    if has_supported:
        edge_handles.append(Line2D([], [], color="0.15", lw=2.6,
                                   label="Supported by a prior evidence source"))
    if has_baseline:
        edge_handles.append(Line2D([], [], color="0.15", lw=2.6, linestyle=(0, (4.0, 2.4)),
                                   label="Baseline class (no supporting source)"))
    # Three rungs over the drawn range, so the width encoding is readable off the legend.
    for p in (lo, (lo + 1.0) / 2, 1.0):
        edge_handles.append(Line2D([], [], color="0.15", lw=_edge_width(p, lo),
                                   label=f"$P(\\mathrm{{edge}}) = {p:.2f}$"))

    leg_n = fig.legend(handles=node_handles, loc="outside lower left", frameon=False,
                       fontsize=fs, title="Node", alignment="left", ncols=ncols[0])
    leg_e = fig.legend(handles=edge_handles, loc="outside lower right", frameon=False,
                       fontsize=fs, title="Edge", alignment="left", ncols=ncols[1])
    for leg in (leg_n, leg_e):
        leg.get_title().set_fontsize(fs + 1)
        leg.get_title().set_fontweight("bold")
    return leg_n, leg_e


def _render_unit(ax, target, enh_edges, tf_edges, lo, s=1.0):
    """Draw one unit into an existing axes and return what its legend needs: the anchor
    diseases in it, whether it has an enhancer carrying no GWAS variant, and the prior codes
    and probabilities of its edges. `lo` is the probability cut the unit was grown at, and
    `s` scales every mark and type size, so the same drawing serves the standalone figure
    (s = 1) and Figure 3D."""
    enhancers = _enhancer_order(enh_edges["parent"].tolist())
    enh_x     = dict(zip(enhancers, _row_x(len(enhancers))))
    tfs       = _tf_order(tf_edges, enh_x) if len(tf_edges) else []
    tf_x      = dict(zip(tfs, _row_x(len(tfs))))

    pos = {t: (tf_x[t], Y_TF) for t in tfs}
    pos.update({e: (enh_x[e], Y_ENH) for e in enhancers})
    pos[target] = (0.5, Y_GENE)

    # Enhancer colours: the disease of its GWAS variant for an anchor, the flat enhancer
    # blue for an enhancer that carries no variant.
    trait_of  = network._enhancer_labels(enhancers)
    is_anchor = {e: e in network.PEAK_TRAITS for e in enhancers}
    abbrev    = network.trait_abbrev_labels([e for e in enhancers if is_anchor[e]])
    enh_colours = [network.TRAIT_COLORS.get(trait_of[e], network.NODE_ENH_COLOR)
                   if is_anchor[e] else network.NODE_ENH_COLOR for e in enhancers]

    _draw_edges(ax, tf_edges, pos, LABEL_FRACTIONS, lo, s)
    _draw_edges(ax, enh_edges, pos, (GENE_LABEL_FRACTION,), lo, s)

    node_s = NODE_SIZE * s ** 2   # a marker area, so it goes as the square of the scale
    _draw_nodes(ax, [tf_x[t] for t in tfs], Y_TF,
                colours=[network.NODE_TF_COLOR] * len(tfs), marker="o",
                sizes=[node_s] * len(tfs), lws=[NODE_EDGE_LW * s] * len(tfs))
    _draw_nodes(ax, [enh_x[e] for e in enhancers], Y_ENH,
                colours=enh_colours, marker=network.NODE_ENH_SHAPE,
                sizes=[node_s * network.NODE_ENH_SIZE_MULT] * len(enhancers),
                lws=[(ANCHOR_EDGE_LW if is_anchor[e] else NODE_EDGE_LW) * s
                     for e in enhancers])
    _draw_nodes(ax, [0.5], Y_GENE, colours=[network.NODE_GENE_COLOR], marker="o",
                sizes=[node_s], lws=[NODE_EDGE_LW * s])

    # TFs (acting as regulators) are roman and the target gene italic. TF names above their
    # row and the target's below its own, both clear of the edge bands.
    for t in tfs:
        ax.annotate(t, (tf_x[t], Y_TF), textcoords="offset points", xytext=(0, 20 * s),
                    ha="center", va="bottom", fontsize=NODE_LABEL_FS * s)
    ax.annotate(target, (0.5, Y_GENE), textcoords="offset points", xytext=(0, -20 * s),
                ha="center", va="top", fontsize=NODE_LABEL_FS * s, fontstyle="italic",
                fontweight="bold")

    # Enhancer names go below their node, pushed to the side away from the target gene,
    # since that is where the node's own edge runs. An anchor's disease abbreviation sits
    # under its coordinates in the colour of the node.
    short = network._enhancer_short_labels(enhancers)
    for e in enhancers:
        right = enh_x[e] <= 0.5                       # label on the far side from the gene
        dx, ha = (ENH_LABEL_DX * s, "left") if right else (-ENH_LABEL_DX * s, "right")
        ax.annotate(short[e], (enh_x[e], Y_ENH), textcoords="offset points",
                    xytext=(dx, -12 * s), ha=ha, va="top", fontsize=ENH_LABEL_FS * s,
                    bbox=LABEL_BBOX, zorder=4)
        if is_anchor[e]:
            ax.annotate(abbrev[e], (enh_x[e], Y_ENH), textcoords="offset points",
                        xytext=(dx, -27 * s), ha=ha, va="top",
                        fontsize=TRAIT_LABEL_FS * s, fontweight="bold",
                        color=network.TRAIT_COLORS.get(trait_of[e], network.NODE_ENH_COLOR),
                        bbox=LABEL_BBOX, zorder=4)

    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(Y_GENE - 0.45, Y_TF + 0.35)
    ax.set_axis_off()

    traits = []
    for e in enhancers:
        if is_anchor[e] and trait_of[e] not in [t for t, _ in traits]:
            traits.append((trait_of[e],
                           network.TRAIT_COLORS.get(trait_of[e], network.NODE_ENH_COLOR)))
    # enh_edges is never empty (a unit with no enhancer is not drawn); tf_edges can be.
    drawn = pd.concat([df for df in (enh_edges, tf_edges) if not df.empty])
    unsigned = ~drawn.apply(_sign_determined, axis=1)
    return dict(traits=traits, plain_enh=not all(is_anchor.values()),
                codes=drawn["prior_code"], unsigned=bool(unsigned.any()), probs=drawn["post_prob"],
                n_tf=len(tfs), n_enh=len(enhancers))


def plot_unit(target, enh_edges, tf_edges, out_path, lo=THRESHOLD):
    """Draw one unit as its own figure and save it as PNG + PDF."""
    n_tf  = tf_edges["parent"].nunique()
    n_enh = enh_edges["parent"].nunique()
    # Width grows with the wider of the two upper rows so the names along it never collide,
    # and height follows width; both are clipped to what fits on a page.
    width  = float(np.clip(1.35 * max(n_tf, n_enh, 3) + 3.5, 8.5, 20.0))
    height = float(np.clip(0.55 * width, 6.5, 9.5))
    fig, ax = plt.subplots(figsize=(width, height), layout="constrained")

    info = _render_unit(ax, target, enh_edges, tf_edges, lo)
    legs = _legends(fig, info["traits"], info["plain_enh"],
                    (info["codes"] > 0).any(), (info["codes"] == 0).any(),
                    info["unsigned"], lo, ncols=LEGEND_NCOLS)
    save_kw = dict(bbox_inches="tight", pad_inches=0.1, bbox_extra_artists=list(legs))
    fig.savefig(out_path, dpi=200, **save_kw)
    fig.savefig(os.path.splitext(out_path)[0] + ".pdf", **save_kw)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Draw the inferred cis-regulatory unit of a target gene.")
    parser.add_argument("--gene", required=True, help="Target gene to expand from.")
    parser.add_argument("--out", required=True, help="Output figure path (.png).")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help=f"PIP an ordered pair must reach to be drawn (default {THRESHOLD}).")
    args = parser.parse_args()

    edges = pd.read_csv(os.path.join(config.EVAL_DIR, "edge_recovery.csv"))
    enh_edges, tf_edges = build_unit(edges, network.load_prior_matrix(), args.gene,
                                     args.threshold)
    if enh_edges.empty:
        raise SystemExit(f"{args.gene}: no enhancer reaches P >= {args.threshold}")

    table = pd.concat([df.assign(layer=name) for df, name in
                       ((enh_edges, "enhancer->gene"), (tf_edges, "TF->enhancer"))
                       if not df.empty])
    table = table[["layer", "parent", "child", "post_prob", "w_post_median",
                   "w_post_lower95", "w_post_upper95", "prior_code", "prior_source"]]
    csv_dir = os.path.join(config.OUT_DIR, "unit")
    os.makedirs(csv_dir, exist_ok=True)
    table.to_csv(os.path.join(csv_dir, f"unit_{args.gene}.csv"), index=False)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plot_unit(args.gene, enh_edges, tf_edges, args.out, args.threshold)
    print(f"{args.gene}: {enh_edges['parent'].nunique()} enhancer(s), "
          f"{tf_edges['parent'].nunique()} TF(s), {len(table)} edges "
          f"({int((table['prior_code'] > 0).sum())} with prior evidence); saved {args.out}")


if __name__ == "__main__":
    main()
