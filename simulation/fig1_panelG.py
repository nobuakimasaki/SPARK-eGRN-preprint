"""
fig1_panelG.py
==============
Panel G of Figure 1: what SPARK returns. Writes paper/figures/fig1_panelG.pdf, which
paper/figures/fig1_panelG.tex places inside the Figure 1 tikzpicture.

Reading order is posterior matrix, then the inferred graph, then the class edge
probabilities. There is no arrow from the matrix to the graph, because the graph
takes its edges from the thresholded matrix but its colours from the coefficients.

Source of the data
------------------
simulation/output/eval/expert_elicited (eval.py), DAG FEATURED_SIM, through the
loaders in config.py. The matrix, the
thresholded edges, their coefficients and three of the five edge-probability
posteriors are that run.

This panel illustrates the method, so two things in it are illustrative rather
than measured. The chr2 subset carries no ChIP-Atlas and no TSS pairs, so those
two classes get invented Beta posteriors, in order that all four sources named in
panel B appear here too. And edge_recovery.csv stores coefficients only for the
true edges, so an inferred edge that is not one gets BETA_FALLBACK. Everything
else is the run's own output.

Node labels are deliberately absent: at this size they add nothing, and the
tripartite column headers carry what a reader needs.

Output is typeset through the pgf backend so the panel's fonts are the same
Computer Modern as the surrounding TikZ. That needs the `underscore` package.

Usage:
    python fig1_panelG.py
"""

import os
import sys

import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "font.family": "serif",
    "pgf.rcfonts": False,
    "pgf.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg  # noqa: E402

FEATURED_SIM = 3
SETTING = "expert_elicited"
THRESHOLD = 0.5

# Panel geometry, in inches. 15.9 cm wide to sit inside a 16.4 cm panel. The
# height is kept as low as the contents allow, because this is the tallest panel
# of Figure 1 and the figure competes with a long caption for the float page.
FIG_W, FIG_H = 15.9 / 2.54, 5.65 / 2.54

# The four sources of panel B plus the baseline class, numbered as panel B numbers
# them and in exactly the colours the Figure 1 preamble defines, so the edges here,
# the legend, the violins and panel B's matrix cells all agree. Four separate hues,
# not two light/dark pairs: shades of one hue read as a quantitative scale over
# the sources, which is not what the classes are.
CLASSES = [
    ("no evidence",      "#BFC6D0"),
    ("evidence source 1", "#E8871A"),
    ("evidence source 2", "#0E9594"),
    ("evidence source 3", "#C64191"),
    ("evidence source 4", "#6A4C93"),
]
# Where each row of CLASSES gets its posterior. An integer is a column of the
# run's rho_samples (prior codes 0, 1, 2); a (a, b) pair is an invented Beta, for
# the two sources the chr2 subset does not contain.
RHO_SOURCE = [0, 1, (7.0, 13.0), 2, (11.0, 9.0)]

# Edge colour is the posterior median edge coefficient, on the same red for
# activation and blue for repression as Figure 3 (effRed and effBlue in the
# preamble). The midpoint is a grey rather than white, so that an edge near zero
# stays visible against the panel.
BETA_CMAP = LinearSegmentedColormap.from_list(
    "beta", ["#2C6FAD", "#C8C8C8", "#C0392B"])
BETA_LIM = 0.5
BETA_FALLBACK = -0.15

KIND_COLOR = cfg.KIND_COLOR
FS_LABEL = 6.0
FS_TICK = 6.0
FS_TITLE = 7.5
EDGE_LW = 1.4      # every edge at one width
NET_SHIFT = 0.03  # figure fraction the graph moves left from its grid cell


def edge_coefs(P, thr=THRESHOLD):
    """Posterior median edge coefficient for every thresholded edge, from the
    run's edge_recovery.csv, with BETA_FALLBACK for an edge it does not list."""
    rec = pd.read_csv(f"{cfg.EVAL_BASE}/{SETTING}/edge_recovery.csv")
    rec = rec[rec["sim"] == FEATURED_SIM]
    med = {(int(i), int(j)): w for i, j, w in
           zip(rec["parent"], rec["child"], rec["w_post_median"])}
    src, dst = np.where(P >= thr)
    return {(i, j): med.get((i, j), BETA_FALLBACK) for i, j in zip(src, dst)}


def draw_heatmap(ax, P, nodes):
    order, n_tf, n_reg = cfg.display_order(nodes)
    M = cfg.reorder(P, order)
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    for b in (n_tf, n_reg):
        ax.axhline(b - 0.5, color="0.35", linewidth=0.4)
        ax.axvline(b - 0.5, color="0.35", linewidth=0.4)
    for s in ax.spines.values():
        s.set_linewidth(0.4)
        s.set_color("0.35")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("target $j$", fontsize=FS_TICK, labelpad=1.5)
    ax.set_ylabel("source $i$", fontsize=FS_TICK, labelpad=1.5)
    return im


def draw_network(ax, P, coefs, nodes, norm, thr=THRESHOLD):
    kind = nodes["kind"].to_numpy()
    pos = nodes["pos"].to_numpy()
    src, dst = np.where(P >= thr)
    used = np.unique(np.concatenate([src, dst]))

    col_x = {"TF": 0.0, "enhancer": 1.0, "gene": 2.0}
    cols = {k: sorted([i for i in used if kind[i] == k], key=lambda i: pos[i])
            for k in col_x}
    yv = {}

    def _assign(members):
        n = len(members)
        for r, i in enumerate(members):
            yv[i] = 1.0 - (r + 0.5) / n if n > 1 else 0.5

    # TF and gene columns in genomic order; the enhancer column ordered by the
    # mean height of its neighbours, which is what keeps the crossings down.
    _assign(cols["TF"])
    _assign(cols["gene"])
    nbr = {i: [] for i in used}
    for a, b in zip(src, dst):
        nbr[a].append(b)
        nbr[b].append(a)
    anchored = {i: yv[i] for i in cols["TF"] + cols["gene"]}
    cols["enhancer"].sort(key=lambda i: (
        -float(np.mean([anchored[j] for j in nbr[i] if j in anchored])), pos[i]))
    _assign(cols["enhancer"])
    xy = {i: (col_x[kind[i]], yv[i]) for i in used}

    # No alpha on the edges, so each lands at exactly its colourbar colour.
    #
    # One width for every edge. Width used to scale with the PIP, but the edges
    # drawn are only those already past the 0.5 threshold, so the encoding spanned
    # a narrow range that no scale explained, and it competed with colour.
    for (i, j), w in coefs.items():
        x0, y0 = xy[i]
        x1, y1 = xy[j]
        rad = 0.0 if x1 > x0 else -0.30
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>", mutation_scale=5.0,
            linewidth=EDGE_LW, color=BETA_CMAP(norm(w)),
            shrinkA=4.0, shrinkB=4.0, zorder=1))

    for i in used:
        x, y = xy[i]
        ax.scatter([x], [y], s=15, color=KIND_COLOR[kind[i]],
                   edgecolors="white", linewidths=0.5, zorder=3)

    for k, lab in (("TF", "TFs"), ("enhancer", "enhancers"), ("gene", "genes")):
        ax.text(col_x[k], 1.06, lab, ha="center", va="bottom",
                fontsize=FS_TITLE - 0.5, color=KIND_COLOR[k])

    ax.set_xlim(-0.22, 2.22)
    ax.set_ylim(-0.06, 1.20)
    ax.axis("off")


def draw_rho(ax, rho, rng):
    data = []
    for spec in RHO_SOURCE:
        if isinstance(spec, int):
            data.append(rho[::10, spec])
        else:
            data.append(rng.beta(spec[0], spec[1], size=rho[::10].shape[0]))

    ypos = list(range(len(CLASSES)))[::-1]
    parts = ax.violinplot(data, positions=ypos, vert=False, widths=0.84,
                          showextrema=False, showmedians=True)
    # The baseline is a pale grey, so at the 0.42 the saturated hues use it almost
    # disappears. Same colour, more opacity, rather than a different colour.
    for n, (body, (_, col)) in enumerate(zip(parts["bodies"], CLASSES)):
        body.set_facecolor(col)
        body.set_alpha(0.72 if n == 0 else 0.42)
        body.set_edgecolor(col)
        body.set_linewidth(0.7)
    parts["cmedians"].set_color("0.15")
    parts["cmedians"].set_linewidth(0.7)

    # Each violin is labelled in its class colour, since the graph no longer
    # carries a class legend.
    ax.set_yticks(ypos)
    ax.set_yticklabels([lab.replace("evidence ", "") for lab, _ in CLASSES],
                       fontsize=FS_LABEL - 0.5)
    for t, (_, col) in zip(ax.get_yticklabels(), CLASSES):
        t.set_color("0.45" if col == "#BFC6D0" else col)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(CLASSES) - 0.4)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"], fontsize=FS_TICK)
    ax.tick_params(axis="x", length=2, width=0.4, pad=1.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4)
    # The posterior of the same quantity panel E puts a prior on, so the label
    # carries the conditioning on the data that tells the two panels apart, with
    # the same \mathcal{Y} panel F's joint posterior uses.
    ax.set_xlabel(r"$\rho_k \mid \mathcal{Y}$", fontsize=FS_TICK, labelpad=1.0)


def main():
    nodes = cfg.load_nodes()
    P = cfg.load_posterior(SETTING, FEATURED_SIM)
    rho = np.load(f"{cfg.EVAL_BASE}/{SETTING}/rho_samples_sim{FEATURED_SIM}.npy")
    coefs = edge_coefs(P)
    norm = Normalize(vmin=-BETA_LIM, vmax=BETA_LIM)
    rng = np.random.default_rng(20260817)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.88, 0.84, 1.24],
                          left=0.035, right=0.995, top=0.900, bottom=0.187,
                          wspace=0.50)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_net = fig.add_subplot(gs[0, 1])
    ax_rho = fig.add_subplot(gs[0, 2])
    for ax in (ax_map, ax_net, ax_rho):
        ax.set_facecolor("none")
    # The graph is nudged left into the gap beside the heatmap, which once held an
    # arrow from the matrix to the graph.
    b = ax_net.get_position()
    ax_net.set_position([b.x0 - NET_SHIFT, b.y0, b.width, b.height])

    im = draw_heatmap(ax_map, P, nodes)
    draw_network(ax_net, P, coefs, nodes, norm)
    draw_rho(ax_rho, rho, rng)

    # Titles as figure text at one common height. Set as axes titles they drift
    # apart, because the square heatmap has a shorter axes box than its neighbours.
    for ax, lab in ((ax_map, "Every ordered pair"),
                    (ax_net, "Inferred graph"),
                    (ax_rho, "Posterior class edge probabilities")):
        b = ax.get_position()
        fig.text((b.x0 + b.x1) / 2, 0.970, lab, ha="center", va="top",
                 fontsize=FS_TITLE)

    cax = ax_map.inset_axes([0.0, -0.115, 1.0, 0.042])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.4)
    cb.set_ticks([0, 1])
    cb.ax.set_xticklabels(["0", "1"], fontsize=FS_TICK)
    cb.ax.tick_params(length=1.5, width=0.4, pad=1.0)
    cb.set_label("PIP", fontsize=FS_TICK, labelpad=1.0)

    # The coefficient colourbar sits at the PIP colourbar's height, so the two
    # read as one row. The heatmap's box is final only once its aspect is applied.
    ax_map.apply_aspect()
    bm = ax_map.get_position()
    bn = ax_net.get_position()
    w = 0.6 * bn.width
    cax_b = fig.add_axes([bn.x0 + (bn.width - w) / 2, bm.y0 - 0.115 * bm.height,
                          w, 0.042 * bm.height])
    cb_b = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=BETA_CMAP),
                        cax=cax_b, orientation="horizontal")
    cb_b.outline.set_linewidth(0.4)
    cb_b.set_ticks([-BETA_LIM, 0, BETA_LIM])
    cb_b.ax.set_xticklabels([f"$-{BETA_LIM:g}$", "0", f"{BETA_LIM:g}"],
                            fontsize=FS_TICK)
    cb_b.ax.tick_params(length=1.5, width=0.4, pad=1.0)
    cb_b.set_label(r"posterior median $\beta_{ij}$", fontsize=FS_TICK,
                   labelpad=1.0)

    out = os.path.join(cfg.FIG_DIR, "fig1_panelG.pdf")
    fig.savefig(out, transparent=True)
    plt.close(fig)
    print(f"wrote {out}  (DAG {FEATURED_SIM}, {SETTING}, "
          f"{len(coefs)} edges at {THRESHOLD})")


if __name__ == "__main__":
    main()
