"""
figures.py
==========
The simulation-study figures, from eval.py's output:

  study_overview_dag7_noninfo.png  Figure 2. For DAG 7 under the non-informative setting:
                                   the topological prior (A), the ground-truth DAG (B),
                                   the PIPs (C) and their precision-recall curve (D); then,
                                   over all ten DAGs, PR-AUC across prior settings and
                                   against DAG0 and the CellOracle-style regression (E),
                                   the RMSE of the class edge probabilities (F), and the
                                   direct effects of the called true edges (G).
  param_recovery_grid.png          Supplementary figure: node intercepts and
                                   overdispersions under each prior setting.

Panel E reads the comparators' predictions in benchmarking/output/.

The heatmaps of A to C share a grid, a node order (config.display_order) and a colour
scheme, so a cell holds the same pair of nodes throughout, and white always means a
forbidden pair.

Usage
-----
  python figures.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import config as cfg
from eval import average_precision

FEATURED_SETTING = "noninfo"
FEATURED_SIM = 7

# Type sizes for Figure 2. It goes into the manuscript at 0.85\textwidth, so everything
# here is read a good deal smaller than it is drawn.
TITLE_FS  = 13.5    # panel titles
LABEL_FS  = 12.5    # axis labels
BLOCK_FS  = 12.5    # the TFs / genes / enhancers block names on A to C
TICK_FS   = 11.5    # tick labels
LEGEND_FS = 10.5    # legend entries and the colourbar

# How far the legends under A to C are raised toward their heatmaps, as a fraction of the
# legend row's height, and the height of that row relative to the heatmap row.
LEGEND_LIFT = 0.22
LEGEND_ROW = 0.24
# Height in inches of row G, added below A to F.
BETA_H = 4.15


# ---------------------------------------------------------------------------
# Heatmap primitives (panels A to C)
# ---------------------------------------------------------------------------

def draw_categorical(ax, M, codes_to_color):
    """imshow a small-integer matrix, mapping each code in a contiguous run to a colour."""
    lo, hi = min(codes_to_color), max(codes_to_color)
    cmap = ListedColormap([codes_to_color[c] for c in range(lo, hi + 1)])
    norm = BoundaryNorm(np.arange(lo - 0.5, hi + 1.5), cmap.N)
    return ax.imshow(M, cmap=cmap, norm=norm, interpolation="nearest")


def edge_class_matrix(Q, A):
    """One display coding of prior and DAG: -1 forbidden, 0 permitted pair with no edge,
    1 + k an edge in evidence class k."""
    M = np.where(Q < 0, -1, 0)
    edges = (A == 1) & (Q >= 0)
    M[edges] = 1 + Q[edges]
    return M


def edge_class_colors():
    """Colours for edge_class_matrix(): white forbidden, grey permitted, then one colour per
    evidence class for the realized edges."""
    colors = {-1: cfg.FORBIDDEN_COLOR, 0: cfg.CANDIDATE_COLOR}
    for k in sorted(cfg.EVIDENCE_CLASSES):
        colors[1 + k] = (cfg.BASELINE_EDGE_COLOR if k == 0
                         else cfg.EVIDENCE_CLASSES[k]["color"])
    return colors


def edge_legend_handles(counts):
    """Legend for edge_class_matrix(): one entry per class with its edge count."""
    handles = []
    for k in sorted(cfg.EVIDENCE_CLASSES):
        color = (cfg.BASELINE_EDGE_COLOR if k == 0
                 else cfg.EVIDENCE_CLASSES[k]["color"])
        handles.append(Patch(facecolor=color, edgecolor="#999999", lw=0.4,
                             label=f"Edge: {cfg.EVIDENCE_CLASSES[k]['name']} ({counts[k]})"))
    handles.append(Patch(facecolor=cfg.CANDIDATE_COLOR, edgecolor="#999999",
                         lw=0.4, label="Permitted pair, no edge"))
    handles.append(Patch(facecolor=cfg.FORBIDDEN_COLOR, edgecolor="#999999",
                         lw=0.4, label="Forbidden pair"))
    return handles


def evidence_legend_handles(counts=None):
    """Legend for the evidence classes, optionally with the number of pairs in each, then
    the forbidden pair."""
    handles = []
    for k in sorted(cfg.EVIDENCE_CLASSES):
        meta = cfg.EVIDENCE_CLASSES[k]
        label = meta["name"]
        if counts is not None and k in counts:
            label += f" ({counts[k]})"
        handles.append(Patch(facecolor=meta["color"], edgecolor="#999999",
                             lw=0.4, label=label))
    handles.append(Patch(facecolor=cfg.FORBIDDEN_COLOR, edgecolor="#999999",
                         lw=0.4, label="Forbidden pair"))
    return handles


def draw_continuous(ax, P, forbidden):
    """imshow the PIPs with forbidden pairs masked out and drawn white."""
    cm = plt.get_cmap("Blues").copy()
    cm.set_bad(cfg.FORBIDDEN_COLOR)
    return ax.imshow(np.ma.masked_where(forbidden, P), cmap=cm,
                     vmin=0.0, vmax=1.0, interpolation="nearest")


def decorate(ax, n_tf, n_reg, n, sep_lw=0.9, frame_lw=0.6):
    """Block separators (TF/gene thin grey, regulator/enhancer black) and frame; no tick
    labels."""
    for b, color, lw in ((n_tf - 0.5, "#888888", sep_lw * 0.7),
                         (n_reg - 0.5, "black", sep_lw)):
        ax.axvline(b, color=color, lw=lw)
        ax.axhline(b, color=color, lw=lw)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(frame_lw)
        spine.set_edgecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])


def block_brackets(ax, n_tf, n_reg, n, fontsize, pad_x, pad_y,
                   labels=("TFs", "genes", "enhancers"), stagger=()):
    """Name the three node blocks just outside the axes, along the top edge and the left
    edge. Blocks in `stagger` are set one line further out, so the narrow TF and gene
    blocks' names do not run into each other."""
    bbox = ax.get_window_extent()
    dpi = ax.figure.dpi
    line_x = 1.3 * fontsize * dpi / 72.0 / max(bbox.width, 1.0)
    line_y = 1.3 * fontsize * dpi / 72.0 / max(bbox.height, 1.0)

    spans = [(0, n_tf), (n_tf, n_reg), (n_reg, n)]
    for i, ((lo, hi), text) in enumerate(zip(spans, labels)):
        mid = ((lo + hi) / 2) / n
        out = i in stagger
        ax.annotate(text, xy=(mid, 1 + pad_y + (line_y if out else 0.0)),
                    xycoords="axes fraction", ha="center", va="bottom",
                    fontsize=fontsize, color="#444444")
        # Axes-fraction y runs bottom-up while the matrix runs top-down.
        ax.annotate(text, xy=(-pad_x - (line_x if out else 0.0), 1 - mid),
                    xycoords="axes fraction", ha="right", va="center",
                    rotation=90, fontsize=fontsize, color="#444444")


def _outline_true_edges(ax, A, lw=0.55, color="#D62728"):
    """Outline every true edge on the posterior panel, so a bright cell inside an outline
    is a hit, an empty outline a miss, and a bright cell with no outline a false
    positive."""
    for i, j in zip(*np.nonzero(A)):
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor=color, lw=lw, zorder=3))


# ---------------------------------------------------------------------------
# Panels D to G
# ---------------------------------------------------------------------------

def _pr_curve(y_true, y_score):
    """Precision-recall curve whose area is eval.average_precision: tie-aware, one point
    per recall level, with a leading point at recall 0. Draw it with steps-pre."""
    y_true  = np.asarray(y_true,  dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    order   = np.argsort(-y_score, kind="mergesort")
    y_true, y_score = y_true[order], y_score[order]
    end = np.r_[np.where(np.diff(y_score))[0], y_true.size - 1]
    tp  = np.cumsum(y_true)[end]
    fp  = (end + 1) - tp
    precision = tp / (tp + fp)
    recall    = tp / tp[-1]
    # Keep only the first point at each recall level: the later ones span zero width and
    # drawn as steps would put a spur at the foot of every tread.
    keep = np.r_[True, np.diff(recall) > 0]
    recall, precision = recall[keep], precision[keep]
    return np.r_[0.0, recall], np.r_[precision[0], precision]


def _draw_pr_curve(ax, A, P, forbidden):
    """Panel D, scored over the permitted pairs as eval.py does. The dotted line is the
    prevalence of true edges among them; the point is the precision and recall at the PIP
    threshold."""
    cand = (~forbidden) & ~np.eye(A.shape[0], dtype=bool)
    y_true, y_score = A[cand].astype(float), P[cand]

    recall, precision = _pr_curve(y_true, y_score)
    ax.step(recall, precision, where="pre", color="#1F4E9C", lw=1.8, zorder=3)
    prevalence = float(y_true.mean())
    ax.axhline(prevalence, color="gray", ls=":", lw=1.1, zorder=2,
               label=f"chance ({prevalence:.2f})")
    called = y_score >= cfg.CALL_THRESHOLD
    if called.any():
        tp = float((y_true[called]).sum())
        ax.plot(tp / y_true.sum(), tp / called.sum(), "o", ms=7, color="black",
                zorder=4, label=f"posterior inclusion\nprobability {cfg.CALL_THRESHOLD:g}")

    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall", fontsize=LABEL_FS)
    ax.set_ylabel("Precision", fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(ls="--", alpha=0.3)
    ax.legend(fontsize=LEGEND_FS, loc="lower left", bbox_to_anchor=(0.0, 0.10),
              frameon=False)


def _draw_benchmark_panel(ax):
    """Panel E: PR-AUC of every DAG under SPARK's four prior settings, then DAG0 and the
    regression, one line per DAG joined across the methods, with the across-DAG mean.
    A bracket over each group and a rule between them mark which columns are SPARK under
    a prior setting and which are separate methods."""
    methods = cfg.SETTINGS + cfg.COMPARATORS
    Q = cfg.load_prior_codes()
    cand = ~np.eye(Q.shape[0], dtype=bool) & (Q != -1)
    pr = {m: {d: average_precision(cfg.load_true_adjacency(d)[cand].ravel(),
                                   cfg.load_prediction(m, d)[cand].ravel())
              for d in range(cfg.N_DAGS)} for m in methods}

    dags = list(range(cfg.N_DAGS))
    colors = {d: plt.cm.tab10(i % 10) for i, d in enumerate(dags)}
    x = np.arange(len(methods))
    for d in dags:
        ax.plot(x, [pr[m][d] for m in methods], "-o", ms=4, lw=0.8, alpha=0.55,
                color=colors[d], zorder=2)
    means = [np.nanmean([pr[m][d] for d in dags]) for m in methods]
    ax.plot(x, means, "-D", ms=7, lw=2.0, color="black", zorder=4,
            label="across-DAG mean")
    ax.set_xticks(x)
    ax.set_xticklabels([cfg.METHOD_SHORT_LABELS[m] for m in methods],
                       rotation=40, ha="right", fontsize=TICK_FS)
    ax.set_xlim(-0.35, len(methods) - 0.65)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("PR-AUC", fontsize=LABEL_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.grid(axis="y", ls="--", alpha=0.3)
    ax.legend(fontsize=LEGEND_FS, loc="lower left", frameon=False)

    cut, y = len(cfg.SETTINGS), 1.035
    ax.axvline(cut - 0.5, color="0.55", ls="--", lw=0.9, zorder=1)
    for lo, hi, text in [(0, cut - 1, "SPARK, by prior setting"),
                         (cut, len(methods) - 1, "Comparator methods")]:
        ax.annotate("", xy=(lo - 0.28, y), xytext=(hi + 0.28, y),
                    xycoords=("data", "axes fraction"),
                    textcoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="-", color="0.35", lw=0.9,
                                    shrinkA=0, shrinkB=0))
        ax.text((lo + hi) / 2, y + 0.015, text, ha="center", va="bottom",
                fontsize=LEGEND_FS, color="0.25",
                transform=ax.get_xaxis_transform(which="grid"))


def _draw_rho_rmse(ax, settings):
    """Panel F: RMSE of the posterior median class edge probabilities against their
    generating values over the ten DAGs, one line per class, under the settings with
    evidence classes."""
    rmse = {k: [] for k in cfg.TRUE_RHO}
    for s in settings:
        meds = np.array([np.median(cfg.load_rho(s, sim), axis=0)
                         for sim in range(cfg.N_DAGS)])
        for k in cfg.TRUE_RHO:
            rmse[k].append(np.sqrt(np.mean((meds[:, k] - cfg.TRUE_RHO[k]) ** 2)))

    x = np.arange(len(settings))
    style = {1: ("JASPAR2024", cfg.EVIDENCE_CLASSES[1]["color"], "o"),
             2: ("PCHi-C",     cfg.EVIDENCE_CLASSES[2]["color"], "s"),
             0: ("No evidence", cfg.BASELINE_EDGE_COLOR, "D")}
    for k in [1, 2, 0]:
        label, color, marker = style[k]
        ax.plot(x, rmse[k], "-", marker=marker, ms=8, lw=2.0, color=color,
                markeredgecolor="white", markeredgewidth=1.0, label=label, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([cfg.SETTING_LABELS[s] for s in settings],
                       rotation=40, ha="right", fontsize=TICK_FS)
    ax.set_xlim(-0.35, len(settings) - 0.65)
    ax.set_ylim(0, 0.6)
    ax.set_ylabel("RMSE", fontsize=LABEL_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.grid(axis="y", ls="--", alpha=0.3)
    ax.legend(fontsize=LEGEND_FS, loc="upper left", frameon=False)


def _draw_beta_row(sfig, settings):
    """Panel G: posterior median against simulated direct effect for the true edges at
    PIP >= the call threshold, one facet per setting, sharing axes."""
    t, m, lo, hi = "true_w", "w_post_median", "w_post_lower95", "w_post_upper95"
    rec = {}
    for s in settings:
        df = pd.read_csv(f"{cfg.EVAL_BASE}/{s}/edge_recovery.csv")
        rec[s] = df[df["post_prob"] >= cfg.CALL_THRESHOLD]
    v = np.concatenate([rec[s][c].to_numpy(float) for s in settings for c in (t, lo, hi)])
    pad = 0.05 * (v.max() - v.min())
    lim = (v.min() - pad, v.max() + pad)

    axes = sfig.subplots(1, len(settings), sharex=True, sharey=True)
    for ax, s in zip(axes, settings):
        df = rec[s]
        ax.plot(lim, lim, ls="--", lw=0.8, color=cfg.REF_COLOR, zorder=1)
        ax.vlines(df[t], df[lo], df[hi], color=cfg.CI_COLOR, lw=0.6, zorder=2)
        ax.scatter(df[t], df[m], color=cfg.POINT_COLOR, s=24, alpha=0.6,
                   edgecolors="white", linewidths=0.4, zorder=3)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        r = np.corrcoef(df[t], df[m])[0, 1]
        ax.text(0.04, 0.96, f"$r$ = {r:.2f}\n$n$ = {len(df)}",
                transform=ax.transAxes, ha="left", va="top", fontsize=LEGEND_FS)
        ax.tick_params(labelsize=TICK_FS)
        ax.grid(ls=":", lw=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_title(cfg.SETTING_LABELS_FLAT[s], fontsize=LABEL_FS, pad=6)
        ax.set_xlabel("Simulated effect", fontsize=LABEL_FS)
    axes[0].set_ylabel(r"Posterior median $\beta_{ij}$", fontsize=LABEL_FS)
    sfig.suptitle("G.  Direct effects across prior settings", fontsize=TITLE_FS)


# ---------------------------------------------------------------------------
# Figure 2
# ---------------------------------------------------------------------------

def figure_overview(setting=FEATURED_SETTING, sim=FEATURED_SIM,
                    name="study_overview_dag7_noninfo.png"):
    settings = cfg.SETTINGS
    nodes = cfg.load_nodes()
    order, n_tf, n_reg = cfg.display_order(nodes)
    Q = cfg.reorder(cfg.load_prior_codes(), order)
    forbidden = Q < 0
    A = cfg.reorder(cfg.load_true_adjacency(sim), order)
    P = cfg.reorder(cfg.load_posterior(setting, sim), order)
    n = Q.shape[0]

    offdiag = ~np.eye(n, dtype=bool)
    cand_counts = {k: int(((Q == k) & offdiag).sum()) for k in cfg.EVIDENCE_CLASSES}
    edge_counts = {k: int(((Q == k) & (A == 1) & offdiag).sum())
                   for k in cfg.EVIDENCE_CLASSES}

    # Rows: the heatmaps A to C, their legends, then D to F; row G is a subfigure of its
    # own below. Twelve columns so the lower panels can take their own widths.
    lower = 1.0
    height = 10.0 * (1.0 + LEGEND_ROW + lower) / (1.30 + 0.72)
    fig = plt.figure(figsize=(13.2, height + BETA_H), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[height, BETA_H], hspace=0.02)
    gs = top.add_gridspec(3, 12, height_ratios=[1.0, LEGEND_ROW, lower])
    axes  = [top.add_subplot(gs[0, 4 * i:4 * (i + 1)]) for i in range(3)]
    laxes = [top.add_subplot(gs[1, 4 * i:4 * (i + 1)]) for i in range(3)]
    # E carries six columns, so it gets a wider share of the row.
    sub = gs[2, :].subgridspec(1, 3, width_ratios=[1, 1.5, 1], wspace=0.05)
    ax_pr      = top.add_subplot(sub[0, 0])
    ax_setting = top.add_subplot(sub[0, 1])
    ax_rho     = top.add_subplot(sub[0, 2])
    for lax in laxes:
        lax.axis("off")

    prior_colors = {-1: cfg.FORBIDDEN_COLOR}
    prior_colors.update({k: v["color"] for k, v in cfg.EVIDENCE_CLASSES.items()})
    draw_categorical(axes[0], Q, prior_colors)
    draw_categorical(axes[1], edge_class_matrix(Q, A), edge_class_colors())
    im = draw_continuous(axes[2], P, forbidden)
    _outline_true_edges(axes[2], A)

    # No node names: 51 of them do not fit, and the panels are read block by block.
    for ax in axes:
        decorate(ax, n_tf, n_reg, n)
        # stagger=(1,) sets "genes" one line out, clear of "TFs" beside it.
        block_brackets(ax, n_tf, n_reg, n, fontsize=BLOCK_FS, pad_x=0.02,
                       pad_y=0.012, stagger=(1,))
        ax.set_xlabel("target  (column $j$)", fontsize=LABEL_FS, labelpad=3)
    axes[0].set_ylabel("source  (row $i$)", fontsize=LABEL_FS, labelpad=24)
    # pad clears the staggered "genes" block name, which sits above the axes.
    axes[0].set_title("A.  Topological prior", fontsize=TITLE_FS, pad=36)
    axes[1].set_title("B.  Representative ground-truth DAG", fontsize=TITLE_FS, pad=36)
    axes[2].set_title("C.  Posterior inclusion probabilities,\nrepresentative DAG",
                      fontsize=TITLE_FS, pad=36)

    # Legends of A to C, raised by LEGEND_LIFT toward their heatmaps and kept out of the
    # layout, which would otherwise grow the row to hold them.
    leg_a = laxes[0].legend(handles=evidence_legend_handles(cand_counts),
                            fontsize=LEGEND_FS, loc="upper center", frameon=False,
                            ncol=1, handlelength=1.2, labelspacing=0.5,
                            bbox_to_anchor=(0.5, 1.0 + LEGEND_LIFT))
    leg_b = laxes[1].legend(handles=edge_legend_handles(edge_counts),
                            fontsize=LEGEND_FS, loc="upper center", frameon=False,
                            ncol=1, handlelength=1.2, labelspacing=0.5,
                            bbox_to_anchor=(0.5, 1.0 + LEGEND_LIFT))
    cax = laxes[2].inset_axes([0.10, 0.80 + LEGEND_LIFT, 0.80, 0.12])
    cb = top.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Posterior inclusion probability", fontsize=LEGEND_FS)
    cb.ax.tick_params(labelsize=LEGEND_FS - 1, length=2)
    leg_c = laxes[2].legend(
        handles=[Line2D([0], [0], marker="s", ls="", markerfacecolor="none",
                        markeredgecolor="#D62728", markeredgewidth=1.1,
                        markersize=7, label="True edge (outlined)"),
                 evidence_legend_handles()[-1]],
        fontsize=LEGEND_FS, loc="lower center", frameon=False, ncol=2,
        handlelength=1.2, columnspacing=1.4, bbox_to_anchor=(0.5, LEGEND_LIFT))
    for artist in (leg_a, leg_b, leg_c, cax):
        artist.set_in_layout(False)

    # D to F, with titles on two lines at a common pad so the three sit level above E's
    # group brackets.
    _draw_pr_curve(ax_pr, A, P, forbidden)
    ax_pr.set_title("D.  Precision-recall curve,\nrepresentative DAG", fontsize=TITLE_FS, pad=34)
    _draw_benchmark_panel(ax_setting)
    ax_setting.set_title("E.  PR-AUC for ten DAGs across\nprior settings and methods",
                         fontsize=TITLE_FS, pad=34)
    # The setting with no evidence classes has no class edge probabilities to score.
    _draw_rho_rmse(ax_rho, [s for s in settings if s != "no_partition"])
    ax_rho.set_title("F.  RMSE for class edge probabilities\nacross prior settings",
                     fontsize=TITLE_FS, pad=34)

    _draw_beta_row(bottom, settings)
    cfg.savefig(fig, name)


# ---------------------------------------------------------------------------
# Supplementary figure: node intercepts and overdispersions
# ---------------------------------------------------------------------------

def figure_param_recovery_grid(name="param_recovery_grid.png"):
    """Generating against posterior median node intercept alpha_j, and the posterior
    medians of psi_j as a histogram; rows are parameters, columns prior settings, every
    mark pooled over the ten DAGs. Each alpha_j panel carries the pooled Pearson r."""
    settings = cfg.SETTINGS
    rec = {s: (pd.read_csv(f"{cfg.EVAL_BASE}/{s}/b_recovery.csv"),
               pd.read_csv(f"{cfg.EVAL_BASE}/{s}/psi_recovery.csv")) for s in settings}

    def lims(cols):
        v = np.concatenate([np.asarray(c, float) for c in cols])
        pad = 0.05 * (v.max() - v.min())
        return v.min() - pad, v.max() + pad

    # The alpha axes fit the points; the few long intervals run off the panel.
    b_lim = lims([c for b, _ in rec.values() for c in (b["b_true"], b["b_post_median"])])
    psi_max = max(p["psi_post_median"].max() for _, p in rec.values())
    psi_bins = np.linspace(0, np.ceil(psi_max * 10) / 10, 31)

    with plt.rc_context({"font.size": 8}):
        fig = plt.figure(figsize=(9.0, 5.4), layout="constrained")
        top, bottom = fig.subfigures(2, 1, hspace=0.04)
        axes_b = top.subplots(1, len(settings), sharex=True, sharey=True)
        axes_p = bottom.subplots(1, len(settings), sharex=True, sharey=True)
        top.suptitle(r"Node intercepts $\alpha_j$", fontsize=10)
        bottom.suptitle(r"Overdispersions $\psi_j$", fontsize=10)
        for c, s in enumerate(settings):
            b, p = rec[s]
            ax_b, ax_p = axes_b[c], axes_p[c]

            t, m, lo, hi = "b_true", "b_post_median", "b_post_lower95", "b_post_upper95"
            ax_b.plot(b_lim, b_lim, ls="--", lw=0.8, color=cfg.REF_COLOR, zorder=1)
            ax_b.vlines(b[t], b[lo], b[hi], color=cfg.CI_COLOR, lw=0.5, zorder=2)
            ax_b.scatter(b[t], b[m], color=cfg.POINT_COLOR, s=8, edgecolors="white",
                         linewidths=0.3, zorder=3)
            ax_b.set_xlim(b_lim)
            ax_b.set_ylim(b_lim)
            ax_b.set_aspect("equal")
            r_pooled = np.corrcoef(b[t], b[m])[0, 1]
            ax_b.text(0.04, 0.96, f"$r$ = {r_pooled:.2f}", transform=ax_b.transAxes,
                      ha="left", va="top", fontsize=7)

            ax_p.hist(p["psi_post_median"], bins=psi_bins, color=cfg.POINT_COLOR,
                      edgecolor="white", linewidth=0.3)
            ax_p.axvline(p["psi_true"].iloc[0], color="black", lw=1.2,
                         label=r"True $\psi_j = 0.1$")
            ax_p.set_xlim(psi_bins[0], psi_bins[-1])

            for ax in (ax_b, ax_p):
                ax.grid(ls=":", lw=0.5, alpha=0.6)
                ax.set_axisbelow(True)
                ax.set_title(cfg.SETTING_LABELS_FLAT[s], fontsize=8.5)
            ax_b.set_xlabel(r"Generating $\alpha_j$")
            ax_p.set_xlabel(r"Posterior median $\psi_j$")
            if c == 0:
                ax_b.set_ylabel(r"Posterior median $\alpha_j$")
                ax_p.set_ylabel("Nodes")
                ax_p.legend(fontsize=7, frameon=False, loc="upper right")
        cfg.savefig(fig, name)


if __name__ == "__main__":
    figure_overview()
    figure_param_recovery_grid()
