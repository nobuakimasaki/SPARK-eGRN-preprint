"""
gwas_panel.py
=============
Figure 3 (panel_network_gwas.png / .pdf): the regulatory circuitry of the immGWAS
enhancers in the inferred network.

  A  the whole inferred network, every edge at PIP >= config.PIP_CUT
     (network._render_network);
  B  the PIPs of the TF -> immGWAS enhancer pairs, over the TFs with an edge at
     PIP >= COMBINED_TF_CUT into an immGWAS enhancer;
  C  the PIPs of the immGWAS enhancer -> target pairs, over the enhancers with >= 1
     permitted outgoing pair and the targets reachable from them, with forbidden pairs
     crossed out;
  D  the cis-regulatory unit of IL1A (unit._render_unit).

The enhancer axes are ordered by disease category (curated clinical families, with the
pleiotropic enhancers as a trailing block) and along the genome within a category; TF rows
are alphabetical; targets are TFs first, then genes, each along the genome by TSS. Every
axis order is fixed and independent of the fit.

Inputs
------
config.EVAL_DIR: edge_recovery.csv, nodes.csv, posterior_dir.npy, W_median.npy (eval.py);
the topological prior, the node table and the GWAS annotation tables (config).

Outputs
-------
config.FIG_DIR/panel_network_gwas.png (+ .pdf), and the matrices behind panels B and C in
config.OUT_DIR/gwas/ (tf_gwas_matrix.csv, gwas_enh_to_target_matrix.csv).

Usage
-----
  python gwas_panel.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.patches import Patch

import config
import network
import unit

# Target-type colours, taken from network.py so a TF is the same colour in panel C's
# target names as in panel A's nodes.
TF_COLOR, GENE_COLOR = network.NODE_TF_COLOR, network.NODE_GENE_COLOR

# Heatmap label sizing. The figure goes into the paper at \textwidth from a canvas several
# times that wide, so every label is scaled down by roughly a quarter on the page; the
# sizes here are set large enough to survive that. ROW_H is the vertical inches allotted
# per heatmap row so many-row axes never overlap.
ENH_LABEL_FS   = 17   # abbreviated chr…:… Mb enhancer names
TF_LABEL_FS    = 17   # TF / gene names
STRIP_LABEL_FS = 19   # disease-category names beside the annotation strips
AXIS_LABEL_FS  = 27   # axis titles
CBAR_LABEL_FS  = 21   # colourbar titles and their tick labels
LEGEND_FS      = 19   # legend entries, with the legend title a shade larger
ROW_H          = 0.36
# Vertical inches reserved for panel C's target-type key.
LEGEND_ROW_H   = 0.9
# Gap between gridspec columns, as the constrained layout engine's `wspace`, set below the
# 0.02 default to close up the run between the disease strips and the heatmap they annotate.
LAYOUT_WSPACE  = 0.008

CBAR_LABEL = "Posterior edge probability"
# Vertical inches per disease-category strip in panel B.
STRIP_H        = 0.46

HEATMAP_CMAP = network.POST_CMAP

# Forbidden ordered pairs. A pair the topological prior forbids has PIP exactly 0 by
# construction, which the ramp would draw at its own pale end, indistinguishable from a
# permitted pair the posterior merely rejected. Forbidden cells are instead drawn white and
# crossed, the way Figure 1 marks them (geometry and colours of fig1_panelB.tex).
FORBID_FACE  = "#FFFFFF"
FORBID_CROSS = "#C7C7C7"
FORBID_SPAN  = 0.59   # cross arm length as a fraction of the cell
FORBID_LW    = 1.0    # cross stroke width in points at this canvas

GWAS_OUT_DIR = os.path.join(config.OUT_DIR, "gwas")


def load_prior_matrix():
    return network.load_prior_matrix()


def _chrom_rank(chrom):
    """Sort rank of a chromosome name: chr1..chr22 by number, then X, Y, MT, then
    anything unrecognised."""
    s = str(chrom)
    s = s[3:] if s.lower().startswith("chr") else s
    if s.isdigit():
        return int(s)
    return {"X": 23, "Y": 24, "M": 25, "MT": 25}.get(s.upper(), 98)


def load_gene_loci():
    """{gene -> (chromosome rank, TSS position)} from the node table, used to lay panel C's
    target columns out along the genome."""
    df = pd.read_csv(config.NODES_PATH)
    loci = {}
    for name, chrom, pos in zip(df["node"], df["chr"], df["pos"]):
        if pd.isna(chrom) or pd.isna(pos):
            continue
        loci[name] = (_chrom_rank(chrom), int(pos))
    return loci


def _genomic_order(names, loci):
    """`names` along the genome, by (chromosome, position); unplaced names sort last, and
    ties break on the name so the order is deterministic."""
    return sorted(names, key=lambda n: (*loci.get(n, (99, 0)), n))


# ---------------------------------------------------------------------------
# immGWAS enhancer -> immune disease categories (network.PEAK_TRAITS). A single study can
# carry several immune disease traits, so an enhancer may belong to several categories;
# the annotation strips mark every one of them.
# ---------------------------------------------------------------------------
peak_cats = network.PEAK_TRAITS
all_gwas_peaks = sorted(peak_cats.keys())

# Curated clinical grouping so similar diseases sit adjacent on the trait axis. Traits are
# listed by disease family in display order; any trait not in this list falls back to
# alphabetical after the listed ones. A multi-trait enhancer whose traits all sit in ONE
# family stays inside that family's block (see _peak_group).
TRAIT_FAMILIES = [
    ("Inflammatory bowel / gut",
     ["Crohn disease", "ulcerative colitis", "inflammatory bowel disease", "celiac disease"]),
    ("Atopy / allergy / respiratory",
     ["allergic disease", "allergic rhinitis", "asthma", "childhood onset asthma",
      "atopic eczema", "food allergy"]),
    ("Skin autoimmune (blistering)",
     ["bullous pemphigoid"]),
    ("Systemic / connective-tissue autoimmune",
     ["systemic lupus erythematosus", "Sjogren syndrome", "mixed connective tissue disease"]),
    ("Neurological autoimmune",
     ["multiple sclerosis", "myasthenia gravis"]),
    ("Endocrine autoimmune",
     ["type 1 diabetes mellitus", "type 1 diabetes nephropathy"]),
    # Last of the named families, so its block sits next to the cross-family pleiotropic
    # enhancers that follow it -- most of those carry psoriasis / ankylosing spondylitis.
    ("Psoriasis / seronegative spondyloarthropathy / joint",
     ["psoriasis", "psoriasis vulgaris", "ankylosing spondylitis", "rheumatoid arthritis"]),
]
TRAIT_FAMILY_ORDER = [t for _, traits in TRAIT_FAMILIES for t in traits]
_FAMILY_RANK = {t: i for i, t in enumerate(TRAIT_FAMILY_ORDER)}
_TRAIT_FAMILY = {t: fam for fam, traits in TRAIT_FAMILIES for t in traits}

# Colours come from network.TRAIT_COLORS (alphabetical rank over all categories), so a
# disease is the same colour in every panel; the display order below does not affect them.
_COLOR_ORDER    = sorted({c for cats in peak_cats.values() for c in cats})
CATEGORY_COLORS = network.TRAIT_COLORS
# Display order (annotation strips, enhancer sorting): curated family order, then any
# unlisted trait alphabetically.
CATEGORY_ORDER  = sorted(_COLOR_ORDER,
                         key=lambda c: (_FAMILY_RANK.get(c, len(TRAIT_FAMILY_ORDER)), c))

# Pseudo-categories for enhancers that do not belong to a single trait block. MULTI_CAT
# collects the enhancers whose traits span several families, which go at the end of the
# axis; NO_TRAIT_CAT covers a peak whose study carries no immune disease trait.
MULTI_CAT    = "Multiple"
NO_TRAIT_CAT = "No GWAS hit"

_GROUP_ORDER = CATEGORY_ORDER + [MULTI_CAT, NO_TRAIT_CAT]
_GROUP_RANK  = {g: i for i, g in enumerate(_GROUP_ORDER)}


def _peak_group(peak):
    """Display group of an immGWAS enhancer -- the one block it occupies on the enhancer
    axis. A single-trait enhancer gets its trait; an enhancer whose traits all belong to the
    same clinical family gets its earliest such trait; an enhancer whose traits span several
    families goes to MULTI_CAT. The annotation strips still mark every trait it carries."""
    cats = sorted((c for c in peak_cats.get(peak, ()) if c in CATEGORY_ORDER),
                  key=lambda c: _GROUP_RANK[c])
    if not cats:
        return NO_TRAIT_CAT
    families = {_TRAIT_FAMILY.get(c) for c in cats}
    if len(cats) == 1 or (len(families) == 1 and None not in families):
        return cats[0]
    return MULTI_CAT


def _peak_locus(peak):
    """(chromosome rank, start coordinate) of a peak named 'chr2-112831548-112837357'."""
    chrom, start, _ = peak.split("-", 2)
    return (_chrom_rank(chrom), int(start))


def _sort_key(peak):
    """Enhancer display order: disease category first, then along the genome."""
    return (_GROUP_RANK[_peak_group(peak)], *_peak_locus(peak))


def _annotation(peaks):
    """(categories, ann) for `peaks`: the binary disease-category annotation matrix
    (n_cats x n_peaks), restricted to the categories that have a member peak."""
    ann_full = np.array([
        [1.0 if cat in peak_cats[p] else 0.0 for p in peaks]
        for cat in CATEGORY_ORDER
    ]) if peaks else np.zeros((len(CATEGORY_ORDER), 0))
    present = [k for k in range(len(CATEGORY_ORDER)) if ann_full[k].any()]
    return ([CATEGORY_ORDER[k] for k in present],
            ann_full[present] if present else ann_full[:0])


def dataset_context():
    """Node sets and immGWAS-enhancer ordering: active_tfs, downstream_genes, the immGWAS
    enhancers in the node set (ordered by disease category) with their annotation matrix,
    the subset of those enhancers that can have an outgoing edge at all with its own
    annotation matrix, the targets reachable from them, and the forbidden pairs."""
    Q_df = load_prior_matrix()
    nodes     = Q_df.columns.tolist()
    tfs       = [n for n in nodes if not n.startswith("chr")]
    enhancers = [n for n in nodes if n.startswith("chr")]
    # Active TFs: at least one non-forbidden prior pair to any enhancer.
    active_tfs = [tf for tf in tfs if (Q_df.loc[tf, enhancers] != -1).any()]
    downstream_genes = [tf for tf in tfs if tf not in set(active_tfs)]

    enh_set = set(enhancers)
    gwas_peaks_sorted = sorted((p for p in all_gwas_peaks if p in enh_set), key=_sort_key)
    categories, ann = _annotation(gwas_peaks_sorted)

    # Enhancers with at least one non-forbidden pair to a non-enhancer node. An enhancer
    # with none would only ever be an all-zero row of panel C, so it is dropped there; it
    # stays in panel B, where it is the child.
    peaks_out = [p for p in gwas_peaks_sorted if (Q_df.loc[p, tfs] != -1).any()]
    categories_out, ann_out = _annotation(peaks_out)
    # ... and the targets reachable from those enhancers; anything else would only ever be
    # an all-zero column.
    targets_out = set(t for t in tfs if (Q_df.loc[peaks_out, t] != -1).any()) if peaks_out \
        else set()
    return dict(active_tfs=active_tfs, downstream_genes=downstream_genes,
                gwas_peaks_sorted=gwas_peaks_sorted, categories=categories, ann=ann,
                peaks_out=peaks_out, categories_out=categories_out, ann_out=ann_out,
                targets_out=targets_out, forbidden=(Q_df == -1))


def _matrix_from_edges(edges, row_names, col_names):
    """PIP matrix (rows x cols) filled from an edge list with columns parent, child,
    post_prob; missing edges are 0."""
    mat = pd.DataFrame(0.0, index=row_names, columns=col_names)
    sub = edges[edges["parent"].isin(row_names) & edges["child"].isin(col_names)]
    for _, r in sub.iterrows():
        mat.loc[r["parent"], r["child"]] = r["post_prob"]
    return mat


# ---------------------------------------------------------------------------
# Panel B: TF -> immGWAS enhancer
# ---------------------------------------------------------------------------

def _tf_gwas_figsize(n_tfs, n_peaks, n_cats=0):
    """Natural size (inches) of panel B: height scales with the TF rows plus a fixed slice
    per disease strip, width with the enhancer columns."""
    return max(12, n_peaks * 0.7), max(7, n_tfs * ROW_H) + n_cats * STRIP_H


def _render_tf_gwas(fig, mat_sorted, active_tfs, gwas_peaks_sorted, ann, categories,
                    cbar_kw=None):
    """Draw panel B (the heatmap and its disease-category strips) into `fig`, a SubFigure.
    Returns the axes, top strip first."""
    n_tfs, n_peaks, n_cats = len(active_tfs), len(gwas_peaks_sorted), len(categories)
    axes = fig.subplots(
        n_cats + 1, 1,
        gridspec_kw={"height_ratios": [STRIP_H] * n_cats + [max(7, n_tfs * ROW_H)],
                     "hspace": 0.05},
        sharex=True)
    for k, cat in enumerate(categories):
        ax = axes[k]
        ax.imshow(ann[k:k + 1], aspect="auto",
                  cmap=ListedColormap(["#ebebeb", CATEGORY_COLORS[cat]]),
                  vmin=0, vmax=1, interpolation="nearest")
        ax.set_yticks([0])
        ax.set_yticklabels([network._trait_display(cat)], fontsize=STRIP_LABEL_FS)
        ax.tick_params(left=False, bottom=False, labelbottom=False, length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    ax_hm = axes[n_cats]
    im = ax_hm.imshow(mat_sorted.values, aspect="auto", cmap=HEATMAP_CMAP,
                      vmin=0, vmax=1, interpolation="nearest")
    shown = network._enhancer_short_labels(gwas_peaks_sorted)
    ax_hm.set_xticks(np.arange(n_peaks))
    ax_hm.set_xticklabels([shown[p] for p in gwas_peaks_sorted], rotation=45,
                          rotation_mode="anchor", ha="right", va="top",
                          fontsize=ENH_LABEL_FS)
    ax_hm.set_yticks(np.arange(n_tfs)); ax_hm.set_yticklabels(active_tfs, fontsize=TF_LABEL_FS)
    ax_hm.set_xlabel("immGWAS enhancer", fontsize=AXIS_LABEL_FS)
    ax_hm.set_ylabel("TF", fontsize=AXIS_LABEL_FS)
    cb = fig.colorbar(im, ax=ax_hm, **{"shrink": 0.5, "pad": 0.01, **(cbar_kw or {})})
    cb.set_label(CBAR_LABEL, fontsize=CBAR_LABEL_FS)
    cb.ax.tick_params(labelsize=CBAR_LABEL_FS)
    return axes


# ---------------------------------------------------------------------------
# Panel C: immGWAS enhancer -> target
# ---------------------------------------------------------------------------

def _strip_label_height(categories):
    """Inches of headroom the diagonal disease names need above the annotation strips,
    estimated from the character count of the longest name at 45 degrees."""
    longest = max((len(network._trait_display(c)) for c in categories), default=0)
    return max(1.0, 0.707 * 0.55 * (STRIP_LABEL_FS / 72.0) * longest)


def _enh_target_figsize(n_peaks, n_targets, categories=()):
    """Natural size (inches) of panel C: the heatmap rows plus the diagonal disease names
    above and the target-type key below."""
    heat_h = max(4.0, n_peaks * 0.5)
    return (max(8, n_targets * 0.5 + 3),
            _strip_label_height(categories) + heat_h + LEGEND_ROW_H + 2.0)


def _forbidden_mask(mat_sorted, forbidden):
    """Boolean array over `mat_sorted`'s cells, True where the topological prior forbids the
    ordered pair (`forbidden` is a frame over the whole node set, read by label)."""
    if forbidden is None:
        return np.zeros(mat_sorted.shape, dtype=bool)
    return np.array([[bool(forbidden.loc[r, c]) for c in mat_sorted.columns]
                     for r in mat_sorted.index])


def _forbidden_cmap():
    """The posterior colormap with NaN drawn in the forbidden colour. Forbidden cells are
    masked rather than painted over, so no antialiased edge of the ramp's zero survives
    around them."""
    cmap = plt.get_cmap(HEATMAP_CMAP).copy()
    cmap.set_bad(FORBID_FACE)
    return cmap


def _mark_forbidden(ax, mask):
    """Cross every forbidden cell of `ax`'s heatmap (the white ground comes from
    _forbidden_cmap)."""
    half = FORBID_SPAN / 2
    for i, j in zip(*np.nonzero(mask)):
        for dy in (-half, half):
            ax.plot([j - half, j + half], [i + dy, i - dy], color=FORBID_CROSS,
                    linewidth=FORBID_LW, solid_capstyle="round", zorder=3, clip_on=False)


def _render_enh_target(fig, mat_sorted, gwas_peaks_sorted, ann, active_tfs, categories,
                       forbidden=None):
    """Draw panel C into `fig` (a SubFigure), with the target names coloured by type so
    that enhancer -> TF edges are distinguishable from enhancer -> gene edges."""
    targets = list(mat_sorted.columns)
    n_peaks, n_targets, n_cats = len(gwas_peaks_sorted), len(targets), len(categories)
    active = set(active_tfs)
    tgt_colors = [TF_COLOR if t in active else GENE_COLOR for t in targets]

    # Three rows: the diagonal disease names, the strips and heatmap, the target-type key.
    # The names and the key are drawn inside blank axes of their own rows, so the layout
    # engine reserves their height without trying to fit them as tick labels.
    gs = fig.add_gridspec(3, 2,
                          height_ratios=[_strip_label_height(categories),
                                         max(4.0, n_peaks * 0.5), LEGEND_ROW_H],
                          width_ratios=[0.25 * n_cats, 4.0])
    ax_lab = fig.add_subplot(gs[0, 0])
    ax_lab.axis("off")

    # All the disease strips in one axes, as a single RGB image.
    ax_strip = fig.add_subplot(gs[1, 0])
    rgb = np.empty((n_peaks, n_cats, 3))
    for k, cat in enumerate(categories):
        on, off = to_rgb(CATEGORY_COLORS[cat]), to_rgb("#ebebeb")
        rgb[:, k] = [on if v else off for v in ann[k]]
    ax_strip.imshow(rgb, aspect="auto", interpolation="nearest")
    # Hairline gaps between categories.
    ax_strip.set_xticks(np.arange(n_cats) - 0.5, minor=True)
    ax_strip.grid(which="minor", color="white", linewidth=3)
    ax_strip.set_xticks([])
    shown = network._enhancer_short_labels(gwas_peaks_sorted)
    ax_strip.set_yticks(np.arange(n_peaks))
    ax_strip.set_yticklabels([shown[p] for p in gwas_peaks_sorted], fontsize=ENH_LABEL_FS)
    ax_strip.tick_params(left=False, length=0)
    for sp in ax_strip.spines.values():
        sp.set_visible(False)
    for k, cat in enumerate(categories):
        lab = ax_lab.text((k + 0.5) / n_cats, 0.0, network._trait_display(cat), rotation=45,
                          rotation_mode="anchor", ha="left", va="center",
                          fontsize=STRIP_LABEL_FS, transform=ax_lab.transAxes)
        lab.set_in_layout(False)

    ax_hm = fig.add_subplot(gs[1, 1], sharey=ax_strip)
    forbid = _forbidden_mask(mat_sorted, forbidden)
    im = ax_hm.imshow(np.where(forbid, np.nan, mat_sorted.values), aspect="auto",
                      cmap=_forbidden_cmap(), vmin=0, vmax=1, interpolation="nearest")
    _mark_forbidden(ax_hm, forbid)
    ax_hm.set_xticks(np.arange(n_targets))
    ax_hm.set_xticklabels(targets, rotation=45, rotation_mode="anchor", ha="right",
                          va="top", fontsize=TF_LABEL_FS)
    for tick, col in zip(ax_hm.get_xticklabels(), tgt_colors):
        tick.set_color(col)
    ax_hm.tick_params(labelleft=False)
    ax_hm.set_xlabel("Target", fontsize=AXIS_LABEL_FS)
    cb = fig.colorbar(im, ax=ax_hm, shrink=0.95, aspect=18, pad=0.02)
    cb.set_label(CBAR_LABEL, fontsize=CBAR_LABEL_FS)
    cb.ax.tick_params(labelsize=CBAR_LABEL_FS)
    # Target-type key, in the blank row reserved for it under the heatmap column.
    ax_leg = fig.add_subplot(gs[2, 1])
    ax_leg.axis("off")
    ax_leg.legend(handles=[Patch(facecolor=TF_COLOR, label="TF"),
                           Patch(facecolor=GENE_COLOR, label="Non-TF gene")],
                  title="Target type", fontsize=LEGEND_FS, title_fontsize=LEGEND_FS + 1,
                  ncol=2, loc="center", frameon=False)
    return ax_strip, ax_hm


# ---------------------------------------------------------------------------
# Panel A: the whole inferred network
# ---------------------------------------------------------------------------

PANEL_LETTER_FS  = 34

# Floor on the width of the left column, in inches.
COMBINED_LEFT_MIN_W = 14.0
# Panel B's TF filter: a TF is kept if any of its edges into an immGWAS enhancer reaches this.
COMBINED_TF_CUT     = 0.5
# Panel D: the target gene whose unit is drawn, the scale of its marks and type relative to
# unit.py's standalone figure, the inches per TF / enhancer slot, and the panel height.
COMBINED_UNIT_GENE  = "IL1A"
COMBINED_UNIT_SCALE = 1.1
COMBINED_UNIT_SLOT_W = 1.6
COMBINED_UNIT_H     = 11.0
COMBINED_UNIT_LEGEND_NCOLS = unit.LEGEND_NCOLS
# Spring-layout seed for panel A, picked by eye from the five (of seeds 0-59) that spread
# the nodes most evenly over the layout's bounding box.
COMBINED_NETWORK_SEED = 8
# Spring-constant multiplier for panel A: three times network.SPRING_K spreads the nodes
# over more of the panel.
COMBINED_NETWORK_K_MULT = 3.0
# Panel A's outermost tenth of the largest component's nodes is pulled in to the 90th
# percentile of distance from its centre (network._cap_radius).
COMBINED_NETWORK_CAP = 0.90
# Panel A's type and mark sizes at this canvas. `node` sets only the three hub names.
COMBINED_NETWORK_STYLE = dict(node=30, legend=LEGEND_FS, legend_title=LEGEND_FS + 1,
                              cbar=CBAR_LABEL_FS, cbar_tick=CBAR_LABEL_FS - 4,
                              node_size=42, arrow=11, edge_mult=0.85)


def _posterior_arrays(in_dir):
    """eval.py's posterior artifacts, in the argument order network._build_network_graph
    takes them: (post, W_median, w_cond_median, w_lower, w_upper, col_names)."""
    col_names = pd.read_csv(f"{in_dir}/nodes.csv")["node"].tolist()
    post      = np.load(f"{in_dir}/posterior_dir.npy")
    W_median  = np.load(f"{in_dir}/W_median.npy")

    n   = len(col_names)
    idx = {name: i for i, name in enumerate(col_names)}
    w_cond_median = np.full((n, n), np.nan)
    w_lower       = np.full((n, n), np.nan)
    w_upper       = np.full((n, n), np.nan)
    edges = pd.read_csv(f"{in_dir}/edge_recovery.csv")
    for parent, child, med, lo, hi in zip(edges["parent"], edges["child"],
                                          edges["w_post_median"],
                                          edges["w_post_lower95"], edges["w_post_upper95"]):
        w_cond_median[idx[parent], idx[child]] = med
        w_lower[idx[parent], idx[child]] = lo
        w_upper[idx[parent], idx[child]] = hi
    return post, W_median, w_cond_median, w_lower, w_upper, col_names


def _full_network_bundle(in_dir):
    """The whole inferred network: every node, and every edge at PIP >= config.PIP_CUT.
    Returns (G, tfs, enhancers, downstream_genes)."""
    arrays    = _posterior_arrays(in_dir)
    col_names = arrays[-1]
    tfs, enhancers, downstream_genes = network._classify_nodes(col_names)
    G = network._build_network_graph(*arrays, threshold=config.PIP_CUT)
    return G, tfs, enhancers, downstream_genes


# ---------------------------------------------------------------------------
# The four-panel figure
# ---------------------------------------------------------------------------

def plot_network_gwas_panel(out_path, mat, mat2, ctx, active_tfs, full_bundle, unit_bundle):
    """A over C in the left column, B over D in the right; the two columns split their
    height independently. The heatmaps keep their natural sizes and the network takes
    whatever the left column has left over. `unit_bundle` is (gene, enh_edges, tf_edges,
    threshold)."""
    tfs_b = [t for t in active_tfs if (mat.loc[t] >= COMBINED_TF_CUT).any()]
    mat_b = mat.loc[tfs_b]
    w_b, h_b = _tf_gwas_figsize(len(tfs_b), len(ctx["gwas_peaks_sorted"]),
                                len(ctx["categories"]))
    w_c, h_c = _enh_target_figsize(len(ctx["peaks_out"]), len(mat2.columns),
                                   ctx["categories_out"])
    w_left = max(w_c, COMBINED_LEFT_MIN_W)
    _, enh_d, tf_d, _ = unit_bundle
    slots = max(tf_d["parent"].nunique(), enh_d["parent"].nunique(), 3)
    w_right = max(w_b, COMBINED_UNIT_SLOT_W * slots)
    h_d = COMBINED_UNIT_H
    h_total = h_b + h_d
    h_a = h_total - h_c

    fig = plt.figure(figsize=(w_left + w_right, h_total), layout="constrained")
    fig.get_layout_engine().set(wspace=LAYOUT_WSPACE)
    sf_left, sf_right = fig.subfigures(1, 2, width_ratios=[w_left, w_right])
    sf_a, sf_c = sf_left.subfigures(2, 1, height_ratios=[h_a, h_c])
    sf_b, sf_d = sf_right.subfigures(2, 1, height_ratios=[h_b, h_d])

    # Panel A: every name off except the three hubs, a flat edge width, the enhancers as one
    # node type, and the node legend tucked under the inset colorbar.
    ax_a = sf_a.add_subplot()
    G, tfs, enhancers, downstream_genes = full_bundle
    legs = network._render_network(sf_a, ax_a, G, tfs, enhancers, downstream_genes,
                                   COMBINED_NETWORK_STYLE, seed=COMBINED_NETWORK_SEED,
                                   label_hubs=3, k_mult=COMBINED_NETWORK_K_MULT,
                                   cap_quantile=COMBINED_NETWORK_CAP)

    # B's colorbar sized as C's, now that B is only a dozen rows tall.
    _render_tf_gwas(sf_b, mat_b, tfs_b, ctx["gwas_peaks_sorted"], ctx["ann"],
                    ctx["categories"], cbar_kw=dict(shrink=0.95, aspect=18))
    # Panel C colours its target names by type over every active TF, not just B's subset.
    _render_enh_target(sf_c, mat2, ctx["peaks_out"], ctx["ann_out"], active_tfs,
                       ctx["categories_out"], forbidden=ctx.get("forbidden"))

    gene, enh_d, tf_d, lo = unit_bundle
    ax_d = sf_d.add_subplot()
    info = unit._render_unit(ax_d, gene, enh_d, tf_d, lo, COMBINED_UNIT_SCALE)
    legs = [*legs, *unit._legends(sf_d, info["traits"], info["plain_enh"],
                                  (info["codes"] > 0).any(), (info["codes"] == 0).any(),
                                  info["unsigned"], lo, fs=LEGEND_FS - 2,
                                  ncols=COMBINED_UNIT_LEGEND_NCOLS)]
    panels = [(sf_a, "A"), (sf_b, "B"), (sf_c, "C"), (sf_d, "D")]

    # Panel letters at the top left of each subfigure.
    letters = [sf.suptitle(letter, x=0.0, ha="left", fontsize=PANEL_LETTER_FS,
                           fontweight="bold")
               for sf, letter in panels]

    # An axis label that overruns its own axes is not covered by that axes' tight bbox, so
    # every axis label is handed to the saved crop directly. bbox_extra_artists *replaces*
    # matplotlib's default extra-artist set, so the legends and panel letters are listed too.
    axis_labels = [ax.xaxis.label for ax in fig.axes if ax.get_xlabel()]
    save_kw = dict(bbox_inches="tight", pad_inches=0.1,
                   bbox_extra_artists=[*legs, *letters, *axis_labels])
    fig.savefig(out_path, dpi=150, **save_kw)
    fig.savefig(os.path.splitext(out_path)[0] + ".pdf", **save_kw)
    plt.close(fig)


def main():
    ctx = dataset_context()
    active_tfs        = ctx["active_tfs"]
    downstream_genes  = ctx["downstream_genes"]
    gwas_peaks_sorted = ctx["gwas_peaks_sorted"]
    print(f"active TFs: {len(active_tfs)}, downstream genes: {len(downstream_genes)}, "
          f"immGWAS enhancers in node set: {len(gwas_peaks_sorted)}")
    edges = pd.read_csv(os.path.join(config.EVAL_DIR, "edge_recovery.csv"))
    os.makedirs(GWAS_OUT_DIR, exist_ok=True)

    # Panel B's matrix, TF rows alphabetically: a fixed index the reader can look a TF up in.
    active_tfs = sorted(active_tfs)
    mat = _matrix_from_edges(edges, active_tfs, gwas_peaks_sorted).loc[active_tfs]
    mat.to_csv(f"{GWAS_OUT_DIR}/tf_gwas_matrix.csv")

    # Panel C's matrix over the enhancers with >= 1 permitted outgoing pair and the targets
    # reachable from them: TFs first, then genes, each along the genome by TSS.
    peaks_out = ctx["peaks_out"]
    reachable = ctx["targets_out"]
    tfs_out   = [t for t in active_tfs if t in reachable]
    genes_out = [g for g in downstream_genes if g in reachable]
    loci = load_gene_loci()
    targets = _genomic_order(tfs_out, loci) + _genomic_order(genes_out, loci)
    mat2 = _matrix_from_edges(edges, peaks_out, tfs_out + genes_out)[targets]
    mat2.to_csv(f"{GWAS_OUT_DIR}/gwas_enh_to_target_matrix.csv")

    full_bundle = _full_network_bundle(config.EVAL_DIR)
    print(f"panel A (network): {full_bundle[0].number_of_edges()} edges over "
          f"{full_bundle[0].number_of_nodes()} nodes, at PIP >= {config.PIP_CUT}")
    enh_edges, tf_edges = unit.build_unit(edges, load_prior_matrix(), COMBINED_UNIT_GENE,
                                          unit.THRESHOLD)
    unit_bundle = (COMBINED_UNIT_GENE, enh_edges, tf_edges, unit.THRESHOLD)

    os.makedirs(config.FIG_DIR, exist_ok=True)
    out_path = os.path.join(config.FIG_DIR, "panel_network_gwas.png")
    plot_network_gwas_panel(out_path, mat, mat2, ctx, active_tfs, full_bundle, unit_bundle)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
