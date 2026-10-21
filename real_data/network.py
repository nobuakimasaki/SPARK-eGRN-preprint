"""
network.py
==========
Shared styling and the whole-network drawing used by the real-data figures.

  - node / edge colours and shapes, shared by gwas_panel.py and unit.py so a node type, an
    effect sign and a disease read the same in every panel;
  - the immune disease annotation of the immGWAS enhancers (PEAK_TRAITS, TRAIT_COLORS,
    trait_abbrev_labels) and the short enhancer labels ('chr2:112.83 Mb');
  - the inferred network (Figure 3A): _build_network_graph builds the graph of edges at
    PIP >= a cut, and _render_network draws it on a force-directed layout with the largest
    connected component on top and every other node in a row beneath it.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

import config

# ---------------------------------------------------------------------------
# Network layout and drawing constants
# ---------------------------------------------------------------------------

# Force-directed layout. k sits above networkx's 1/sqrt(n) default to push the components
# apart; callers scale it with `k_mult`.
SPRING_K          = 2.2
SPRING_ITERATIONS = 400

NETWORK_MARGIN = 0.01
# Where the node legend goes: directly beneath the colorbar, in the colorbar's OWN axes
# coordinates, so it tracks the bar wherever the layout puts it. Centred on the bar and low
# enough to clear the extend arrow and the bar's bottom end label.
LEGEND_UNDER_CBAR = (0.5, -0.30)
# The colorbar as an inset of the network axes, in axes coordinates:
# [left, bottom, width, height]. An inset is positioned against the drawing rather than at
# the edge of the subplot's grid cell.
CBAR_INSET_RECT = (1.02, 0.30, 0.022, 0.40)

# Posterior edge probabilities: a white -> dark ramp, so that a high probability is the
# darkest thing on the page. The same ramp as Figure 1 panel G.
POST_CMAP = "Blues"

# Node colours by type, as in Figure 1 panel G and the paper preamble (nodeTF / nodeGene /
# nodeEnh). Enhancers are drawn as diamonds, TFs and genes as circles, so node type
# survives where an enhancer's disease colour lands near the TF red. A diamond of a given
# area reads smaller than a circle of the same area, hence the bump.
NODE_TF_COLOR   = "#DC143C"
NODE_GENE_COLOR = "#2CA02C"
NODE_ENH_COLOR  = "#3B7DD8"
NODE_ENH_SHAPE     = "D"
NODE_ENH_SIZE_MULT = 1.15

# Edge colours by the SIGN of the regulatory effect: red up-regulation, blue down.
EDGE_POS_COLOR = "#b2182b"
EDGE_NEG_COLOR = "#2166ac"
# Width of every edge in the network drawing.
EDGE_FLAT_WIDTH = 2.35
# Backing box for the hub names, which land in the densest part of the drawing.
# Translucent so the edges under it stay visible, and borderless so the box reads as a
# halo on the name rather than as a mark of its own.
HUB_LABEL_BBOX = dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.75)

# Edge colouring: the conditional-on-present median W on a symmetric RdBu_r scale,
# saturating at this quantile of |median W| over the drawn edges. The medians are
# concentrated near 0 with a long tail, so scaling to the max would push the typical edge
# to the washed-out middle of the ramp.
EDGE_CMAP = "RdBu_r"
EDGE_VABS_QUANTILE = 0.95

CODE_LOOKUP = config.CODE_LOOKUP


def load_prior_matrix():
    """The integer-coded topological prior, indexed by node name."""
    return pd.read_csv(config.PRIOR_PATH, index_col=0)


def _chr_key(c):
    c = str(c).replace("chr", "")
    return {"X": 23, "Y": 24, "M": 25, "MT": 25}.get(c, int(c) if c.isdigit() else 99)


def _is_enhancer(name):
    return name.startswith("chr") and name.count("-") >= 2


# ---------------------------------------------------------------------------
# Immune disease annotation of the immGWAS enhancers
# ---------------------------------------------------------------------------

# Qualitative palette for immune disease categories. Colours are assigned by alphabetical
# rank over all immune disease categories in the GWAS table, so co-occurring categories
# never share a colour. Deliberately avoids the node-type colours.
_TRAIT_PALETTE = ["#E45756", "#F58518", "#EECA3B", "#B279A2", "#72B7B2", "#9D755D",
                  "#D37295", "#17BECF", "#FF9DA6", "#843C39", "#5254A3", "#E377C2",
                  "#FDBF6F", "#A55194", "#CE6DBD", "#B15928", "#7B4173", "#F7B6D2",
                  "#8C6D31", "#AD494A", "#DE9ED6", "#6B6ECF"]


def load_peak_traits():
    """Map each immGWAS enhancer peak (rp_id) to the set of immune disease trait labels of
    the study its variant comes from: rp_id -> study (gwas_variants_immune.csv) ->
    trait_id/trait_label (study_trait_map.csv), keeping only trait_ids flagged as immune
    diseases (efo_trait_sets.csv)."""
    gv  = pd.read_csv(config.GWAS_PATH)                             # rp_id, study, ...
    stm = pd.read_csv(config.STUDY_TRAIT_PATH).dropna(subset=["trait_label"])
    efo = pd.read_csv(config.EFO_TRAIT_PATH)
    imm = efo["is_immune_disease"]
    imm = imm if imm.dtype == bool else imm.astype(str).str.upper().eq("TRUE")
    immune_ids = set(efo.loc[imm, "trait_id"])
    stm = stm[stm["trait_id"].isin(immune_ids)]
    study_labels = stm.groupby("study")["trait_label"].agg(lambda s: sorted(set(s)))
    peak_traits = {}
    for peak, study in zip(gv["rp_id"], gv["study"]):
        peak_traits.setdefault(peak, set()).update(study_labels.get(study, []))
    return peak_traits


# rp_id -> set(immune disease trait label); built once at import.
PEAK_TRAITS = load_peak_traits()
# One colour per disease category (alphabetical), plus the two pseudo-categories.
TRAIT_COLORS = {cat: _TRAIT_PALETTE[i % len(_TRAIT_PALETTE)]
                for i, cat in enumerate(sorted({c for cs in PEAK_TRAITS.values() for c in cs}))}
# Hand-pinned where the positional palette lands two traits that co-occur in a figure on
# near-identical hues: asthma's #9D755D and lupus's #8C6D31 are both mid-browns. Keyed by
# name, so a change in the trait set cannot silently undo the separation.
TRAIT_COLORS["systemic lupus erythematosus"] = "#0F5257"
TRAIT_COLORS["Multiple"]    = "#000000"
TRAIT_COLORS["No GWAS hit"] = "#D3D3D3"

# Short codes for naming an enhancer's disease at the node. Clinical abbreviations where
# one is standard; anything unlisted falls back to its initials.
TRAIT_ABBREV = {
    "Crohn disease": "CD", "ulcerative colitis": "UC",
    "inflammatory bowel disease": "IBD", "celiac disease": "CeD",
    "allergic rhinitis": "AR", "asthma": "AST", "childhood onset asthma": "COA",
    "allergic disease": "AD", "atopic eczema": "AE", "food allergy": "FA",
    "bullous pemphigoid": "BP",
    "systemic lupus erythematosus": "SLE", "Sjogren syndrome": "SjS",
    "mixed connective tissue disease": "MCTD",
    "multiple sclerosis": "MS", "myasthenia gravis": "MG",
    "type 1 diabetes mellitus": "T1D", "type 1 diabetes nephropathy": "T1DN",
    "psoriasis": "PsO", "psoriasis vulgaris": "PsV",
    "ankylosing spondylitis": "AS", "rheumatoid arthritis": "RA",
}


def _trait_abbrev(cat):
    return TRAIT_ABBREV.get(cat) or "".join(w[0] for w in cat.split()).upper()


def trait_abbrev_labels(enhancers):
    """{enhancer peak -> short disease code}. A peak carrying several diseases gets them
    slash-joined ('AR/AST')."""
    out = {}
    for e in enhancers:
        cats = sorted(PEAK_TRAITS.get(e, set()), key=str.lower)
        out[e] = "/".join(_trait_abbrev(c) for c in cats) if cats else "no GWAS hit"
    return out


def _trait_display(cat):
    """Trait label as it appears in a legend or strip: sentence case, so 'asthma' reads
    'Asthma' while names that already lead with a proper noun are left as they are."""
    return cat[:1].upper() + cat[1:]


def _enhancer_short_labels(names):
    """Display labels for enhancer peaks: 'chr2-112831548-112837357' -> 'chr2:112.83 Mb',
    keyed on the peak's start coordinate. Rounding can make two nearby peaks share a label,
    so the precision escalates until the labels are unique over `names`, falling back to the
    raw name if even 4 decimals collide. Non-enhancer names pass through unchanged."""
    enh = [n for n in names if _is_enhancer(n)]
    for dp in (2, 3, 4):
        short = {}
        for n in enh:
            chrom, start, _ = n.split("-", 2)
            short[n] = f"{chrom}:{int(start) / 1e6:.{dp}f} Mb"
        if len(set(short.values())) == len(enh):
            return {n: short.get(n, n) for n in names}
    return {n: n for n in names}


def _enhancer_labels(enhancers):
    """Map each enhancer peak to a single disease label for node colouring: 'No GWAS hit'
    if its variant study has no immune disease trait, the trait label if exactly one, else
    'Multiple'."""
    labels = {}
    for e in enhancers:
        cats = sorted(PEAK_TRAITS.get(e, set()))
        labels[e] = ("No GWAS hit" if not cats else
                     cats[0] if len(cats) == 1 else "Multiple")
    return labels


# ---------------------------------------------------------------------------
# The inferred network
# ---------------------------------------------------------------------------

def _classify_nodes(col_names):
    """Split node names into (TFs, enhancers, downstream genes). Enhancers are the `chr...`
    peaks; a node with no permitted outgoing prior edge is a downstream gene; everything
    else is a TF."""
    Q = load_prior_matrix().loc[col_names, col_names].to_numpy().astype(np.int64)
    no_outgoing = np.all(Q == -1, axis=1)
    tfs, enhancers, downstream_genes = [], [], []
    for i, name in enumerate(col_names):
        if name.startswith("chr"):       enhancers.append(name)
        elif no_outgoing[i]:             downstream_genes.append(name)
        else:                            tfs.append(name)
    return tfs, enhancers, downstream_genes


def _edge_vabs(vals):
    """Symmetric colour limit for the edge colouring: the EDGE_VABS_QUANTILE quantile of
    |vals|. Falls back to the max, then to a small positive number, so a degenerate
    all-zero graph still gets a valid norm."""
    a = np.abs(np.asarray(vals, dtype=float))
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 1e-6
    v = float(np.quantile(a, EDGE_VABS_QUANTILE))
    if not (v > 0):
        v = float(a.max())
    return max(v, 1e-6)


def _build_network_graph(post, W_median, w_cond_median, w_lower, w_upper, col_names,
                         threshold):
    """DiGraph over all nodes with an edge i->j iff P(edge) >= `threshold`.

    Edge attrs: weight = P(edge); w_median = marginal median W; w_cond = median W
    conditional on the edge being present; significant = the 95% CrI excludes 0.

    W_median.npy is *marginal*: each draw contributes 0 when the edge is absent, so any
    edge present in <50% of draws has a marginal median of exactly 0. w_cond
    (edge_recovery.csv's w_post_median) conditions on presence, and is what edges are
    coloured by; the marginal is only a fallback where w_cond is NaN."""
    n = len(col_names)
    G = nx.DiGraph()
    G.add_nodes_from(col_names)
    for i in range(n):
        for j in range(n):
            if i == j or post[i, j] < threshold:
                continue
            # CrI excludes 0 iff it lies entirely above or entirely below 0. Edges with too
            # few present draws have NaN bounds; the comparisons are then False.
            ci_excludes_0 = bool((w_lower[i, j] > 0) or (w_upper[i, j] < 0))
            w_c = w_cond_median[i, j]
            if not np.isfinite(w_c):
                w_c = W_median[i, j]
            G.add_edge(col_names[i], col_names[j],
                       weight=float(post[i, j]), w_median=float(W_median[i, j]),
                       w_cond=float(w_c), significant=ci_excludes_0)
    return G


# The row of stranded nodes spans this share of the main component's width, from its left
# edge. It stops short of the right edge so the node legend, which sits in the colorbar's
# column at the bottom right, never lands on the row's last nodes.
STRIP_ROW_FRAC = 0.78


def _cap_radius(xy, q):
    """Pull the outermost points of `xy` in to the q-th quantile of their distance from the
    median point, each along its own direction, then stretch the result back over the
    original bounding box. Distances in a spring layout carry no meaning, and a few long
    arms otherwise set the frame the rest of the drawing is scaled into."""
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    u = 2 * (xy - lo) / (hi - lo) - 1
    c = np.median(u, axis=0)
    d = u - c
    r = np.linalg.norm(d, axis=1)
    rq = np.quantile(r, q)
    u = c + d * np.where(r > rq, rq / np.maximum(r, 1e-12), 1.0)[:, None]
    u = 2 * (u - u.min(axis=0)) / (u.max(axis=0) - u.min(axis=0)) - 1
    return lo + (u + 1) / 2 * (hi - lo)


def _strip_spring_layout(Gu, k, iterations, seed, order_key, row_frac=STRIP_ROW_FRAC,
                         cap_quantile=None):
    """spring_layout on the largest connected component alone, with every other component
    -- the isolated nodes and any small stranded pieces -- set in one row beneath it.

    In the row, the small multi-node components come first, each on a slight diagonal so its
    edges stay visible, then the isolates sorted by `order_key`. `cap_quantile`, if given,
    pulls the largest component's outermost nodes in (see _cap_radius). Coordinates come
    back in roughly [-1, 1]."""
    comps = sorted(nx.connected_components(Gu), key=len, reverse=True)
    # G's own node order, not the set's: set iteration order varies between processes
    # under hash randomisation and would reach spring_layout's initial positions.
    main = [v for v in Gu.nodes() if v in comps[0]]
    sub = nx.Graph()
    sub.add_nodes_from(main)
    # With their attributes: spring_layout weights each spring by the edge's `weight`.
    sub.add_edges_from((u, v, d) for u, v, d in Gu.edges(data=True)
                       if u in comps[0] and v in comps[0])
    pos = nx.spring_layout(sub, k=k, iterations=iterations, seed=seed)

    small = [[v for v in Gu.nodes() if v in c] for c in comps[1:]]
    pieces = [c for c in small if len(c) > 1]
    single = sorted((c[0] for c in small if len(c) == 1), key=order_key)
    xy = np.array(list(pos.values()))
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = hi - lo
    slots = sum(len(c) for c in pieces) + len(single)
    xs = np.linspace(lo[0], lo[0] + row_frac * span[0], max(slots, 2))
    y = lo[1] - 0.09 * span[1]
    i = 0
    for comp in pieces:
        dys = np.linspace(0.025, -0.025, len(comp)) * span[1]
        for v, dy in zip(comp, dys):
            pos[v] = np.array([xs[i], y + dy])
            i += 1
    for v in single:
        pos[v] = np.array([xs[i], y])
        i += 1

    coords = np.array(list(pos.values()))
    centre = (coords.max(axis=0) + coords.min(axis=0)) / 2
    scale  = np.abs(coords - centre).max() or 1.0
    pos = {v: (np.asarray(xy) - centre) / scale for v, xy in pos.items()}
    if cap_quantile is not None:
        pos.update(zip(main, _cap_radius(np.array([pos[v] for v in main]), cap_quantile)))
    return pos


def _render_network(fig, ax, G, tfs, enhancers, downstream_genes, fonts, seed,
                    label_hubs=3, k_mult=1.0, cap_quantile=None):
    """Draw graph G into `ax` on a force-directed layout and return the legend artists.

    Every node is drawn: the largest connected component is laid out on its own and every
    other node set in a row beneath it (_strip_spring_layout). Node type is shown by colour
    and shape (TFs and genes circles, enhancers diamonds), and only the `label_hubs` nodes
    of highest out-degree are named. Every edge is drawn at the same width, coloured by the
    median W conditional on the edge being present on a symmetric diverging ramp with an
    inset colorbar; the node legend sits under that colorbar. The drawing is stretched to
    fill the axes (a spring layout has no metric meaning). `k_mult` scales the spring
    constant SPRING_K, and `cap_quantile` pulls in the largest component's outermost nodes."""
    nodes_in_G = set(G.nodes())
    tfs_g   = [c for c in tfs              if c in nodes_in_G]
    enh_g   = [c for c in enhancers        if c in nodes_in_G]
    genes_g = [c for c in downstream_genes if c in nodes_in_G]

    n_drawn = len(nodes_in_G)
    # Laid out on the undirected graph: edge direction should not bias the geometry. The
    # row of stranded nodes reads in blocks by node type: TFs, then genes, then enhancers.
    k = k_mult * SPRING_K / np.sqrt(max(n_drawn, 1))
    tf_set, enh_row = set(tfs_g), set(enh_g)
    pos = _strip_spring_layout(G.to_undirected(), k=k, iterations=SPRING_ITERATIONS,
                               seed=seed,
                               order_key=lambda v: (v not in tf_set, v in enh_row, v),
                               cap_quantile=cap_quantile)
    shown = _enhancer_short_labels(list(G.nodes()))

    enh_names = set(enhancers)
    node_colors = {}
    for c in G.nodes():
        if c in downstream_genes:     node_colors[c] = NODE_GENE_COLOR
        elif c in enh_names:          node_colors[c] = NODE_ENH_COLOR
        else:                         node_colors[c] = NODE_TF_COLOR

    enh_set = set(enh_g)
    for names, shape, size in ((
            [c for c in G.nodes() if c not in enh_set], "o", fonts["node_size"]),
            (list(enh_g), NODE_ENH_SHAPE, fonts["node_size"] * NODE_ENH_SIZE_MULT)):
        if names:
            nx.draw_networkx_nodes(G, pos, nodelist=names, node_shape=shape,
                                   node_color=[node_colors[c] for c in names],
                                   node_size=size, edgecolors="black", linewidths=0.5,
                                   ax=ax)

    ax.set_aspect("auto")
    ax.margins(NETWORK_MARGIN)

    # Hub names, just above their node. Ties broken by name so the choice is reproducible.
    LOFF = 0.030
    hubs = [nm for nm, _ in sorted(G.out_degree(),
                                   key=lambda kv: (-kv[1], kv[0]))[:label_hubs]]
    if hubs:
        nx.draw_networkx_labels(
            G, {nname: (pos[nname][0], pos[nname][1] + LOFF) for nname in hubs},
            labels={nname: shown[nname] for nname in hubs}, font_size=fonts["node"],
            verticalalignment="bottom", bbox=HUB_LABEL_BBOX, ax=ax)

    cbar_ax = None
    if G.number_of_edges() > 0:
        edgelist = list(G.edges())
        e_width  = np.full(len(edgelist), EDGE_FLAT_WIDTH) * fonts["edge_mult"]
        e_weff   = np.array([G[u][v]["w_cond"] for u, v in edgelist])
        w_vabs   = _edge_vabs(e_weff)
        cmap     = plt.get_cmap(EDGE_CMAP)
        norm     = Normalize(vmin=-w_vabs, vmax=w_vabs)
        e_color  = [cmap(norm(w)) for w in e_weff]
        nx.draw_networkx_edges(
            G, pos, edgelist=edgelist, width=list(e_width), edge_color=e_color,
            style="solid", arrows=True, arrowsize=fonts["arrow"],
            connectionstyle="arc3,rad=0.06", ax=ax)

        cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), extend="both",
                            cax=ax.inset_axes(CBAR_INSET_RECT))
        # What the two ends of the ramp mean, rather than the name of the quantity. Both
        # pads are measured from the axes, which stops at the bar itself, so they have to
        # clear the extend arrow drawn beyond it.
        cbar.ax.set_title("Up-regulation", fontsize=fonts["cbar"],
                          pad=0.8 * fonts["cbar"])
        cbar.ax.set_xlabel("Down-regulation", fontsize=fonts["cbar"],
                           labelpad=1.6 * fonts["cbar"])
        cbar.ax.tick_params(labelsize=fonts["cbar_tick"])
        cbar_ax = cbar.ax

    # Node legend. Marks rather than swatches, so the legend teaches the shape as well as
    # the colour; only the node types actually drawn are listed.
    def _mark(colour, shape, label):
        return Line2D([0], [0], linestyle="none", marker=shape, markerfacecolor=colour,
                      markeredgecolor="black", markeredgewidth=0.5, color="none",
                      markersize=np.sqrt(fonts["node_size"]), label=label)

    node_handles = [h for h, present in
                    ((_mark(NODE_TF_COLOR, "o", "TF"), tfs_g),
                     (_mark(NODE_GENE_COLOR, "o", "Non-TF gene"), genes_g))
                    if present]
    node_handles.append(_mark(NODE_ENH_COLOR, NODE_ENH_SHAPE, "Enhancer"))
    leg_kw = dict(fontsize=fonts["legend"], title_fontsize=fonts["legend_title"],
                  framealpha=0.95, labelspacing=0.3, handletextpad=0.4,
                  borderpad=0.35, markerscale=0.85)
    # Tucked under the colorbar: a roughly circular layout leaves dead space in that corner,
    # and using it keeps the legend from growing the figure downwards.
    node_leg = ax.legend(handles=node_handles, loc="upper center",
                         bbox_to_anchor=LEGEND_UNDER_CBAR,
                         bbox_transform=cbar_ax.transAxes,
                         title="Node type", **leg_kw)
    ax.axis("off")
    return (node_leg,)
