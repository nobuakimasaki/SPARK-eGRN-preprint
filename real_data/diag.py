"""
diag.py
=======
MCMC diagnostics of the real-data run (supplementary figures). Reads the sampler output
(config.CHAIN_PATH) and writes to config.FIG_DIR:

  real_diag_allchains_trace.png   : log-posterior traces of every chain (full trace with
                                    the end of burn-in marked, and after burn-in)
  real_diag_edge_counts_trace.png : edge-count traces of every chain (same layout)
  real_diag_acceptance_rates.png  : per-chain acceptance rates of the graph and
                                    regression parameter updates
  real_diag_swap_rates.png        : swap acceptance rate per adjacent chain pair

The acceptance and swap rates are the sampler's running counters, so they cover the whole
run, burn-in included. The per-draw sample lists are not needed and are streamed past
(eval.stream_load), which keeps peak memory at a few GB.

Usage
-----
  python diag.py
"""

import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import config
from eval import stream_load

# Plotting stride for both panels of the every-chain plots.
FULL_STRIDE = 100

# The six per-draw sample lists in a result pickle. Nothing here reads them.
_SAMPLE_LISTS = ("graphs_samples", "b_samples", "W_samples",
                 "log_psi_samples", "alpha_samples", "rho_logits_samples")

# Colour of every single-series rate plot, and of the burn-in marker.
TRACE_COLOR  = "#4C78A8"
BURNIN_COLOR = "#C0392B"


def _style(ax):
    ax.grid(axis="y", color="0.9", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _temperature_colors(betas):
    """One colour per chain on a cold (blue) to hot (red) ramp over the chain
    temperatures T_r = 1 / beta_r, with the ScalarMappable for its colour bar."""
    temps = 1.0 / np.asarray(betas)
    cmap, norm = plt.cm.coolwarm, Normalize(vmin=temps.min(), vmax=temps.max())
    return [cmap(norm(t)) for t in temps], ScalarMappable(norm=norm, cmap=cmap)


def _trace_pair(series, burn_in, ylabel, out_path, colors, mappable):
    """Full trace with the end of burn-in marked (left) beside the trace after burn-in
    (right). `series` is (n_chains, n_iter), the cold chain drawn on top. Both panels are
    drawn every FULL_STRIDE-th iteration, which keeps 20 chains of millions of iterations
    drawable without changing what the panels show."""
    series = np.atleast_2d(series)
    n_chains, n_it = series.shape
    full = np.arange(n_it)[::FULL_STRIDE]
    post = full[full >= burn_in]

    fig, (ax_full, ax_post) = plt.subplots(1, 2, figsize=(12, 3.2), layout="constrained")
    for c in reversed(range(n_chains)):
        ax_full.plot(full, series[c, full], lw=0.4, color=colors[c], alpha=0.85)
        ax_post.plot(post, series[c, post], lw=0.4, color=colors[c], alpha=0.85)
    ax_full.axvline(burn_in, color=BURNIN_COLOR, ls="--", lw=1.2, label="End of burn-in")
    # Above the panel's top-right corner, beside its short title, rather than inside
    # the axes, where 20 chains leave no clear spot and the legend covered the hottest.
    ax_full.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=9,
                   frameon=False, borderaxespad=0.1)
    ax_full.set_title("Full trace", fontsize=11)
    ax_post.set_title("After burn-in", fontsize=11)
    for ax in (ax_full, ax_post):
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
        _style(ax)
    cb = fig.colorbar(mappable, ax=[ax_full, ax_post], pad=0.01, aspect=25)
    cb.set_label("Chain temperature $T_r$")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def make_diagnostics(path, out_dir, burn_in):
    print(f"loading {os.path.basename(path)}")
    res = stream_load(path, {k: (lambda _x: None) for k in _SAMPLE_LISTS})
    n_it = len(res["logpost_trace"])
    print(f"{n_it:,} iterations, burn-in = {burn_in:,}")

    diagnostics = res["diagnostics"]
    betas = diagnostics["temperatures"]["betas"]
    n_chains = len(betas)
    colors, mappable = _temperature_colors(betas)

    # The log posterior and edge count of every chain.
    _trace_pair(np.asarray(res["logpost_trace_all"]), burn_in, "Log posterior",
                f"{out_dir}/real_diag_allchains_trace.png", colors, mappable)
    _trace_pair(np.asarray(res["edge_counts_all"]), burn_in, "Number of edges",
                f"{out_dir}/real_diag_edge_counts_trace.png", colors, mappable)

    # Per-chain acceptance rate of the graph and regression parameter updates.
    chain_ids = np.arange(n_chains)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), layout="constrained", sharex=True)
    for ax, key, title in zip(axes, ("graph", "theta"),
                              ("Graph update", "Regression parameter update")):
        ax.plot(chain_ids, diagnostics[key]["rates"], marker="o", ms=4, lw=1, color=TRACE_COLOR)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Acceptance rate")
        ax.set_xlabel(f"Chain $r$ (0 = cold, {n_chains - 1} = hottest)")
        ax.set_xticks(chain_ids[:: 2 if n_chains > 10 else 1])
        _style(ax)
    savepath = f"{out_dir}/real_diag_acceptance_rates.png"
    fig.savefig(savepath, dpi=200)
    plt.close(fig)
    print(f"Saved {savepath}")

    # Swap acceptance rate per adjacent chain pair.
    pair_ids = np.arange(n_chains - 1)
    fig, ax = plt.subplots(figsize=(9, 3.4), layout="constrained")
    ax.plot(pair_ids, diagnostics["swap"]["adjacent_pairs"]["rates"],
            marker="o", ms=4, lw=1, color=TRACE_COLOR)
    ax.set_xticks(pair_ids)
    ax.set_xticklabels([f"{i}–{i + 1}" for i in pair_ids], fontsize=8)
    ax.set_xlabel("Adjacent chain pair $(r, r+1)$")
    ax.set_ylabel("Swap acceptance rate")
    _style(ax)
    savepath = f"{out_dir}/real_diag_swap_rates.png"
    fig.savefig(savepath, dpi=200)
    plt.close(fig)
    print(f"Saved {savepath}")


def main():
    parser = argparse.ArgumentParser(description="MCMC diagnostic plots of the real-data run.")
    parser.add_argument("--chain", default=config.CHAIN_PATH,
                        help="MCMC output (.pkl or .pkl.zst).")
    parser.add_argument("--burn-in", type=int, default=config.BURN_IN,
                        help=f"Iterations marked as burn-in (default {config.BURN_IN:,}).")
    parser.add_argument("--out", default=config.FIG_DIR,
                        help=f"Output directory (default {config.FIG_DIR}).")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    make_diagnostics(args.chain, args.out, args.burn_in)


if __name__ == "__main__":
    main()
