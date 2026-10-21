"""
tables.py
=========
The simulation-study tables and the numbers the Results quote, from eval.py's output.
Writes to config.TABLE_DIR:

  edge_detection.csv / .tex  : per method (SPARK under each prior setting, DAG0, the
                               regression) and DAG -- true edges, called edges
                               (PIP >= 0.5, or p < 0.001 for the regression), correct
                               calls, precision, recall, PR-AUC, with the mean over
                               DAGs (Table S2)
  class_edge_prob.csv / .tex : per prior setting and DAG -- posterior median and 95% CrI
                               of each class edge probability, with the RMSE of the
                               medians against the generating values (Table S3)
  results.txt                : the prior subset's pair counts, DAG 7's precision and
                               recall under the non-informative setting, and the mean
                               PR-AUC (+/- SE) of each setting and comparator

Usage
-----
  python tables.py
"""

import os

import numpy as np
import pandas as pd

import config as cfg
from eval import average_precision

TEX_LABELS_DETECTION = {
    # Split over the first rows of each block, so the long names do not widen the column.
    "expert_elicited": ["SPARK,", "expert-elicited"],
    "noninfo":         ["SPARK,", "non-informative"],
    "wrong_extreme":   ["SPARK, strongly", "informative,", "misspecified"],
    "no_partition":    ["SPARK,", "single class"],
    "dag0":            ["DAG0"],
    "regression":      ["Regularized regression", "(CellOracle)"],
}
# Every method of Table S2, in its order.
METHODS = cfg.SETTINGS + cfg.COMPARATORS
TEX_LABELS_RHO = {
    "expert_elicited": ["Expert-elicited"],
    "noninfo":         ["Non-informative"],
    "wrong_extreme":   ["Strongly", "informative,", "misspecified"],
    "no_partition":    ["Single class"],
}
# Class edge probability table column order: JASPAR2024, PCHi-C, baseline.
CLASSES = [1, 2, 0]


def _label(labels, s, r_i):
    return labels[s][r_i] if r_i < len(labels[s]) else ""


# ---------------------------------------------------------------------------
# Edge detection (Table S2)
# ---------------------------------------------------------------------------

def edge_detection(Q):
    """Per method and DAG: true edges, called pairs (PIP >= 0.5, or p < 0.001 for the
    regression, whose score is -log p), correct calls, precision, recall and PR-AUC."""
    cand = (Q >= 0) & ~np.eye(Q.shape[0], dtype=bool)
    rows = []
    for s in METHODS:
        for sim in range(cfg.N_DAGS):
            y = cfg.load_true_adjacency(sim)[cand].astype(int)
            p = cfg.load_prediction(s, sim)[cand]
            called = (p > -np.log(cfg.REG_CALL_P) if s == "regression"
                      else p >= cfg.CALL_THRESHOLD)
            n_true, n_called, n_correct = int(y.sum()), int(called.sum()), int(y[called].sum())
            rows.append({"setting": s, "sim": sim, "true_edges": n_true,
                         "called": n_called, "correct": n_correct,
                         "precision": n_correct / n_called if n_called else np.nan,
                         "recall": n_correct / n_true,
                         "pr_auc": average_precision(y, p)})
    return pd.DataFrame(rows)


def edge_detection_latex(df):
    f2 = lambda x: "n/a" if np.isnan(x) else f"{x:.2f}"
    f3 = lambda x: f"{x:.3f}"
    lines = [r"\begin{tabular}{llrrrccc}", r"\toprule",
             r"Method & DAG & True edges & Called & Correct & Precision & Recall & PR-AUC \\",
             r"\midrule"]
    for s_i, s in enumerate(METHODS):
        sub = df[df["setting"] == s].sort_values("sim")
        for r_i, r in enumerate(sub.itertuples()):
            lines.append(f"{_label(TEX_LABELS_DETECTION, s, r_i)} & {r.sim} & {r.true_edges} & "
                         f"{r.called} & {r.correct} & {f2(r.precision)} & {f2(r.recall)} & "
                         f"{f3(r.pr_auc)} \\\\")
        lines.append(r"\cmidrule{2-8}")
        lines.append(f" & Mean & {sub.true_edges.mean():.1f} & {sub.called.mean():.1f} & "
                     f"{sub.correct.mean():.1f} & {f2(sub.precision.mean())} & "
                     f"{f2(sub.recall.mean())} & {f3(sub.pr_auc.mean())} \\\\")
        lines.append(r"\bottomrule" if s_i == len(METHODS) - 1 else r"\midrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Class edge probabilities (Table S3)
# ---------------------------------------------------------------------------

def class_edge_prob(Q):
    """Posterior median and 95% CrI of every class edge probability. Under no_partition the
    single edge probability covers every permitted pair and sits in the baseline column."""
    cand = (Q >= 0) & ~np.eye(Q.shape[0], dtype=bool)
    rows = []
    for s in cfg.SETTINGS:
        for sim in range(cfg.N_DAGS):
            A = cfg.load_true_adjacency(sim)
            rho = cfg.load_rho(s, sim)
            for k in range(rho.shape[1]):
                in_k = cand if s == "no_partition" else (Q == k)
                r = rho[:, k]
                rows.append({"setting": s, "sim": sim, "class": k,
                             "n_pairs": int(in_k.sum()), "n_edges": int(A[in_k].sum()),
                             "median": float(np.median(r)),
                             "lo": float(np.quantile(r, 0.025)),
                             "hi": float(np.quantile(r, 0.975))})
    return pd.DataFrame(rows)


def class_edge_prob_latex(df):
    f2 = lambda x: f"{x:.2f}"
    cell = lambda r: f"{f2(r['median'])} [{f2(r['lo'])}, {f2(r['hi'])}]"
    lines = [
        r"\begin{tabular}{llccc}", r"\toprule",
        r" & & \multicolumn{3}{c}{Posterior median class edge probability"
        r" [$95\%$ CrI]} \\",
        r"\cmidrule(lr){3-5}",
        r"Setting & DAG & JASPAR2024 & PCHi-C & No evidence \\",
        r" & & (true value $" + f"{cfg.TRUE_RHO[1]}" + r"$)"
        r" & (true value $" + f"{cfg.TRUE_RHO[2]}" + r"$)"
        r" & (true value $" + f"{cfg.TRUE_RHO[0]}" + r"$) \\",
        r"\midrule",
    ]
    for s_i, s in enumerate(cfg.SETTINGS):
        sub = df[df["setting"] == s]
        for r_i, sim in enumerate(sorted(sub["sim"].unique())):
            by_k = {r["class"]: r for _, r in sub[sub["sim"] == sim].iterrows()}
            cells = [cell(by_k[k]) if k in by_k else "" for k in CLASSES]
            lines.append(f"{_label(TEX_LABELS_RHO, s, r_i)} & {sim} & " + " & ".join(cells) + r" \\")
        # RMSE of the posterior medians against the generating value, over the DAGs.
        rmse = {k: np.sqrt(((d["median"] - cfg.TRUE_RHO[k]) ** 2).mean())
                for k, d in sub.groupby("class")}
        lines.append(r"\cmidrule{2-5}")
        lines.append(" & RMSE & " + " & ".join(f2(rmse[k]) if k in rmse else ""
                                                for k in CLASSES) + r" \\")
        lines.append(r"\bottomrule" if s_i == len(cfg.SETTINGS) - 1 else r"\midrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def main():
    os.makedirs(cfg.TABLE_DIR, exist_ok=True)
    Q = cfg.load_prior_codes()

    det = edge_detection(Q)
    det.to_csv(os.path.join(cfg.TABLE_DIR, "edge_detection.csv"), index=False)
    with open(os.path.join(cfg.TABLE_DIR, "edge_detection.tex"), "w") as f:
        f.write(edge_detection_latex(det))

    rho = class_edge_prob(Q)
    rho.to_csv(os.path.join(cfg.TABLE_DIR, "class_edge_prob.csv"), index=False)
    with open(os.path.join(cfg.TABLE_DIR, "class_edge_prob.tex"), "w") as f:
        f.write(class_edge_prob_latex(rho))

    off = ~np.eye(Q.shape[0], dtype=bool)
    nodes = cfg.load_nodes()
    out = ["Topological prior (chr2 subset)",
           f"  nodes: {len(nodes)} ({', '.join(f'{k} {n}' for k, n in nodes['kind'].value_counts().items())})",
           f"  permitted pairs: {int(((Q >= 0) & off).sum())}; "
           + ", ".join(f"class {k}: {int(((Q == k) & off).sum())}" for k in (1, 2, 0)),
           "", "DAG 7, non-informative setting"]
    r7 = det[(det["setting"] == "noninfo") & (det["sim"] == 7)].iloc[0]
    out.append(f"  true edges {r7.true_edges}, called {r7.called}, correct {r7.correct}; "
               f"precision {r7.precision:.2f}, recall {r7.recall:.2f}, PR-AUC {r7.pr_auc:.3f}")
    out += ["", "Mean PR-AUC over the ten DAGs (+/- SE)"]
    for s in METHODS:
        v = det.loc[det["setting"] == s, "pr_auc"]
        label = cfg.SETTING_LABELS_FLAT.get(s, cfg.METHOD_SHORT_LABELS[s].replace("\n", " "))
        out.append(f"  {label}: {v.mean():.3f} +/- {v.std(ddof=1) / np.sqrt(len(v)):.3f}")
    with open(os.path.join(cfg.TABLE_DIR, "results.txt"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
