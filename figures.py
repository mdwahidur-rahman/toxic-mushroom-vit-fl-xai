"""Every manuscript figure, generated from the CSV artefacts only.

Charts use the validated categorical palette (CVD-checked); saliency overlays use
a perceptually uniform luminance ramp, which is the convention for attribution
maps and keeps intensity ordering readable in greyscale print.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

from . import viz
from .viz import C, SERIES, SEQ, INK, INK2, MUTED, CRIT, COL1, COL2, save, panel_tag

CAM_CMAP = "inferno"
BLUES = LinearSegmentedColormap.from_list("blues", SEQ)


def _series_color(i):
    return SERIES[i % len(SERIES)]


# ------------------------------------------------------------------ Fig 1
def fig_dataset(manifest: pd.DataFrame, splits: pd.DataFrame, outdir):
    fig = plt.figure(figsize=(COL2, 2.0))
    gs = GridSpec(1, 3, width_ratios=[1, 1.15, 1], wspace=0.38, figure=fig)

    ax = fig.add_subplot(gs[0])
    cnt = manifest.label.value_counts().reindex(["edible", "toxic"])
    bars = ax.bar(["Edible", "Toxic"], cnt.values, width=0.6,
                  color=[C["aqua"], C["orange"]], zorder=3)
    for b, v in zip(bars, cnt.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 40, f"{v:,}", ha="center",
                fontsize=7.5, fontweight="bold", color=INK)
    ax.set_ylabel("Images"); ax.set_ylim(0, cnt.max() * 1.18)
    ax.set_title("Class composition"); ax.xaxis.grid(False)
    panel_tag(ax, "a")

    ax = fig.add_subplot(gs[1])
    px = np.sqrt(manifest.width * manifest.height)
    ax.hist(px, bins=np.logspace(np.log10(px.min()), np.log10(px.max()), 40),
            color=C["blue"], edgecolor="white", linewidth=0.3, zorder=3)
    ax.set_xscale("log"); ax.set_xlabel("Geometric mean side length (px)")
    ax.set_ylabel("Images"); ax.set_title("Native resolution")
    ax.axvline(224, color=CRIT, lw=1.2, ls="--", zorder=4)
    ax.annotate("model input\n224 px", xy=(224, ax.get_ylim()[1] * 0.78), xytext=(6, 0),
                textcoords="offset points", fontsize=6.8, color=CRIT, fontweight="bold", va="top")
    panel_tag(ax, "b")

    ax = fig.add_subplot(gs[2])
    s = splits[splits.split.isin(["train", "val", "test"])]
    bot = np.zeros(len(s))
    for i, (k, lab) in enumerate([("edible", "Edible"), ("toxic", "Toxic")]):
        col = [C["aqua"], C["orange"]][i]
        ax.bar(s.split.str.capitalize(), s[k], bottom=bot, width=0.6, color=col,
               label=lab, zorder=3, edgecolor="white", linewidth=0.8)
        bot += s[k].values
    for x, (t, n) in enumerate(zip(s.split, s.images)):
        ax.text(x, n + 40, f"{n:,}", ha="center", fontsize=7, fontweight="bold", color=INK)
    ax.set_ylabel("Images"); ax.set_title("Leakage-safe splits")
    ax.set_ylim(0, s.images.max() * 1.2); ax.xaxis.grid(False)
    ax.legend(loc="upper right", ncol=1)
    panel_tag(ax, "c")
    return save(fig, outdir, "fig1_dataset")


# ------------------------------------------------------------------ Fig 2
def fig_convergence(central_hist: pd.DataFrame, fed_hists: dict, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.2))

    ax = axes[0]
    ax.plot(central_hist.epoch, central_hist.train_loss, color=C["blue"], marker="o", ms=3)
    viz.label_end(ax, central_hist.epoch.iloc[-1], central_hist.train_loss.iloc[-1],
                  "train loss", C["blue"])
    ax.set_xlabel("Epoch"); ax.set_ylabel("Training loss")
    ax.set_title("Centralised fine-tuning")
    ax.set_xlim(-0.3, central_hist.epoch.max() + 1.4)
    a2 = ax.twiny(); a2.axis("off")
    panel_tag(ax, "a")

    ax = axes[1]
    for i, (name, h) in enumerate(fed_hists.items()):
        col = _series_color(i)
        ax.plot(h["round"] + 1, h.val_balanced_accuracy, color=col, marker="o", ms=3, label=name)
        viz.label_end(ax, h["round"].iloc[-1] + 1, h.val_balanced_accuracy.iloc[-1], name, col)
    if central_hist is not None:
        b = central_hist.val_balanced_accuracy.max()
        ax.axhline(b, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
        ax.annotate("centralised ceiling", xy=(1, b), xytext=(2, 3), textcoords="offset points",
                    fontsize=6.8, color=INK2, va="bottom")
    ax.set_xlabel("Communication round"); ax.set_ylabel("Validation balanced accuracy")
    ax.set_title("Federated convergence")
    xmax = max(h["round"].max() for h in fed_hists.values()) + 1
    ax.set_xlim(0.6, xmax * 1.30)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(loc="lower right", ncol=1)
    panel_tag(ax, "b")
    fig.subplots_adjust(wspace=0.32)
    return save(fig, outdir, "fig2_convergence")


# ------------------------------------------------------------------ Fig 3
def fig_roc_pr(bands: dict, aucs: dict, outdir):
    """bands[name] = (roc_df, pr_df) with columns x, mean, lo, hi."""
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.6))
    for ax, k, (xl, yl, ttl) in zip(
            axes, [0, 1],
            [("False positive rate", "True positive rate (toxic recall)", "ROC"),
             ("Recall (toxic)", "Precision (toxic)", "Precision-recall")]):
        for i, (name, pair) in enumerate(bands.items()):
            d = pair[k]; col = _series_color(i)
            ax.fill_between(d.x, d.lo, d.hi, color=col, alpha=0.16, lw=0, zorder=2)
            ax.plot(d.x, d["mean"], color=col, lw=1.8, zorder=3,
                    label=f"{name}  {aucs[name][k]:.3f}")
        if k == 0:
            ax.plot([0, 1], [0, 1], color=MUTED, lw=0.9, ls=(0, (3, 3)), zorder=1)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(ttl)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right" if k == 0 else "lower left",
                  title="AUC (95% CI band)", title_fontsize=7, alignment="left")
        panel_tag(ax, "ab"[k])
    fig.subplots_adjust(wspace=0.30)
    return save(fig, outdir, "fig3_roc_pr")


# ------------------------------------------------------------------ Fig 4
def fig_confusion(mats: dict, outdir):
    """mats[name] = 2x2 array [[TN,FP],[FN,TP]]."""
    n = len(mats)
    fig, axes = plt.subplots(1, n, figsize=(COL1 * 0.92 * n, 2.3), squeeze=False)
    for j, (name, m) in enumerate(mats.items()):
        ax = axes[0, j]
        row = m / np.maximum(m.sum(1, keepdims=True), 1)
        ax.imshow(row, cmap=BLUES, vmin=0, vmax=1)
        for r in range(2):
            for c in range(2):
                dangerous = (r == 1 and c == 0)
                ax.text(c, r - 0.10, f"{int(m[r, c]):,}", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if row[r, c] > 0.45 else INK)
                ax.text(c, r + 0.22, f"{100*row[r,c]:.1f}%", ha="center", va="center",
                        fontsize=7, color="white" if row[r, c] > 0.45 else INK2)
                if dangerous:
                    ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, fill=False,
                                               edgecolor=CRIT, lw=1.8, zorder=5))
        ax.set_xticks([0, 1], ["Edible", "Toxic"]); ax.set_yticks([0, 1], ["Edible", "Toxic"])
        ax.set_xlabel("Predicted"); ax.set_title(name)
        if j == 0:
            ax.set_ylabel("True")
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        panel_tag(ax, "abcd"[j])
    fig.text(0.5, -0.16, "Red outline: toxic specimens classified as edible — the safety-critical error.",
             ha="center", fontsize=7, color=CRIT)
    fig.subplots_adjust(wspace=0.35)
    return save(fig, outdir, "fig4_confusion")


# ------------------------------------------------------------------ Fig 5
def fig_calibration(rel: dict, ece: dict, outdir):
    fig, ax = plt.subplots(figsize=(COL1, 2.4))
    ax.plot([0.5, 1], [0.5, 1], color=MUTED, lw=0.9, ls=(0, (3, 3)), zorder=1)
    ax.annotate("perfect calibration", xy=(0.74, 0.74), xytext=(0, -9),
                textcoords="offset points", fontsize=6.8, color=INK2, rotation=38, ha="center")
    for i, (name, d) in enumerate(rel.items()):
        d = d.dropna(subset=["confidence"]); col = _series_color(i)
        ax.plot(d.confidence, d.accuracy, color=col, marker="o", ms=4, lw=1.6,
                label=f"{name}  ECE {ece[name]:.3f}", zorder=3)
    ax.set_xlabel("Mean predicted confidence"); ax.set_ylabel("Observed accuracy")
    ax.set_title("Reliability"); ax.set_xlim(0.45, 1.02); ax.set_ylim(0.45, 1.02)
    ax.legend(loc="upper left")
    return save(fig, outdir, "fig5_calibration")


# ------------------------------------------------------------------ Fig 6
def fig_heterogeneity(sweep: pd.DataFrame, partitions: dict, outdir):
    """sweep: columns alpha, algorithm, metric, mean, ci95_lo, ci95_hi."""
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.3))

    ax = axes[0]
    for i, (alg, d) in enumerate(sweep.groupby("algorithm")):
        d = d.sort_values("alpha"); col = _series_color(i)
        ax.errorbar(d.alpha, d["mean"], yerr=[d["mean"] - d.ci95_lo, d.ci95_hi - d["mean"]],
                    color=col, marker="o", ms=4, lw=1.7, capsize=2, label=alg)
        viz.label_end(ax, d.alpha.iloc[-1], d["mean"].iloc[-1], alg, col)
    ax.set_xscale("log"); ax.set_xlabel(r"Dirichlet $\alpha$  (lower = more heterogeneous)")
    ax.set_ylabel("Test balanced accuracy"); ax.set_title("Robustness to non-IID data")
    ax.set_xlim(sweep.alpha.min() * 0.6, sweep.alpha.max() * 3.2)
    ax.legend(loc="lower right")
    panel_tag(ax, "a")

    ax = axes[1]
    key = sorted(partitions)[0]
    p = partitions[key]
    ax.bar(p.client, p.n_edible, color=C["aqua"], label="Edible", zorder=3,
           edgecolor="white", linewidth=0.8)
    ax.bar(p.client, p.n_toxic, bottom=p.n_edible, color=C["orange"], label="Toxic",
           zorder=3, edgecolor="white", linewidth=0.8)
    ax.set_xlabel("Client"); ax.set_ylabel("Local training images")
    ax.set_title(rf"Client label skew ($\alpha$={key})")
    ax.set_xticks(p.client); ax.xaxis.grid(False); ax.legend(loc="upper right")
    panel_tag(ax, "b")
    fig.subplots_adjust(wspace=0.32)
    return save(fig, outdir, "fig6_heterogeneity")


# ------------------------------------------------------------------ Fig 7
def fig_xai_grid(panels, outdir, methods=None, max_rows=6, name="fig7_xai_grid"):
    methods = methods or list(panels[0]["maps"].keys())
    rows = min(max_rows, len(panels))
    sel = panels[:rows]
    ncol = 1 + len(methods)
    fig, axes = plt.subplots(rows, ncol, figsize=(COL2, 1.22 * rows), squeeze=False)
    names = ["Edible", "Toxic"]
    for r, pn in enumerate(sel):
        ok = pn["pred"] == pn["y"]
        axes[r][0].imshow(pn["img"])
        axes[r][0].set_ylabel(f"true: {names[pn['y']]}\npred: {names[pn['pred']]}"
                              f"\np(toxic)={pn['p']:.2f}", fontsize=6.6,
                              color=INK if ok else CRIT, rotation=0, ha="right", va="center",
                              labelpad=26)
        if not ok:
            for s in axes[r][0].spines.values():
                s.set_visible(True); s.set_color(CRIT); s.set_linewidth(1.6)
        for c, m in enumerate(methods, 1):
            axes[r][c].imshow(pn["img"])
            im = axes[r][c].imshow(pn["maps"][m], cmap=CAM_CMAP, alpha=0.52, vmin=0, vmax=1)
        for c in range(ncol):
            axes[r][c].set_xticks([]); axes[r][c].set_yticks([]); axes[r][c].grid(False)
            if c > 0 or ok:
                for s in axes[r][c].spines.values():
                    s.set_visible(False)
    for c, t in enumerate(["Input"] + methods):
        axes[0][c].set_title(t, fontsize=7.5, loc="center", fontweight="bold", pad=4)
    cax = fig.add_axes([0.92, 0.12, 0.012, 0.76])
    cb = fig.colorbar(im, cax=cax); cb.set_label("Normalised attribution", fontsize=7)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6.5, color=MUTED)
    fig.subplots_adjust(wspace=0.04, hspace=0.06, right=0.90, left=0.11)
    return save(fig, outdir, name)


# ------------------------------------------------------------------ Fig 8
def fig_xai_faithfulness(x: pd.DataFrame, outdir):
    """x: per-image rows with method, deletion_auc, insertion_auc."""
    g = x.groupby("method").agg(d=("deletion_auc", "mean"), ds=("deletion_auc", "sem"),
                                i=("insertion_auc", "mean"), isem=("insertion_auc", "sem")).reset_index()
    order = g.sort_values("i", ascending=False).method.tolist()
    g = g.set_index("method").loc[order].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.2))
    for ax, (val, err, lab, good) in zip(axes, [("d", "ds", "Deletion AUC", "lower is better"),
                                                ("i", "isem", "Insertion AUC", "higher is better")]):
        col = C["blue"] if val == "i" else C["orange"]
        y = np.arange(len(g))
        ax.barh(y, g[val], xerr=g[err], height=0.6, color=col, zorder=3,
                error_kw=dict(ecolor=INK2, lw=0.9, capsize=2))
        for yi, v in zip(y, g[val]):
            ax.text(v + 0.012, yi, f"{v:.3f}", va="center", fontsize=7,
                    fontweight="bold", color=INK)
        ax.set_yticks(y, g.method, fontsize=7.5); ax.invert_yaxis()
        ax.set_xlabel(f"{lab}  ({good})"); ax.set_title(lab)
        ax.set_xlim(0, max(g[val]) * 1.25); ax.yaxis.grid(False)
        panel_tag(ax, "ab"[0 if val == "d" else 1])
    fig.subplots_adjust(wspace=0.45)
    return save(fig, outdir, "fig8_xai_faithfulness")


# ------------------------------------------------------------------ Fig 11
def fig_complexity(scaling: pd.DataFrame, comm: pd.DataFrame, outdir):
    """Measured cost against the asymptotic bounds."""
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.2))

    ax = axes[0]
    if len(scaling) >= 2:
        n, t = scaling.tokens.values, scaling.latency_ms_mean.values
        ax.plot(n, t, color=C["blue"], marker="o", ms=4.5, lw=1.7, zorder=4, label="measured")
        for ref, st, lab in [(1.0, (0, (4, 2)), r"$O(N)$"), (2.0, (0, (1.5, 1.5)), r"$O(N^2)$")]:
            ax.plot(n, t[0] * (n / n[0]) ** ref, color=MUTED, lw=1.0, ls=st, zorder=2, label=lab)
        ax.set_xscale("log"); ax.set_yscale("log")
        k = scaling.get("fitted_exponent_vs_tokens")
        if k is not None and len(k):
            ax.annotate(rf"fitted exponent $k$={float(k.iloc[0]):.2f}",
                        xy=(0.04, 0.92), xycoords="axes fraction", fontsize=7,
                        color=INK, fontweight="bold")
    ax.set_xlabel("Tokens $N$"); ax.set_ylabel("Forward latency (ms)")
    ax.set_title("Measured vs asymptotic time cost"); ax.legend(loc="lower right")
    panel_tag(ax, "a")

    ax = axes[1]
    for i, r in comm.reset_index(drop=True).iterrows():
        rounds = np.arange(1, int(r.rounds) + 1)
        col = _series_color(i)
        ax.plot(rounds, rounds * r.comm_MB_per_round / 1024, color=col, lw=1.7, marker="o", ms=3)
        viz.label_end(ax, rounds[-1], rounds[-1] * r.comm_MB_per_round / 1024, str(r.model), col)
    ax.set_xlabel("Communication round"); ax.set_ylabel("Cumulative transfer (GB)")
    ax.set_title(r"Federated communication  $O(R\,K\,|\theta|)$")
    ax.set_xlim(0.6, comm.rounds.max() * 1.28)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    panel_tag(ax, "b")
    fig.subplots_adjust(wspace=0.34)
    return save(fig, outdir, "fig11_complexity")


# ------------------------------------------------------------------ Fig 9
def fig_operating_point(y, probs: dict, target=0.95, outdir=".", ):
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(COL1, 2.4))
    for i, (name, p) in enumerate(probs.items()):
        fpr, tpr, thr = roc_curve(y, p); col = _series_color(i)
        ax.plot(thr[1:], tpr[1:], color=col, lw=1.7, label=f"{name}: toxic recall")
        ax.plot(thr[1:], 1 - fpr[1:], color=col, lw=1.3, ls=(0, (4, 2)),
                label=f"{name}: edible retention")
        ok = np.where(tpr >= target)[0]
        if len(ok):
            t = thr[ok[np.argmin(fpr[ok])]]
            ax.axvline(t, color=col, lw=0.9, ls=":", zorder=2)
    ax.axhline(target, color=CRIT, lw=1.1, ls="--", zorder=2)
    ax.annotate(f"required toxic recall = {target:.2f}", xy=(0.02, target), xytext=(0, 4),
                textcoords="offset points", fontsize=6.8, color=CRIT, fontweight="bold")
    ax.set_xlabel("Decision threshold on p(toxic)"); ax.set_ylabel("Rate")
    ax.set_title("Safety operating point"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.03)
    ax.legend(loc="lower center", fontsize=6.6)
    return save(fig, outdir, "fig9_operating_point")
