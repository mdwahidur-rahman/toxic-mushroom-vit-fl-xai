"""Evaluation statistics for a safety-critical binary classifier.

Everything is computed from a tidy per-sample prediction table
(path, y_true, p_toxic, split, run_id) so that confidence intervals,
significance tests and calibration all come from the SAME artefact and can be
recomputed by a reviewer without re-training.

Positive class = toxic (index 1). The operationally important error is the
FALSE NEGATIVE: a toxic mushroom called edible.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score,
                             roc_curve, precision_recall_curve, log_loss)

EPS = 1e-12


# ------------------------------------------------------------------ core
def point_metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    """All scalar metrics at one operating threshold."""
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    n = len(y)
    both = len(np.unique(y)) > 1
    sens = tp / max(tp + fn, 1)                       # toxic recall  (safety metric)
    spec = tn / max(tn + fp, 1)                       # edible recall
    return dict(
        threshold=thr, n=n, TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn),
        accuracy=accuracy_score(y, yh),
        error_rate=1.0 - accuracy_score(y, yh),
        balanced_accuracy=balanced_accuracy_score(y, yh),
        balanced_error_rate=1.0 - balanced_accuracy_score(y, yh),
        precision_toxic=precision_score(y, yh, zero_division=0),
        recall_toxic=sens,
        specificity=spec,
        f1_toxic=f1_score(y, yh, zero_division=0),
        f1_macro=f1_score(y, yh, average="macro", zero_division=0),
        npv=tn / max(tn + fn, 1),
        fnr_toxic_as_edible=fn / max(tp + fn, 1),     # the dangerous error rate
        fpr=fp / max(tn + fp, 1),
        mcc=matthews_corrcoef(y, yh) if both else np.nan,
        cohen_kappa=cohen_kappa_score(y, yh) if both else np.nan,
        youden_j=sens + spec - 1.0,
        roc_auc=roc_auc_score(y, p) if both else np.nan,
        pr_auc=average_precision_score(y, p) if both else np.nan,
        brier=float(np.mean((p - y) ** 2)),
        nll=log_loss(y, np.clip(p, EPS, 1 - EPS), labels=[0, 1]) if both else np.nan,
        ece=expected_calibration_error(y, p),
        mce=expected_calibration_error(y, p, worst=True),
    )


def expected_calibration_error(y, p, bins: int = 15, worst: bool = False) -> float:
    """ECE / MCE over equal-width confidence bins (Guo et al., 2017)."""
    conf = np.maximum(p, 1 - p)
    corr = ((p >= 0.5).astype(int) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    gaps, ws = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        gaps.append(abs(corr[m].mean() - conf[m].mean()))
        ws.append(m.mean())
    if not gaps:
        return float("nan")
    return float(max(gaps)) if worst else float(np.average(gaps, weights=ws))


def reliability_bins(y, p, bins: int = 10) -> pd.DataFrame:
    conf = np.maximum(p, 1 - p)
    corr = ((p >= 0.5).astype(int) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        rows.append(dict(bin_lo=lo, bin_hi=hi, n=int(m.sum()),
                         confidence=float(conf[m].mean()) if m.sum() else np.nan,
                         accuracy=float(corr[m].mean()) if m.sum() else np.nan))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ uncertainty
def bootstrap_ci(y, p, fn, n_boot=2000, alpha=0.05, seed=0, thr=0.5):
    """Stratified percentile bootstrap CI for any scalar metric."""
    rng = np.random.RandomState(seed)
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(i0, len(i0), True), rng.choice(i1, len(i1), True)])
        try:
            vals.append(fn(y[idx], p[idx], thr))
        except Exception:
            pass
    v = np.array([x for x in vals if np.isfinite(x)])
    if v.size == 0:
        return (np.nan, np.nan, np.nan)
    return float(np.mean(v)), float(np.percentile(v, 100 * alpha / 2)), float(np.percentile(v, 100 * (1 - alpha / 2)))


def metrics_with_ci(y, p, thr=0.5, n_boot=2000, seed=0,
                    keys=("accuracy", "error_rate", "balanced_accuracy", "balanced_error_rate",
                          "f1_toxic", "f1_macro", "recall_toxic", "specificity",
                          "precision_toxic", "npv", "fnr_toxic_as_edible", "fpr", "mcc",
                          "cohen_kappa", "roc_auc", "pr_auc", "brier", "ece")) -> pd.DataFrame:
    """Point estimate + 95% bootstrap CI + standard error for every headline metric.

    One resample -> one full metric sweep, so the cost is O(n_boot), not
    O(n_boot * n_metrics).
    """
    base = point_metrics(y, p, thr)
    rng = np.random.RandomState(seed)
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(i0, len(i0), True), rng.choice(i1, len(i1), True)])
        try:
            m = point_metrics(y[idx], p[idx], thr)
        except Exception:
            continue
        for k in keys:
            draws[k].append(m[k])
    rows = []
    for k in keys:
        v = np.asarray(draws[k], dtype=float)
        v = v[np.isfinite(v)]
        lo, hi = (np.percentile(v, 2.5), np.percentile(v, 97.5)) if v.size else (np.nan, np.nan)
        rows.append(dict(metric=k, value=base[k],
                         boot_mean=float(v.mean()) if v.size else np.nan,
                         boot_sd=float(v.std(ddof=1)) if v.size > 1 else np.nan,
                         ci_lo=lo, ci_hi=hi,
                         ci_halfwidth=(hi - lo) / 2 if np.isfinite(hi) else np.nan,
                         se=(hi - lo) / (2 * 1.96) if np.isfinite(hi) else np.nan,
                         n_boot=int(v.size)))
    return pd.DataFrame(rows)


def operating_point(y, p, target_recall=0.95) -> dict:
    """Lowest-FPR threshold that still reaches the required toxic recall.

    For a foraging-safety system the threshold is chosen by the tolerable
    miss rate, not by argmax accuracy.
    """
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(tpr >= target_recall)[0]
    if len(ok) == 0:
        return dict(target_recall=target_recall, achievable=False, threshold=np.nan)
    i = ok[np.argmin(fpr[ok])]
    t = float(np.clip(thr[i], 0, 1))
    m = point_metrics(y, p, t)
    return dict(target_recall=target_recall, achievable=True, threshold=t,
                recall_toxic=m["recall_toxic"], fpr=m["fpr"], specificity=m["specificity"],
                accuracy=m["accuracy"], precision_toxic=m["precision_toxic"],
                edible_rejected_pct=100 * m["fpr"])


# ------------------------------------------------------------------ comparisons
def mcnemar(y, pa, pb, thr=0.5) -> dict:
    """Exact McNemar test on the SAME test set - the correct paired test here."""
    a, b = (pa >= thr).astype(int) == y, (pb >= thr).astype(int) == y
    n01 = int((a & ~b).sum())        # A right, B wrong
    n10 = int((~a & b).sum())
    n = n01 + n10
    pval = 1.0 if n == 0 else float(min(1.0, 2 * stats.binom.cdf(min(n01, n10), n, 0.5)))
    return dict(n_only_A_correct=n01, n_only_B_correct=n10, n_discordant=n,
                statistic=(abs(n01 - n10) - 1) ** 2 / n if n else 0.0,
                p_value=pval,
                odds_ratio=(n01 / n10) if n10 else np.inf)


def paired_bootstrap_delta(y, pa, pb, metrics=("accuracy",), n_boot=2000, seed=0,
                           thr=0.5) -> pd.DataFrame:
    """CI on the DIFFERENCE between two models evaluated on identical samples.

    The SAME resample indices are applied to both models, which is what makes the
    comparison paired; all requested metrics share one bootstrap loop.
    """
    if isinstance(metrics, str):
        metrics = (metrics,)
    rng = np.random.RandomState(seed)
    d = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = rng.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        ma, mb = point_metrics(y[idx], pa[idx], thr), point_metrics(y[idx], pb[idx], thr)
        for m in metrics:
            d[m].append(ma[m] - mb[m])
    base_a, base_b = point_metrics(y, pa, thr), point_metrics(y, pb, thr)
    rows = []
    for m in metrics:
        v = np.asarray(d[m], dtype=float); v = v[np.isfinite(v)]
        rows.append(dict(metric=m, value_reference=base_a[m], value_comparison=base_b[m],
                         delta=base_a[m] - base_b[m],
                         ci_lo=float(np.percentile(v, 2.5)) if v.size else np.nan,
                         ci_hi=float(np.percentile(v, 97.5)) if v.size else np.nan,
                         p_two_sided=float(2 * min((v <= 0).mean(), (v >= 0).mean())) if v.size else np.nan))
    return pd.DataFrame(rows)


def aggregate_seeds(df: pd.DataFrame, group=("model",), value="value") -> pd.DataFrame:
    """mean +/- sd + 95% t-CI across independent seeds - the table for the paper."""
    g = df.groupby(list(group) + ["metric"])[value]
    out = g.agg(["count", "mean", "std", "min", "max"]).reset_index()
    out["sem"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    tcrit = stats.t.ppf(0.975, out["count"].clip(lower=2) - 1)
    out["ci95_lo"] = out["mean"] - tcrit * out["sem"]
    out["ci95_hi"] = out["mean"] + tcrit * out["sem"]
    out["latex"] = out.apply(
        lambda r: f"{r['mean']:.4f} $\\pm$ {0 if np.isnan(r['std']) else r['std']:.4f}", axis=1)
    return out


def curve_table(y, p, kind="roc") -> pd.DataFrame:
    if kind == "roc":
        fpr, tpr, thr = roc_curve(y, p)
        return pd.DataFrame(dict(x=fpr, y=tpr, threshold=thr))
    pr, rc, thr = precision_recall_curve(y, p)
    return pd.DataFrame(dict(x=rc, y=pr, threshold=np.append(thr, np.nan)))


def bootstrap_curve_band(y, p, grid=None, kind="roc", n_boot=400, seed=0) -> pd.DataFrame:
    """Pointwise 95% band for an ROC/PR curve on a fixed x-grid."""
    grid = np.linspace(0, 1, 101) if grid is None else grid
    rng = np.random.RandomState(seed)
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    ys = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(i0, len(i0), True), rng.choice(i1, len(i1), True)])
        c = curve_table(y[idx], p[idx], kind)
        x, yy = (c.x.values, c.y.values)
        o = np.argsort(x)
        ys.append(np.interp(grid, x[o], yy[o]))
    Y = np.vstack(ys)
    c = curve_table(y, p, kind)
    o = np.argsort(c.x.values)
    return pd.DataFrame(dict(x=grid, mean=np.interp(grid, c.x.values[o], c.y.values[o]),
                             lo=np.percentile(Y, 2.5, axis=0), hi=np.percentile(Y, 97.5, axis=0)))
