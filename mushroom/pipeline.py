"""End-to-end CLI: prep -> run -> analyze -> xai -> figures.

    python -m mushroom.pipeline all --data "<dataset root>" --out results

Every stage writes CSV artefacts; later stages read only those artefacts, so any
table or figure can be regenerated without touching a GPU.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import data as D, metrics as M, models as Mo, figures as Fg, viz


def _dirs(out):
    out = Path(out)
    for s in ["tables", "figures", "runs", "xai"]:
        (out / s).mkdir(parents=True, exist_ok=True)
    return out


def _cfg(a, **over):
    c = dict(model=a.model, pretrained=not a.no_pretrained, bs=a.bs, lr=a.lr, wd=a.wd,
             epochs=a.epochs, img_size=a.img_size, workers=a.workers, seed=a.seed,
             clients=a.clients, rounds=a.rounds, local_epochs=a.local_epochs,
             alpha=a.alpha, mu=0.0, participation=a.participation,
             label_smoothing=a.label_smoothing, verbose=True)
    c.update(over)
    return c


# --------------------------------------------------------------------- stages
def stage_prep(a):
    out = _dirs(a.out)
    t0 = time.time()
    print("[prep] auditing images ...", flush=True)
    man = D.build_manifest(a.data, out / "tables" / "manifest.csv")
    audit = D.audit_report(man, out / "tables" / "table1_dataset_audit.csv")
    man = D.group_stratified_split(man, a.val_frac, a.test_frac, a.seed)
    man.to_csv(out / "tables" / "manifest.csv", index=False)
    sp = D.split_report(man, out / "tables" / "table2_splits.csv")
    print(audit.to_string(index=False))
    print(sp.to_string(index=False))
    print(f"[prep] done in {time.time()-t0:.1f}s", flush=True)
    return man


def _load_manifest(out):
    p = Path(out) / "tables" / "manifest.csv"
    if not p.exists():
        raise SystemExit("run the 'prep' stage first")
    return pd.read_csv(p)


def _save_predictions(df_split, y, p, out, tag):
    rows = df_split[["path", "label", "y", "cluster"]].copy()
    rows["y_true"], rows["p_toxic"], rows["run"] = y, p, tag
    f = Path(out) / "runs" / f"predictions_{tag}.csv"
    rows.to_csv(f, index=False)
    return f


def stage_run(a):
    out = _dirs(a.out)
    man = _load_manifest(out)
    te_df = man[man.split == "test"].reset_index(drop=True)
    seeds = [int(s) for s in str(a.seeds).split(",")]
    index = []

    for seed in seeds:
        # ---- centralised
        tag = f"central_{a.model}_s{seed}"
        print(f"[run] {tag}", flush=True)
        r = Mo.train_centralized(man, _cfg(a, seed=seed))
        _, _, te = D.loaders(man, a.bs, a.img_size, a.workers)
        y, p = Mo.predict(r["model"], te)
        _save_predictions(te_df, y, p, out, tag)
        r["history"].to_csv(out / "runs" / f"history_{tag}.csv", index=False)
        if a.save_ckpt:
            torch.save(r["model"].state_dict(), out / "runs" / f"{tag}.pt")
        elif seed == seeds[0]:
            torch.save(r["model"].state_dict(), out / "runs" / "central_best.pt")
        index.append(dict(run=tag, paradigm="Centralised", algorithm="-", alpha=np.nan,
                          seed=seed, model=a.model, best_val=r["best_val"],
                          train_seconds=r["train_seconds"], n_params=r["n_params"], comm_MB=0.0))
        del r
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # ---- federated: algorithms x heterogeneity
        for alg, mu in [("FedAvg", 0.0)] + ([("FedProx", a.mu)] if a.fedprox else []):
            for alpha in [float(x) for x in str(a.alphas).split(",")]:
                tag = f"{alg.lower()}_a{alpha}_{a.model}_s{seed}"
                print(f"[run] {tag}", flush=True)
                r = Mo.train_federated(man, _cfg(a, seed=seed, alpha=alpha, mu=mu))
                y, p = Mo.predict(r["model"], te)
                _save_predictions(te_df, y, p, out, tag)
                r["history"].to_csv(out / "runs" / f"history_{tag}.csv", index=False)
                r["partition"].to_csv(out / "runs" / f"partition_{tag}.csv", index=False)
                if seed == seeds[0] and alpha == float(str(a.alphas).split(",")[0]) and alg == "FedAvg":
                    torch.save(r["model"].state_dict(), out / "runs" / "federated_best.pt")
                index.append(dict(run=tag, paradigm="Federated", algorithm=alg, alpha=alpha,
                                  seed=seed, model=a.model, best_val=r["best_val"],
                                  train_seconds=r["train_seconds"], n_params=r["n_params"],
                                  comm_MB=r["comm_MB"]))
                del r
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

    idx = pd.DataFrame(index)
    f = out / "tables" / "run_index.csv"
    if f.exists():
        idx = pd.concat([pd.read_csv(f), idx]).drop_duplicates("run", keep="last")
    idx.to_csv(f, index=False)
    print(idx.to_string(index=False))


def stage_analyze(a):
    out = _dirs(a.out)
    idx = pd.read_csv(out / "tables" / "run_index.csv")
    preds = {}
    for r in idx.run:
        f = out / "runs" / f"predictions_{r}.csv"
        if f.exists():
            preds[r] = pd.read_csv(f)
    if not preds:
        raise SystemExit("no predictions found - run the 'run' stage first")

    # ---- per-run metrics with bootstrap CI
    per_run = []
    for run, d in preds.items():
        y, p = d.y_true.values, d.p_toxic.values
        m = M.metrics_with_ci(y, p, n_boot=a.n_boot, seed=a.seed)
        m.insert(0, "run", run)
        per_run.append(m)
    per_run = pd.concat(per_run, ignore_index=True)
    per_run = per_run.merge(idx[["run", "paradigm", "algorithm", "alpha", "seed", "model",
                                 "comm_MB", "train_seconds"]], on="run", how="left")
    per_run.to_csv(out / "tables" / "table3_metrics_per_run.csv", index=False)

    # ---- aggregated across seeds (the headline table)
    per_run["config"] = per_run.apply(
        lambda r: "Centralised" if r.paradigm == "Centralised"
        else f"{r.algorithm} (alpha={r.alpha:g})", axis=1)
    agg = M.aggregate_seeds(per_run.rename(columns={"config": "model_config"}),
                            group=("model_config",), value="value")
    agg.to_csv(out / "tables" / "table4_metrics_mean_sd.csv", index=False)

    # ---- headline comparison: best federated vs centralised, same test set
    cen = [r for r in preds if r.startswith("central")]
    fed = [r for r in preds if not r.startswith("central")]
    comp = []
    if cen and fed:
        base = preds[cen[0]]
        y = base.y_true.values
        for r in fed:
            d = preds[r]
            if len(d) != len(y):
                continue
            mc = M.mcnemar(y, base.p_toxic.values, d.p_toxic.values)
            pb = M.paired_bootstrap_delta(
                y, base.p_toxic.values, d.p_toxic.values,
                ("accuracy", "balanced_accuracy", "f1_toxic", "recall_toxic", "roc_auc"),
                n_boot=max(a.n_boot // 2, 200), seed=a.seed)
            pb.insert(0, "reference", cen[0]); pb.insert(1, "comparison", r)
            pb["mcnemar_p"] = mc["p_value"]
            pb["mcnemar_discordant"] = mc["n_discordant"]
            pb["n_ref_only_correct"] = mc["n_only_A_correct"]
            pb["n_cmp_only_correct"] = mc["n_only_B_correct"]
            comp.append(pb)
    (pd.concat(comp, ignore_index=True) if comp else pd.DataFrame()).to_csv(
        out / "tables" / "table5_significance.csv", index=False)

    # ---- calibration + operating points + curves
    rel, opp, curves = [], [], []
    for run, d in preds.items():
        y, p = d.y_true.values, d.p_toxic.values
        r = M.reliability_bins(y, p, a.cal_bins); r.insert(0, "run", run); rel.append(r)
        for t in [0.90, 0.95, 0.99]:
            opp.append(dict(run=run, **M.operating_point(y, p, t)))
        for kind in ["roc", "pr"]:
            b = M.bootstrap_curve_band(y, p, kind=kind, n_boot=max(a.n_boot // 5, 100), seed=a.seed)
            b.insert(0, "run", run); b.insert(1, "curve", kind); curves.append(b)
    pd.concat(rel, ignore_index=True).to_csv(out / "tables" / "table6_reliability_bins.csv", index=False)
    pd.DataFrame(opp).to_csv(out / "tables" / "table7_operating_points.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(out / "tables" / "curves_roc_pr.csv", index=False)

    # ---- LaTeX-ready headline table
    head = per_run[per_run.metric.isin(
        ["accuracy", "error_rate", "balanced_accuracy", "f1_toxic", "recall_toxic",
         "specificity", "fnr_toxic_as_edible", "mcc", "roc_auc", "ece"])]
    piv = (head.groupby(["config", "metric"]).value.agg(["mean", "std"])
           .apply(lambda r: f"{r['mean']:.4f}" + ("" if np.isnan(r['std']) else f" $\\pm$ {r['std']:.4f}"), axis=1)
           .unstack("metric"))
    piv.to_csv(out / "tables" / "table8_headline_latex.csv")
    with open(out / "tables" / "table8_headline.tex", "w") as f:
        f.write(piv.to_latex(escape=False, column_format="l" + "r" * piv.shape[1]))
    print(piv.to_string())
    print(f"[analyze] wrote {len(list((out/'tables').glob('*.csv')))} CSV tables")


def stage_xai(a):
    from . import xai as X
    out = _dirs(a.out)
    man = _load_manifest(out)
    te = man[man.split == "test"].reset_index(drop=True)
    ck = Path(a.ckpt) if a.ckpt else out / "runs" / "central_best.pt"
    if not Path(ck).exists():
        raise SystemExit(f"checkpoint not found: {ck}")
    model = Mo.build_model(a.model, pretrained=False)
    model.load_state_dict(torch.load(ck, map_location=Mo.DEVICE))
    print(f"[xai] {ck.name}: {a.n_vis} images x {len(X.METHODS)} methods", flush=True)
    df, panels, san = X.explain_dataset(model, te, a.img_size, a.n_vis, a.xai_steps,
                                        a.seed, do_sanity=not a.no_sanity)
    df.to_csv(out / "tables" / "table9_xai_faithfulness_per_image.csv", index=False)
    summ = (df.groupby("method").agg(
        n=("deletion_auc", "size"),
        deletion_auc_mean=("deletion_auc", "mean"), deletion_auc_sd=("deletion_auc", "std"),
        deletion_auc_sem=("deletion_auc", "sem"),
        insertion_auc_mean=("insertion_auc", "mean"), insertion_auc_sd=("insertion_auc", "std"),
        insertion_auc_sem=("insertion_auc", "sem"),
        faithfulness_gap_mean=("faithfulness_gap", "mean")).reset_index()
        .sort_values("insertion_auc_mean", ascending=False))
    summ.to_csv(out / "tables" / "table10_xai_summary.csv", index=False)
    if len(san):
        san.to_csv(out / "tables" / "table11_xai_sanity_check.csv", index=False)
    np.save(out / "xai" / "panels.npy", np.array(panels, dtype=object), allow_pickle=True)
    print(summ.to_string(index=False))

    viz.use_style()
    Fg.fig_xai_grid(panels, out / "figures", max_rows=a.grid_rows)
    wrong = [p for p in panels if p["pred"] != p["y"]]
    if wrong:
        Fg.fig_xai_grid(wrong, out / "figures", max_rows=min(4, len(wrong)),
                        name="fig10_xai_failures")
    Fg.fig_xai_faithfulness(df, out / "figures")


def stage_complexity(a):
    """Asymptotic + measured time/space cost, and federated communication cost."""
    from . import complexity as Cx
    out = _dirs(a.out)
    models = [m.strip() for m in (a.complexity_models or a.model).split(",")]
    print(f"[complexity] profiling {models} on {Mo.DEVICE}", flush=True)
    rep = Cx.report(models, bs=a.bs, img=a.img_size, rounds=a.rounds,
                    clients=a.clients, reps=a.profile_reps,
                    out_csv=out / "tables" / "table13_complexity.csv")
    cols = ["model", "params_M", "model_size_MB", "gflops_forward", "latency_ms_per_image",
            "throughput_img_per_s", "peak_training_memory_MB", "comm_MB_total",
            "time_complexity", "space_complexity_activations"]
    print(rep[[c for c in cols if c in rep]].to_string(index=False))

    sc = Cx.empirical_scaling(models[0], out_csv=out / "tables" / "table14_scaling.csv",
                              reps=max(a.profile_reps // 2, 3))
    if len(sc):
        print(sc[["img_size", "tokens", "latency_ms_mean", "throughput_img_per_s"]].to_string(index=False))
    # measured wall-clock from the actual runs, for the cost table in the paper
    ridx = out / "tables" / "run_index.csv"
    if ridx.exists():
        idx = pd.read_csv(ridx)
        w = (idx.groupby(["paradigm", "algorithm"])
             .agg(runs=("run", "size"), train_seconds_mean=("train_seconds", "mean"),
                  train_seconds_sd=("train_seconds", "std"),
                  comm_MB_mean=("comm_MB", "mean")).reset_index())
        w.to_csv(out / "tables" / "table15_wallclock.csv", index=False)
        print(w.to_string(index=False))

    viz.use_style()
    if len(sc):
        Fg.fig_complexity(sc, rep, out / "figures")


def stage_figures(a):
    out = _dirs(a.out)
    viz.use_style()
    T = out / "tables"
    man = _load_manifest(out)
    idx = pd.read_csv(T / "run_index.csv")

    Fg.fig_dataset(man, pd.read_csv(T / "table2_splits.csv"), out / "figures")

    cen_runs = idx[idx.paradigm == "Centralised"].run.tolist()
    fed_runs = idx[idx.paradigm == "Federated"].run.tolist()
    ch = pd.read_csv(out / "runs" / f"history_{cen_runs[0]}.csv") if cen_runs else None
    fh = {}
    for r in fed_runs:
        row = idx[idx.run == r].iloc[0]
        if row.seed != idx.seed.min():
            continue
        fh[f"{row.algorithm} α={row.alpha:g}"] = pd.read_csv(out / "runs" / f"history_{r}.csv")
    if ch is not None and fh:
        Fg.fig_convergence(ch, fh, out / "figures")

    # headline runs: first centralised + first federated of the first seed
    head = ([cen_runs[0]] if cen_runs else []) + ([fed_runs[0]] if fed_runs else [])
    curves = pd.read_csv(T / "curves_roc_pr.csv")
    per_run = pd.read_csv(T / "table3_metrics_per_run.csv")
    bands, aucs, mats, rel, ece, probs, ytrue = {}, {}, {}, {}, {}, {}, None
    for r in head:
        nm = "Centralised" if r.startswith("central") else idx[idx.run == r].iloc[0].algorithm
        c = curves[curves.run == r]
        bands[nm] = (c[c.curve == "roc"], c[c.curve == "pr"])
        g = per_run[per_run.run == r].set_index("metric").value
        aucs[nm] = (g.get("roc_auc", np.nan), g.get("pr_auc", np.nan))
        d = pd.read_csv(out / "runs" / f"predictions_{r}.csv")
        ytrue = d.y_true.values; probs[nm] = d.p_toxic.values
        m = M.point_metrics(d.y_true.values, d.p_toxic.values)
        mats[nm] = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]], dtype=float)
        rb = pd.read_csv(T / "table6_reliability_bins.csv")
        rel[nm] = rb[rb.run == r]
        ece[nm] = float(g.get("ece", np.nan))
    if bands:
        Fg.fig_roc_pr(bands, aucs, out / "figures")
        Fg.fig_confusion(mats, out / "figures")
        Fg.fig_calibration(rel, ece, out / "figures")
        Fg.fig_operating_point(ytrue, probs, a.target_recall, out / "figures")

    # heterogeneity sweep
    fed = per_run[(per_run.paradigm == "Federated") & (per_run.metric == "balanced_accuracy")]
    if len(fed):
        sw = (fed.groupby(["algorithm", "alpha"]).value
              .agg(["mean", "std", "count"]).reset_index())
        sw["sem"] = sw["std"].fillna(0) / np.sqrt(sw["count"].clip(lower=1))
        sw["ci95_lo"] = sw["mean"] - 1.96 * sw["sem"]
        sw["ci95_hi"] = sw["mean"] + 1.96 * sw["sem"]
        sw.to_csv(T / "table12_heterogeneity_sweep.csv", index=False)
        parts = {}
        for f in sorted((out / "runs").glob("partition_*.csv")):
            al = float(f.stem.split("_a")[1].split("_")[0])
            parts.setdefault(al, pd.read_csv(f))
        if parts:
            Fg.fig_heterogeneity(sw, parts, out / "figures")
    print("[figures] ->", sorted(p.name for p in (out / 'figures').glob('*.pdf')))


# --------------------------------------------------------------------- CLI
def main(argv=None):
    p = argparse.ArgumentParser(description="ViT + Federated Learning + XAI for toxic mushroom ID")
    p.add_argument("stage", choices=["prep", "run", "analyze", "xai", "complexity", "figures", "all"])
    p.add_argument("--data", default="data"); p.add_argument("--out", default="results")
    p.add_argument("--model", default="vit_s"); p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--img_size", type=int, default=224); p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-5); p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=10); p.add_argument("--workers", type=int, default=2)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--val_frac", type=float, default=0.15); p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--clients", type=int, default=5); p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--local_epochs", type=int, default=1); p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--alphas", default="0.1,0.5,100"); p.add_argument("--participation", type=float, default=1.0)
    p.add_argument("--fedprox", action="store_true"); p.add_argument("--mu", type=float, default=0.01)
    p.add_argument("--n_boot", type=int, default=2000); p.add_argument("--cal_bins", type=int, default=10)
    p.add_argument("--target_recall", type=float, default=0.95)
    p.add_argument("--n_vis", type=int, default=12); p.add_argument("--grid_rows", type=int, default=6)
    p.add_argument("--xai_steps", type=int, default=25); p.add_argument("--no_sanity", action="store_true")
    p.add_argument("--ckpt", default=""); p.add_argument("--save_ckpt", action="store_true")
    p.add_argument("--complexity_models", default=""); p.add_argument("--profile_reps", type=int, default=12)
    a = p.parse_args(argv)

    viz.use_style()
    stages = ["prep", "run", "analyze", "xai", "complexity", "figures"] if a.stage == "all" else [a.stage]
    for s in stages:
        print(f"\n{'='*70}\n  STAGE: {s.upper()}\n{'='*70}", flush=True)
        {"prep": stage_prep, "run": stage_run, "analyze": stage_analyze, "xai": stage_xai,
         "complexity": stage_complexity, "figures": stage_figures}[s](a)


if __name__ == "__main__":
    main()
