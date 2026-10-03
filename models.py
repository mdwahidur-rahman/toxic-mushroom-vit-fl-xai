"""Backbones, centralised fine-tuning, and simulated federated learning.

Federated protocol: FedAvg (McMahan et al., 2017) with an optional FedProx
proximal term (Li et al., 2020). Statistical heterogeneity across clients is
induced with a Dirichlet(alpha) label partition - the standard non-IID
benchmark - where small alpha means each site sees a skewed toxic/edible mix,
which is exactly the situation of geographically separated mycology collections.
"""
from __future__ import annotations
import copy
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import data as D

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BACKBONES = {                      # short name -> timm id
    "vit_t": "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
    "vit_s": "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "vit_b": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "deit_s": "deit3_small_patch16_224.fb_in1k",
    "swin_t": "swin_tiny_patch4_window7_224.ms_in1k",
    "resnet50": "resnet50.a1_in1k",
    "effb0": "efficientnet_b0.ra_in1k",
}


def set_seed(s: int):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(name="vit_s", pretrained=True, drop_path=0.1) -> nn.Module:
    import timm
    kw = dict(pretrained=pretrained, num_classes=2)
    if name.startswith(("vit", "deit", "swin")):
        kw["drop_path_rate"] = drop_path
    return timm.create_model(BACKBONES.get(name, name), **kw).to(DEVICE)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ------------------------------------------------------------------ train/eval
def _make_optimizer(model, lr, wd, layer_decay=None):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


def train_one_epoch(model, loader, opt, lossf, scaler, scheduler=None,
                    mu: float = 0.0, global_params=None, clip=1.0) -> float:
    model.train()
    tot, nb = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(DEVICE, enabled=(DEVICE == "cuda")):
            loss = lossf(model(x), y)
        if mu > 0 and global_params is not None:          # FedProx proximal regulariser
            prox = sum(((p - g.to(p.device)) ** 2).sum() for p, g in zip(model.parameters(), global_params))
            loss = loss + 0.5 * mu * prox
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(opt); scaler.update()
        if scheduler is not None:
            scheduler.step()
        tot += float(loss.detach()); nb += 1
    return tot / max(nb, 1)


@torch.no_grad()
def predict(model, loader) -> tuple[np.ndarray, np.ndarray]:
    """Returns (y_true, p_toxic)."""
    model.eval()
    Y, P = [], []
    for x, y in loader:
        with torch.autocast(DEVICE, enabled=(DEVICE == "cuda")):
            logits = model(x.to(DEVICE, non_blocking=True))
        P.append(F.softmax(logits.float(), 1)[:, 1].cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(Y), np.concatenate(P)


def _val_score(y, p):
    """Model selection criterion: balanced accuracy (imbalance-robust)."""
    from sklearn.metrics import balanced_accuracy_score
    return balanced_accuracy_score(y, (p >= 0.5).astype(int))


def train_centralized(df, cfg) -> dict:
    """Standard fine-tuning baseline. Returns state dict, history and timings."""
    set_seed(cfg["seed"])
    tr, va, te = D.loaders(df, cfg["bs"], cfg["img_size"], cfg["workers"])
    model = build_model(cfg["model"], cfg["pretrained"])
    lossf = nn.CrossEntropyLoss(weight=D.class_weights(df).to(DEVICE),
                                label_smoothing=cfg.get("label_smoothing", 0.05))
    opt = _make_optimizer(model, cfg["lr"], cfg["wd"])
    steps = max(len(tr) * cfg["epochs"], 1)
    warm = max(int(0.1 * steps), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1))))
    scaler = torch.amp.GradScaler(DEVICE, enabled=(DEVICE == "cuda"))

    hist, best, best_state, t0 = [], -1.0, None, time.time()
    for ep in range(cfg["epochs"]):
        tl = train_one_epoch(model, tr, opt, lossf, scaler, sched)
        yv, pv = predict(model, va)
        s = _val_score(yv, pv)
        hist.append(dict(epoch=ep, train_loss=tl, val_balanced_accuracy=s,
                         lr=opt.param_groups[0]["lr"], elapsed_s=time.time() - t0))
        if s > best:
            best, best_state = s, copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        if cfg.get("verbose", True):
            print(f"  [central] ep{ep:02d} loss {tl:.4f} val_bacc {s:.4f}", flush=True)
    model.load_state_dict(best_state)
    return dict(model=model, history=pd.DataFrame(hist), best_val=best,
                train_seconds=time.time() - t0, n_params=n_params(model))


# ------------------------------------------------------------------ federated
def dirichlet_partition(y: np.ndarray, n_clients: int, alpha: float, seed=42, min_size=16):
    """Label-skewed non-IID partition; retries until every client is usable."""
    rng = np.random.RandomState(seed)
    for _ in range(200):
        parts = [[] for _ in range(n_clients)]
        for c in np.unique(y):
            idx = rng.permutation(np.where(y == c)[0])
            prop = rng.dirichlet([alpha] * n_clients)
            cuts = (np.cumsum(prop) * len(idx)).astype(int)[:-1]
            for i, s in enumerate(np.split(idx, cuts)):
                parts[i] += s.tolist()
        if min(len(p) for p in parts) >= min_size:
            return [np.array(sorted(p)) for p in parts]
    return [np.array(sorted(p)) for p in parts]


def partition_report(parts, y, out_csv=None) -> pd.DataFrame:
    rows = []
    for i, p in enumerate(parts):
        lab = y[p]
        rows.append(dict(client=i, n=len(p), n_edible=int((lab == 0).sum()),
                         n_toxic=int((lab == 1).sum()),
                         toxic_fraction=float((lab == 1).mean())))
    df = pd.DataFrame(rows)
    # Kullback-Leibler divergence of each client's label mix from the global mix
    g = np.array([(y == 0).mean(), (y == 1).mean()])
    df["kl_from_global"] = [float(np.sum(np.where(q > 0, q * np.log((q + 1e-12) / g), 0)))
                            for q in (df[["n_edible", "n_toxic"]].values / df[["n"]].values)]
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df


def fedavg_aggregate(states, sizes):
    """Sample-weighted parameter average (McMahan et al.)."""
    w = np.asarray(sizes, dtype=np.float64); w /= w.sum()
    out = copy.deepcopy(states[0])
    for k in out:
        if out[k].is_floating_point():
            out[k] = sum(float(wi) * s[k].float() for wi, s in zip(w, states)).to(out[k].dtype)
        else:                                            # e.g. num_batches_tracked
            out[k] = states[int(np.argmax(w))][k]
    return out


def train_federated(df, cfg) -> dict:
    """Simulated cross-silo FL. One process, sequential client updates."""
    set_seed(cfg["seed"])
    tr_df = df[(df.split == "train") & df.is_representative].reset_index(drop=True)
    parts = dirichlet_partition(tr_df.y.values, cfg["clients"], cfg["alpha"], cfg["seed"])
    part_df = partition_report(parts, tr_df.y.values)

    _, va, te = D.loaders(df, cfg["bs"], cfg["img_size"], cfg["workers"])
    client_loaders = [
        DataLoader(D.MushroomDataset(tr_df.iloc[p], True, cfg["img_size"]),
                   batch_size=cfg["bs"], shuffle=True, num_workers=cfg["workers"],
                   pin_memory=torch.cuda.is_available(), drop_last=len(p) > cfg["bs"])
        for p in parts]

    g = build_model(cfg["model"], cfg["pretrained"])
    pbytes = n_params(g) * 4 / 1e6                        # MB per model transfer (fp32)
    scaler = torch.amp.GradScaler(DEVICE, enabled=(DEVICE == "cuda"))
    hist, best, best_state, t0 = [], -1.0, None, time.time()

    for rnd in range(cfg["rounds"]):
        sel = np.random.RandomState(cfg["seed"] + rnd).choice(
            len(parts), max(1, int(round(cfg.get("participation", 1.0) * len(parts)))), replace=False)
        gparams = [p.detach().clone() for p in g.parameters()]
        states, sizes, closs = [], [], []
        for cid in sel:
            local = copy.deepcopy(g)
            yl = tr_df.y.values[parts[cid]]
            cnt = np.bincount(yl, minlength=2).astype(float)
            w = torch.tensor(len(yl) / (2 * np.maximum(cnt, 1)), dtype=torch.float)
            lossf = nn.CrossEntropyLoss(weight=w.to(DEVICE),
                                        label_smoothing=cfg.get("label_smoothing", 0.05))
            opt = _make_optimizer(local, cfg["lr"], cfg["wd"])
            for _ in range(cfg["local_epochs"]):
                l = train_one_epoch(local, client_loaders[cid], opt, lossf, scaler,
                                    mu=cfg.get("mu", 0.0), global_params=gparams)
            states.append({k: v.cpu() for k, v in local.state_dict().items()})
            sizes.append(len(parts[cid])); closs.append(l)
            del local
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        g.load_state_dict(fedavg_aggregate(states, sizes))
        yv, pv = predict(g, va)
        s = _val_score(yv, pv)
        hist.append(dict(round=rnd, clients_sampled=len(sel), mean_client_loss=float(np.mean(closs)),
                         val_balanced_accuracy=s, cumulative_comm_MB=(rnd + 1) * len(sel) * 2 * pbytes,
                         elapsed_s=time.time() - t0))
        if s > best:
            best, best_state = s, copy.deepcopy({k: v.cpu() for k, v in g.state_dict().items()})
        if cfg.get("verbose", True):
            print(f"  [fed] round {rnd:02d} val_bacc {s:.4f} comm {hist[-1]['cumulative_comm_MB']:.0f}MB", flush=True)

    g.load_state_dict(best_state)
    return dict(model=g, history=pd.DataFrame(hist), partition=part_df, best_val=best,
                train_seconds=time.time() - t0, n_params=n_params(g),
                comm_MB=float(hist[-1]["cumulative_comm_MB"]) if hist else 0.0)
