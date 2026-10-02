"""Explainability: CAM family + attention rollout, with quantitative faithfulness.

Saliency maps are implemented directly (no third-party CAM dependency) so the
token-grid reshaping for ViT is explicit and auditable, and so the same code
path serves CNN baselines.

We do not present saliency as evidence on its own. Each map is scored by
    - deletion AUC  (lower is better): remove most-salient pixels first
    - insertion AUC (higher is better): add most-salient pixels to a blurred image
and submitted to the cascading model-randomisation sanity check of
Adebayo et al. (2018): a map that does not change when the network's weights
are randomised is not explaining the network.
"""
from __future__ import annotations
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from .models import DEVICE


# ------------------------------------------------------------------ plumbing
def _is_vit(model) -> bool:
    return hasattr(model, "blocks") and hasattr(model, "patch_embed")


def target_layer(model):
    if _is_vit(model):
        return model.blocks[-1].norm1
    for attr in ("layer4", "conv_head", "features"):
        if hasattr(model, attr):
            m = getattr(model, attr)
            return m[-1] if hasattr(m, "__getitem__") and not isinstance(m, torch.nn.Conv2d) else m
    return list(model.modules())[-3]


class _Tap:
    """Captures activations and gradients of one layer for a single forward/backward."""

    def __init__(self, layer):
        self.a = self.g = None
        self.h = [layer.register_forward_hook(self._fw)]

    def _fw(self, m, i, o):
        self.a = o
        if o.requires_grad:
            self.h.append(o.register_hook(self._bw))

    def _bw(self, grad):
        self.g = grad

    def close(self):
        for h in self.h:
            h.remove()


def _to_grid(t: torch.Tensor) -> torch.Tensor:
    """(B,N,C) token sequence -> (B,C,h,w); (B,C,H,W) passes through."""
    if t.dim() == 4:
        return t
    x = t[:, 1:, :] if t.shape[1] % 2 == 1 else t           # drop CLS when present
    s = int(round(x.shape[1] ** 0.5))
    return x.reshape(x.shape[0], s, s, x.shape[2]).permute(0, 3, 1, 2)


def _norm(c: torch.Tensor, size=224) -> np.ndarray:
    c = F.interpolate(c[:, None], size=(size, size), mode="bilinear", align_corners=False)[:, 0]
    c = c - c.amin(dim=(1, 2), keepdim=True)
    c = c / (c.amax(dim=(1, 2), keepdim=True) + 1e-8)
    return c.detach().cpu().numpy()


# ------------------------------------------------------------------ CAMs
def grad_cam(model, x, cls, pp=False) -> np.ndarray:
    """Grad-CAM (Selvaraju 2017); pp=True -> Grad-CAM++ (Chattopadhay 2018)."""
    tap = _Tap(target_layer(model))
    model.zero_grad(set_to_none=True)
    out = model(x)
    score = out.gather(1, cls.view(-1, 1)).sum()
    score.backward(retain_graph=False)
    A, G = _to_grid(tap.a), _to_grid(tap.g)
    tap.close()
    if pp:
        g2, g3 = G ** 2, G ** 3
        denom = 2 * g2 + (A * g3).sum(dim=(2, 3), keepdim=True)
        alpha = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
        w = (alpha * F.relu(G)).sum(dim=(2, 3), keepdim=True)
    else:
        w = G.mean(dim=(2, 3), keepdim=True)
    return _norm(F.relu((w * A).sum(1)), x.shape[-1])


def eigen_cam(model, x, cls=None) -> np.ndarray:
    """Eigen-CAM (Muhammad 2020): first principal component of the activations."""
    tap = _Tap(target_layer(model))
    with torch.no_grad():
        model(x)
    A = _to_grid(tap.a).float()
    tap.close()
    B, C, H, W = A.shape
    cams = []
    for b in range(B):
        M = A[b].reshape(C, H * W)
        M = M - M.mean(1, keepdim=True)
        U, S, V = torch.linalg.svd(M, full_matrices=False)
        cams.append((V[0] * S[0]).reshape(H, W).abs())
    return _norm(torch.stack(cams), x.shape[-1])


def attention_rollout(model, x, head_fusion="mean", discard_ratio=0.0) -> np.ndarray:
    """Abnar & Zuidema (2020). Requires the non-fused attention path to expose maps."""
    if not _is_vit(model):
        return np.zeros((x.shape[0], x.shape[-1], x.shape[-1]), dtype=np.float32)
    maps, hooks, saved = [], [], []
    for blk in model.blocks:
        saved.append(getattr(blk.attn, "fused_attn", None))
        blk.attn.fused_attn = False
        hooks.append(blk.attn.attn_drop.register_forward_hook(lambda m, i, o: maps.append(i[0].detach())))
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    for blk, s in zip(model.blocks, saved):
        if s is not None:
            blk.attn.fused_attn = s
    n = maps[0].shape[-1]
    R = torch.eye(n, device=x.device).expand(x.shape[0], n, n).clone()
    for A in maps:
        A = A.max(1).values if head_fusion == "max" else A.mean(1)
        if discard_ratio > 0:
            k = int(A.shape[-1] ** 2 * discard_ratio)
            flat = A.reshape(A.shape[0], -1)
            idx = flat.topk(k, -1, largest=False).indices
            flat.scatter_(-1, idx, 0)
            A = flat.reshape_as(A)
        A = A + torch.eye(n, device=x.device)[None]
        A = A / A.sum(-1, keepdim=True)
        R = A @ R
    s = int(round((n - 1) ** 0.5))
    return _norm(R[:, 0, 1:].reshape(-1, s, s), x.shape[-1])


METHODS = {
    "Grad-CAM": lambda m, x, c: grad_cam(m, x, c, pp=False),
    "Grad-CAM++": lambda m, x, c: grad_cam(m, x, c, pp=True),
    "Eigen-CAM": lambda m, x, c: eigen_cam(m, x, c),
    "Attention rollout": lambda m, x, c: attention_rollout(m, x),
}


# ------------------------------------------------------------------ faithfulness
@torch.no_grad()
def _curve(model, x, cam, cls, mode, steps=25, baseline=None):
    H = x.shape[-1]
    order = np.argsort(-cam.reshape(-1))
    base = baseline if baseline is not None else torch.zeros_like(x)
    scores, frac = [], []
    for s in range(steps + 1):
        k = int(len(order) * s / steps)
        mask = torch.ones(H * H, device=x.device)
        mask[torch.as_tensor(order[:k].copy(), device=x.device)] = 0.0
        mask = mask.view(1, 1, H, H)
        inp = x * mask + base * (1 - mask) if mode == "deletion" else x * (1 - mask) + base * mask
        scores.append(float(F.softmax(model(inp).float(), 1)[0, cls]))
        frac.append(s / steps)
    return np.array(frac), np.array(scores)


def faithfulness(model, x, cam, cls, steps=25):
    """Deletion (lower better) and insertion (higher better) AUC for one map."""
    blur = _gaussian_blur(x, 11, 5.0)
    f, d = _curve(model, x, cam, cls, "deletion", steps)
    _, i = _curve(model, x, cam, cls, "insertion", steps, baseline=blur)
    return dict(deletion_auc=float(np.trapezoid(d, f)), insertion_auc=float(np.trapezoid(i, f)),
                faithfulness_gap=float(np.trapezoid(i, f) - np.trapezoid(d, f)))


def _gaussian_blur(x, k=11, sigma=5.0):
    c = torch.arange(k, device=x.device, dtype=torch.float32) - (k - 1) / 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2)); g /= g.sum()
    ker = (g[:, None] @ g[None, :]).expand(x.shape[1], 1, k, k)
    return F.conv2d(F.pad(x, (k // 2,) * 4, mode="reflect"), ker, groups=x.shape[1])


def sanity_check(model, x, cls, method="Grad-CAM", n_blocks=4, seed=0) -> dict:
    """Cascading randomisation (Adebayo 2018): correlation must COLLAPSE."""
    base = METHODS[method](model, x, cls)[0]
    m2 = copy.deepcopy(model)
    torch.manual_seed(seed)
    blocks = list(m2.blocks) if _is_vit(m2) else [m2]
    for blk in blocks[-n_blocks:]:
        for p in blk.parameters():
            torch.nn.init.normal_(p, 0, 0.02) if p.dim() > 1 else torch.nn.init.zeros_(p)
    rnd = METHODS[method](m2, x, cls)[0]
    del m2
    degenerate = base.std() < 1e-8 or rnd.std() < 1e-8
    rho = 0.0 if degenerate else float(spearmanr(base.flatten(), rnd.flatten()).statistic)
    if not np.isfinite(rho):
        rho, degenerate = 0.0, True
    return dict(method=method, randomized_blocks=n_blocks, spearman_rho=rho,
                degenerate_map=bool(degenerate),
                passes_sanity=bool(abs(rho) < 0.5),
                note="randomised map is constant - correlation undefined, treated as 0"
                     if degenerate else "")


# ------------------------------------------------------------------ driver
def explain_dataset(model, frame, img_size=224, n=12, steps=25, seed=0, do_sanity=True):
    """Compute every map + faithfulness score for a stratified sample of test images."""
    from . import data as D
    model.eval()
    rng = np.random.RandomState(seed)
    # Stratified by true label and INTERLEAVED, so any truncated grid still shows
    # both classes rather than a block of one.
    per = {lab: list(rng.choice(frame[frame.y == lab].index.values,
                                min(n - n // 2, int((frame.y == lab).sum())), replace=False))
           for lab in [0, 1]}
    sel = [i for pair in zip(*([per[0], per[1]])) for i in pair][:n]
    sel += [i for lab in [0, 1] for i in per[lab] if i not in sel][:max(0, n - len(sel))]
    sub = frame.loc[sel]
    ds = D.MushroomDataset(sub, train=False, img_size=img_size)

    recs, panels = [], []
    for i in range(len(ds)):
        xt, y = ds[i]
        x = xt[None].to(DEVICE)
        with torch.no_grad():
            p = F.softmax(model(x).float(), 1)[0]
        pred = int(p.argmax()); cls = torch.tensor([pred], device=DEVICE)
        maps = {}
        for name, fn in METHODS.items():
            try:
                cam = fn(model, x, cls)[0]
            except Exception as e:                       # e.g. rollout on a CNN
                cam = np.zeros((img_size, img_size), dtype=np.float32)
                print(f"    [xai] {name} failed: {type(e).__name__}", flush=True)
            maps[name] = cam
            f = faithfulness(model, x, cam, pred, steps)
            recs.append(dict(image=sub.iloc[i].path, y_true=int(y), y_pred=pred,
                             correct=int(pred == y), p_toxic=float(p[1]),
                             method=name, **f))
        panels.append(dict(img=D.denormalize(xt), maps=maps, y=int(y), pred=pred,
                           p=float(p[1]), path=sub.iloc[i].path))

    df = pd.DataFrame(recs)
    san = pd.DataFrame([sanity_check(model, torch.as_tensor(ds[0][0])[None].to(DEVICE),
                                     torch.tensor([0], device=DEVICE), m, seed=seed)
                        for m in ["Grad-CAM", "Grad-CAM++"]]) if do_sanity else pd.DataFrame()
    return df, panels, san
