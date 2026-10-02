"""Time and space complexity: asymptotic derivation + measured cost.

Asymptotics for one ViT forward pass (per image), with
    N = (S/P)^2 + 1 tokens,  D = embedding width,  L = depth,  H = heads

  Self-attention     QKV projection      O(N D^2)
                     scores  QK^T        O(N^2 D)
                     weighted sum AV     O(N^2 D)
                     output projection   O(N D^2)
  MLP (ratio r)                          O(r N D^2)
  Per block                              O(N^2 D + (4 + 2r) N D^2)
  Whole encoder                          O(L (N^2 D + N D^2))

  Space (activations, per image)         O(L (H N^2 + N D))
  Parameters                             O(L D^2) + patch embed O(P^2 C D)

Because N = 197 and D = 384 for ViT-S/16 at 224 px, the N D^2 term dominates the
quadratic N^2 D term by a factor of about D/N ~ 2, i.e. this model operates in
the regime where width, not sequence length, drives cost. That is the relevant
statement for a 224-px classification study and should be said explicitly rather
than quoting "attention is quadratic" as if it were the binding constraint.

Federated cost adds, per round, 2 * K * |theta| * 4 bytes of transfer (download
of the global model + upload of each client update), independent of dataset
size - communication is a function of the model, not of the data.
"""
from __future__ import annotations
import time

import numpy as np
import pandas as pd
import torch

from .models import DEVICE, build_model, n_params


def analytic_vit(img=224, patch=16, depth=12, width=384, heads=6, mlp_ratio=4.0) -> dict:
    N = (img // patch) ** 2 + 1
    D, L, r = width, depth, mlp_ratio
    attn_qkvo = 4 * N * D * D
    attn_quad = 2 * N * N * D
    mlp = 2 * r * N * D * D
    per_block = attn_qkvo + attn_quad + mlp
    return dict(tokens_N=N, width_D=D, depth_L=L, heads_H=heads,
                madds_attention_projections=attn_qkvo * L,
                madds_attention_quadratic=attn_quad * L,
                madds_mlp=int(mlp * L),
                madds_total=int(per_block * L),
                gmacs_forward=per_block * L / 1e9,          # the "GFLOPs" of the ViT papers
                gflops_forward=2 * per_block * L / 1e9,     # multiply-add counted as 2 ops
                quadratic_share=attn_quad / per_block,
                activation_floats_per_image=int(L * (heads * N * N + N * D)),
                time_complexity="O(L(N^2 D + N D^2))",
                space_complexity_activations="O(L(H N^2 + N D))",
                space_complexity_parameters="O(L D^2)")


@torch.no_grad()
def measure(model, bs=32, img=224, reps=12, warmup=4) -> dict:
    model.eval()
    x = torch.randn(bs, 3, img, img, device=DEVICE)
    for _ in range(warmup):
        model(x)
    if DEVICE == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        model(x)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    ts = np.array(ts)
    p = n_params(model)
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if DEVICE == "cuda" else float("nan")
    return dict(device=DEVICE, batch_size=bs, params=p, params_M=p / 1e6,
                model_size_MB=p * 4 / 2 ** 20,
                latency_ms_mean=1e3 * ts.mean(), latency_ms_sd=1e3 * ts.std(ddof=1),
                latency_ms_per_image=1e3 * ts.mean() / bs,
                throughput_img_per_s=bs / ts.mean(),
                peak_inference_memory_MB=peak)


def train_memory(model, bs=32, img=224) -> dict:
    """Peak memory with activations retained for backward - the training bound."""
    if DEVICE != "cuda":
        return dict(peak_training_memory_MB=float("nan"))
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(bs, 3, img, img, device=DEVICE)
    loss = model(x).square().mean()
    loss.backward()
    m = torch.cuda.max_memory_allocated() / 2 ** 20
    model.zero_grad(set_to_none=True)
    return dict(peak_training_memory_MB=m)


def federated_cost(n_params_: int, rounds: int, clients: int, participation=1.0,
                   bytes_per_param=4) -> dict:
    k = max(1, int(round(clients * participation)))
    per_round = 2 * k * n_params_ * bytes_per_param
    return dict(clients=clients, participating_per_round=k, rounds=rounds,
                payload_MB_per_transfer=n_params_ * bytes_per_param / 1e6,
                comm_MB_per_round=per_round / 1e6,
                comm_MB_total=per_round * rounds / 1e6,
                comm_complexity="O(R * K * |theta|)  - independent of dataset size")


def vit_config(model, img=224) -> dict | None:
    """Read depth/width/heads/patch off the instantiated model - never assume."""
    if not (hasattr(model, "blocks") and hasattr(model, "patch_embed")):
        return None
    ps = model.patch_embed.patch_size
    ps = ps[0] if isinstance(ps, (tuple, list)) else ps
    blk = model.blocks[0]
    width = getattr(model, "embed_dim", None) or blk.norm1.normalized_shape[0]
    heads = getattr(blk.attn, "num_heads", max(width // 64, 1))
    ratio = blk.mlp.fc1.out_features / width if hasattr(blk, "mlp") else 4.0
    return dict(img=img, patch=ps, depth=len(model.blocks), width=width,
                heads=heads, mlp_ratio=ratio)


def report(models=("vit_s",), bs=32, img=224, rounds=12, clients=5, out_csv=None,
           reps=12) -> pd.DataFrame:
    rows = []
    for name in models:
        m = build_model(name, pretrained=False)
        cfg = vit_config(m, img)
        a = analytic_vit(**cfg) if cfg else {}
        r = dict(model=name, **measure(m, bs, img, reps))
        r.update(train_memory(m, bs, img))
        r.update({k: v for k, v in a.items()})
        r.update(federated_cost(r["params"], rounds, clients))
        rows.append(r)
        del m
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df


def empirical_scaling(model_name="vit_s", sizes=(112, 160, 224, 288), bs=8,
                      out_csv=None, reps=6) -> pd.DataFrame:
    """Latency vs token count - shows the measured exponent against the O(N^2) bound."""
    rows = []
    for s in sizes:
        try:
            import timm
            from .models import BACKBONES
            m = timm.create_model(BACKBONES.get(model_name, model_name), pretrained=False,
                                  num_classes=2, img_size=s).to(DEVICE)
        except Exception:
            continue
        r = measure(m, bs, s, reps)
        rows.append(dict(img_size=s, tokens=(s // 16) ** 2 + 1, **r))
        del m
    df = pd.DataFrame(rows)
    if len(df) > 2:
        # fit latency ~ tokens^k
        k = np.polyfit(np.log(df.tokens), np.log(df.latency_ms_mean), 1)[0]
        df["fitted_exponent_vs_tokens"] = k
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df
