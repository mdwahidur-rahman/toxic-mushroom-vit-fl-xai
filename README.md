# Toxic mushroom classification — ViT + Federated Learning + XAI

Reproducible pipeline for a short IEEE Access / Elsevier submission. Every table
and figure in the manuscript is generated from CSV artefacts, so a reviewer can
recompute any number without a GPU.

## Dataset

Binary **edible vs toxic** sporocarp images, four source folders mapped to two
classes. Audited in `prep`: 3,401 readable images, 1,181 edible / 2,220 toxic
(1.88:1), median native resolution 262x194 px.

## Why the splits are defensible

Web-scraped mushroom corpora contain re-uploads of the same photograph. A random
split leaks them across train/test and inflates accuracy. This pipeline computes
a 64-bit DCT perceptual hash per image, clusters at Hamming distance <= 5
(single linkage), and splits **by cluster**, stratified by class. The split
report includes an explicit leakage check that must read 0.

On this corpus: 93 near-duplicate clusters covering 189 images; 96 redundant
copies are excluded from the training pool; **0 clusters span both labels**
(so no label noise from duplicates) and **0 clusters span two splits**.

## Install

```bash
pip install torch torchvision timm scikit-learn scipy pandas matplotlib
```

## Run (Colab / Kaggle T4)

```bash
python -m pkg.pipeline prep    --data "/content/data" --out results
python -m pkg.pipeline run     --out results --model vit_s --epochs 10 \
                               --seeds 42,43,44 --clients 5 --rounds 12 \
                               --alphas 0.1,0.5,100 --fedprox
python -m pkg.pipeline analyze --out results
python -m pkg.pipeline xai     --out results --n_vis 12
python -m pkg.pipeline figures --out results
```

`prep` expects the four source folders directly under `--data`.
`all` chains every stage. Budget on a T4: roughly 6–8 min per centralised run
and 10–14 min per federated configuration.

Backbones via `--model`: `vit_s` (default), `vit_b`, `deit_s`, `swin_t`,
`resnet50`, `effb0`. The CNNs give the ablation baseline; attention rollout is
skipped for them automatically.

## Method summary

**Preprocessing** (deliberately minimal, so the contribution is the model, not
the augmentation stack): EXIF transpose, RGB, `RandomResizedCrop(224, 0.65–1.0)`,
horizontal flip, mild colour jitter, ImageNet normalisation. Evaluation uses
deterministic resize + centre crop only.

**Centralised**: ViT-S/16 (ImageNet-21k → 1k), AdamW, cosine schedule with 10%
warmup, class-weighted cross-entropy with 0.05 label smoothing, AMP, gradient
clipping. Selection on validation balanced accuracy.

**Federated**: FedAvg with optional FedProx proximal term. Statistical
heterogeneity via a Dirichlet(α) label partition — α=0.1 strongly skewed,
α=100 effectively IID. Reports per-round validation accuracy and cumulative
communication cost in MB.

**XAI**: Grad-CAM, Grad-CAM++, Eigen-CAM and attention rollout, implemented
directly so the ViT token-grid reshape is explicit. Each map is scored by
deletion AUC (lower better) and insertion AUC (higher better), and submitted to
the cascading model-randomisation sanity check of Adebayo et al. (2018).

## Outputs

`results/tables/`

| file | contents |
|---|---|
| `table1_dataset_audit.csv` | corpus characteristics |
| `table2_splits.csv` | split sizes + leakage check |
| `table3_metrics_per_run.csv` | every metric per run with bootstrap 95% CI and SE |
| `table4_metrics_mean_sd.csv` | mean ± sd across seeds, with t-based CI |
| `table5_significance.csv` | McNemar + paired bootstrap deltas, centralised vs federated |
| `table6_reliability_bins.csv` | calibration bins |
| `table7_operating_points.csv` | thresholds at 90/95/99% required toxic recall |
| `table8_headline.tex` | LaTeX table, paste straight into the manuscript |
| `table9/10_xai_*.csv` | per-image and summarised faithfulness |
| `table11_xai_sanity_check.csv` | model-randomisation test |
| `table12_heterogeneity_sweep.csv` | accuracy vs Dirichlet α |

Metrics include accuracy, **error rate**, balanced accuracy, balanced error
rate, precision/recall/F1, specificity, NPV, **FNR (toxic classified as
edible)**, FPR, MCC, Cohen's κ, Youden's J, ROC-AUC, PR-AUC, Brier, NLL, ECE
and MCE.

`results/figures/` — vector PDF (for LaTeX) + 600 dpi PNG:
dataset composition, convergence, ROC/PR with bootstrap CI bands, confusion
matrices, reliability, heterogeneity sweep, Grad-CAM grid, failure-case grid,
XAI faithfulness, and the safety operating point.

## Reporting notes for the manuscript

- Lead with **balanced accuracy and toxic recall**, not raw accuracy: the corpus
  is 1.88:1 toxic-heavy, so a trivial always-toxic classifier scores 65%.
- Report the **false-negative rate** explicitly and frame the threshold choice
  around it (`table7`). Accuracy-optimal is not safety-optimal.
- Quote mean ± sd over seeds **and** the bootstrap CI on the pooled test set;
  they answer different questions (training variance vs sampling variance).
- Use McNemar for centralised vs federated — the models share a test set, so an
  unpaired test is wrong.
- State the leakage control in the Methods. It is the cheapest credibility win
  available in this literature.

## Limitations to state in the paper

Genus- and species-level identity is not available in this corpus, so the
binary label inherits whatever labelling policy the source collection used, and
edibility is not a property that can be read reliably from a photograph of a
sporocarp alone. The system is an assistive screening tool and must not be
presented as a foraging safety device.
