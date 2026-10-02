# Toxic Mushroom Classification with Vision Transformers, Federated Learning and Explainable AI

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style](https://img.shields.io/badge/reproducible-seeded-success.svg)](#reproducibility)

A reproducible research pipeline for binary **edible vs toxic** mushroom sporocarp
classification. It fine-tunes a Vision Transformer, repeats the experiment under
simulated **federated learning** with non-IID clients, explains the predictions with
four attribution methods, and reports every result with confidence intervals and
paired significance tests.

Each stage writes CSV artefacts. Later stages read only those artefacts, so any table
or figure in the manuscript can be regenerated on a laptop without a GPU — and a
reviewer can recompute any number from the released predictions.

---

## Why this repository is structured the way it is

Three design choices do most of the work, and each one answers an objection that
image-based mushroom classification papers usually fail to answer.

**1. Near-duplicate leakage control.** Web-scraped fungal corpora contain re-uploads of
the same photograph at different resolutions. A random train/test split puts those
copies on both sides and inflates test accuracy. This pipeline computes a 64-bit DCT
perceptual hash per image, clusters at Hamming distance ≤ 5 by single linkage, and
splits **by cluster**, stratified by class. The split report contains an explicit
leakage check that must read zero.

**2. The error that matters is named and measured.** A toxic specimen classified as
edible is not symmetric with the reverse. The positive class is toxic, the false
negative rate is reported as a first-class metric, and the decision threshold is
selected from a required toxic-recall level rather than from argmax accuracy.

**3. Explanations are scored, not displayed.** Saliency maps are evaluated by deletion
and insertion AUC and submitted to the cascading model-randomisation sanity check of
Adebayo et al. (2018). A map that does not change when the network's weights are
randomised is not explaining the network, and the pipeline will say so.

---

## Installation

```bash
git clone https://github.com/<your-username>/toxic-mushroom-vit-fl-xai.git
cd toxic-mushroom-vit-fl-xai
pip install -r requirements.txt
```

Tested on Python 3.10–3.13, PyTorch 2.x, CUDA 11.8+ and CPU. A single mid-range GPU
(Colab/Kaggle T4) is enough for the full study.

## Dataset layout

`prep` expects class folders directly under `--data`. Folder names are mapped to binary
labels in `mushroom/data.py::FOLDER_MAP`; edit that dictionary for a different corpus.

```
data/
├── edible mushroom sporocarp/
├── edible sporocarp/
├── poisonous mushroom sporocarp/
└── poisonous sporocarp/
```

## Quickstart

```bash
# 1. Audit images, remove near-duplicate redundancy, build leakage-safe splits
python -m mushroom.pipeline prep --data ./data --out results

# 2. Centralised + federated training across seeds
python -m mushroom.pipeline run --out results \
    --model vit_s --epochs 10 --seeds 42,43,44 \
    --clients 5 --rounds 12 --alphas 0.1,0.5,100 --fedprox

# 3. Metrics, confidence intervals, significance tests, calibration
python -m mushroom.pipeline analyze --out results

# 4. Attribution maps and faithfulness scores
python -m mushroom.pipeline xai --out results --n_vis 12

# 5. Time and space complexity: asymptotic + measured
python -m mushroom.pipeline complexity --out results

# 6. All figures
python -m mushroom.pipeline figures --out results
```

`all` chains every stage:

```bash
python -m mushroom.pipeline all --data ./data --out results --model vit_s --epochs 10
```

Approximate budget on a T4: 6–8 min per centralised run, 10–14 min per federated
configuration.

---

## Repository structure

```
mushroom/
├── data.py         Image audit, perceptual-hash deduplication, group-stratified
│                   splits, preprocessing transforms, class weighting
├── models.py       Backbones, centralised fine-tuning, FedAvg / FedProx,
│                   Dirichlet non-IID partitioning, prediction
├── metrics.py      Every scalar metric, bootstrap CIs, McNemar, paired bootstrap,
│                   calibration (ECE/MCE/Brier), operating-point selection
├── xai.py          Grad-CAM, Grad-CAM++, Eigen-CAM, attention rollout,
│                   deletion/insertion faithfulness, randomisation sanity check
├── complexity.py   Asymptotic derivation + measured latency, memory, FLOPs,
│                   federated communication cost
├── figures.py      All manuscript figures, generated from CSV artefacts only
├── viz.py          Figure style: palette, typography, geometry, export
└── pipeline.py     CLI orchestrating prep → run → analyze → xai → complexity → figures
```

### Supported backbones

Select with `--model`. Attention rollout is skipped automatically for the CNNs.

| Key | Model | Params |
|---|---|---|
| `vit_t` | ViT-Tiny/16 | 5.5 M |
| `vit_s` | ViT-Small/16 (default) | 21.7 M |
| `vit_b` | ViT-Base/16 | 86 M |
| `deit_s` | DeiT-III Small/16 | 22 M |
| `swin_t` | Swin-Tiny | 28 M |
| `resnet50` | ResNet-50 | 25.6 M |
| `effb0` | EfficientNet-B0 | 5.3 M |

---

## Method

### Preprocessing

Deliberately minimal, so the contribution is attributable to the model rather than to
an augmentation stack: EXIF transpose, RGB conversion, `RandomResizedCrop(224,
scale=0.65–1.0)`, horizontal flip (mushroom toxicity has no chirality), mild colour
jitter so the model cannot key on one collector's white balance, and ImageNet
normalisation. Evaluation applies deterministic resize and centre crop only.

### Centralised training

ViT-S/16 pretrained on ImageNet-21k and fine-tuned on ImageNet-1k, AdamW, cosine
schedule with 10% warmup, class-weighted cross-entropy with 0.05 label smoothing,
mixed precision, gradient clipping at 1.0. Model selection on **validation balanced
accuracy**; the test split is evaluated exactly once.

### Federated learning

FedAvg (McMahan et al., 2017) with an optional FedProx proximal term (Li et al., 2020),
enabled by `--fedprox --mu 0.01`. Statistical heterogeneity is induced by a
Dirichlet(α) label partition, the standard non-IID benchmark: α = 0.1 is strongly
skewed, α = 100 is effectively IID. The pipeline logs per-round validation accuracy,
per-client label composition with KL divergence from the global mix, and cumulative
communication cost in MB.

### Explainability

Grad-CAM, Grad-CAM++, Eigen-CAM and attention rollout are implemented directly rather
than through a third-party CAM package, so the ViT token-grid reshape is explicit and
auditable and the same code path serves CNN baselines. Every map is scored by deletion
AUC (lower is better) and insertion AUC against a blurred baseline (higher is better).

### Complexity

Both derived and measured. For one ViT forward pass with N tokens, width D, depth L,
heads H:

| Quantity | Complexity |
|---|---|
| Time | O(L(N²D + ND²)) |
| Activation memory | O(L(HN² + ND)) |
| Parameters | O(LD²) |
| Federated communication | O(R·K·‖θ‖), independent of dataset size |

The analytic FLOP count is validated against the literature: ViT-S/16 at 224 px yields
4.54 GMACs against the ≈4.6 GMACs reported in the original ViT papers. Note that at
224 px the ND² term dominates the quadratic N²D term by roughly D/N ≈ 2, so measured
latency scales close to N¹·² rather than N² — worth stating explicitly instead of
repeating that "attention is quadratic".

---

## Outputs

`results/tables/` — tidy CSV, one row per observation:

| File | Contents |
|---|---|
| `table1_dataset_audit.csv` | Corpus characteristics, duplicate and label-noise counts |
| `table2_splits.csv` | Split sizes, class balance, leakage check |
| `table3_metrics_per_run.csv` | Every metric per run with bootstrap 95% CI and SE |
| `table4_metrics_mean_sd.csv` | Mean ± sd across seeds with t-based CI |
| `table5_significance.csv` | McNemar and paired bootstrap deltas |
| `table6_reliability_bins.csv` | Calibration bins |
| `table7_operating_points.csv` | Thresholds at 90/95/99% required toxic recall |
| `table8_headline.tex` | LaTeX table, paste directly into the manuscript |
| `table9/10_xai_*.csv` | Per-image and summarised faithfulness |
| `table11_xai_sanity_check.csv` | Model-randomisation test |
| `table12_heterogeneity_sweep.csv` | Accuracy against Dirichlet α |
| `table13_complexity.csv` | Parameters, FLOPs, latency, memory, communication |
| `table14_scaling.csv` | Measured latency against token count |
| `table15_wallclock.csv` | Observed training time per paradigm |

Metrics reported: accuracy, **error rate**, balanced accuracy, balanced error rate,
precision, recall, F1 (toxic and macro), specificity, NPV, **FNR (toxic classified as
edible)**, FPR, MCC, Cohen's κ, Youden's J, ROC-AUC, PR-AUC, Brier, NLL, ECE and MCE.

`results/figures/` — vector PDF for LaTeX plus 600-dpi PNG, covering dataset
composition, convergence, ROC/PR with bootstrap CI bands, confusion matrices,
reliability, the heterogeneity sweep, attribution grids, failure cases, faithfulness,
complexity scaling and the safety operating point.

`results/runs/` — per-sample predictions (`path, y_true, p_toxic`), training histories,
client partitions and checkpoints. The prediction files are the ground truth for every
statistic; release them with the paper.

---

## Reproducibility

Every run is seeded (`--seeds 42,43,44`), cuDNN is set deterministic, and the run index
records parameters, wall-clock time and communication cost per run. Statistics are
computed from released per-sample predictions, so results can be re-derived
independently of the training environment.

Report both uncertainty estimates — they answer different questions. Mean ± sd across
seeds captures training variance; the bootstrap CI on the pooled test set captures
sampling variance of the test split.

---

## Reporting guidance

- Lead with **balanced accuracy and toxic recall**. On a corpus that is 1.88:1
  toxic-heavy, a trivial always-toxic classifier already scores 65% accuracy.
- Report the false-negative rate explicitly and justify the threshold from it.
  Accuracy-optimal is not safety-optimal.
- Use McNemar for centralised versus federated. The models share a test set, so an
  unpaired test is the wrong instrument.
- State the leakage control in the Methods section. It is the cheapest credibility
  gain available in this literature.

---

## Limitations

Species-level identity is unavailable in this corpus, so the binary label inherits
whatever labelling policy the source collection applied, and edibility cannot be
verified per image. Web-scraped fungal datasets also contain stock-agency watermarks
and botanical illustrations alongside field photographs; a classifier can key on the
source signature rather than on the specimen, and this should be audited and declared.

Most importantly: edibility is not a property that can be read reliably from a
photograph of a sporocarp. Correct identification depends on spore prints, habitat,
substrate, odour and microscopy.

> **Safety notice.** This software is a research artefact and an assistive screening
> tool. It is not a foraging safety device. Do not use its output to decide whether a
> wild mushroom is safe to eat.

---

## Citation

```bibtex
@software{rahman_toxic_mushroom_vit_fl_xai,
  author  = {Rahman, M. W.},
  title   = {Toxic Mushroom Classification with Vision Transformers,
             Federated Learning and Explainable AI},
  year    = {2026},
  url     = {https://github.com/<your-username>/toxic-mushroom-vit-fl-xai},
  note    = {WRESLab, Uttara University}
}
```

## References

- Dosovitskiy et al. (2021). *An Image is Worth 16x16 Words.* ICLR.
- McMahan et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS.
- Li et al. (2020). *Federated Optimization in Heterogeneous Networks.* MLSys.
- Selvaraju et al. (2017). *Grad-CAM.* ICCV.
- Chattopadhay et al. (2018). *Grad-CAM++.* WACV.
- Muhammad & Yeasin (2020). *Eigen-CAM.* IJCNN.
- Abnar & Zuidema (2020). *Quantifying Attention Flow in Transformers.* ACL.
- Adebayo et al. (2018). *Sanity Checks for Saliency Maps.* NeurIPS.
- Guo et al. (2017). *On Calibration of Modern Neural Networks.* ICML.

## License

MIT — see [LICENSE](LICENSE).
