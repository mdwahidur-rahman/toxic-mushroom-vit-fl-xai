"""Dataset construction, integrity audit, near-duplicate control, leakage-safe splits.

Design note (this is the part reviewers attack first):
Web-scraped mushroom corpora contain re-uploads of the same photograph at
different resolutions. A random split puts near-duplicates on both sides and
inflates test accuracy. We therefore cluster images by perceptual hash
(Hamming <= PHASH_THR on a 64-bit pHash, single-linkage) and split by CLUSTER,
never by image, so no visual near-duplicate crosses the train/test boundary.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

PHASH_THR = 5                      # Hamming radius for "near-duplicate"
CLASSES = ["edible", "toxic"]      # index 0, 1 -> toxic is the positive (dangerous) class
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

# folder name -> binary label (this corpus ships four folders, two per class)
FOLDER_MAP = {
    "edible mushroom sporocarp": "edible",
    "edible sporocarp": "edible",
    "poisonous mushroom sporocarp": "toxic",
    "poisonous sporocarp": "toxic",
}


# --------------------------------------------------------------------- audit
def _phash_bits(img: Image.Image) -> np.ndarray:
    """64-bit perceptual hash (DCT) as a bit vector - no external dependency."""
    from scipy.fftpack import dct
    g = np.asarray(img.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    d = dct(dct(g, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
    return (d > np.median(d[1:, 1:])).flatten().astype(np.uint8)


def build_manifest(root: str | Path, out_csv: str | Path) -> pd.DataFrame:
    """Verify every image, record geometry, compute pHash. Drops unreadable files."""
    root = Path(root)
    rows, rejected = [], []
    for folder, label in FOLDER_MAP.items():
        d = root / folder
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            try:
                Image.open(f).verify()                       # structural check
                im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
                rows.append(dict(path=str(f), folder=folder, label=label,
                                 y=CLASSES.index(label), width=im.size[0], height=im.size[1],
                                 megapixels=im.size[0] * im.size[1] / 1e6,
                                 aspect=im.size[0] / im.size[1],
                                 phash="".join(map(str, _phash_bits(im)))))
            except Exception as e:                            # corrupt / truncated
                rejected.append(dict(path=str(f), error=repr(e)[:200]))
    df = pd.DataFrame(rows)
    df = _cluster_near_duplicates(df)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    if rejected:
        pd.DataFrame(rejected).to_csv(Path(out_csv).with_name("rejected_images.csv"), index=False)
    return df


def _cluster_near_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Single-linkage clustering of pHash bit vectors within Hamming <= PHASH_THR."""
    B = np.array([[int(c) for c in h] for h in df.phash], dtype=np.uint8)
    n = len(B)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(0, n, 256):                                # chunked XOR, memory-safe
        blk = B[i:i + 256]
        dist = (blk[:, None, :] ^ B[None, :, :]).sum(-1)
        for r in range(blk.shape[0]):
            for j in np.where(dist[r] <= PHASH_THR)[0]:
                if j > i + r:
                    ra, rb = find(i + r), find(int(j))
                    if ra != rb:
                        parent[rb] = ra
    df = df.copy()
    df["cluster"] = [find(i) for i in range(n)]
    # keep one representative per cluster for training; the rest are flagged, not deleted
    df["is_representative"] = ~df.duplicated("cluster", keep="first")
    return df


def audit_report(df: pd.DataFrame, out_csv: str | Path) -> pd.DataFrame:
    """Dataset characteristics table -> straight into the manuscript."""
    g = df.groupby("cluster")
    rec = [
        ("Images (readable)", len(df)),
        ("Classes", df.label.nunique()),
        ("Source folders", df.folder.nunique()),
        ("Edible images", int((df.label == "edible").sum())),
        ("Toxic images", int((df.label == "toxic").sum())),
        ("Class imbalance ratio (toxic:edible)", round((df.label == "toxic").sum() / max((df.label == "edible").sum(), 1), 3)),
        ("Near-duplicate clusters (size>1)", int((g.size() > 1).sum())),
        ("Images inside such clusters", int(g.size()[g.size() > 1].sum())),
        ("Redundant images removed from training pool", int(len(df) - df.cluster.nunique())),
        ("Clusters spanning BOTH labels (label noise)", int((g.label.nunique() > 1).sum())),
        ("Unique visual instances (clusters)", int(df.cluster.nunique())),
        ("Median resolution (px)", f"{int(df.width.median())}x{int(df.height.median())}"),
        ("Min resolution (px)", f"{int(df.width.min())}x{int(df.height.min())}"),
        ("Max resolution (px)", f"{int(df.width.max())}x{int(df.height.max())}"),
        ("Median aspect ratio", round(float(df.aspect.median()), 3)),
    ]
    out = pd.DataFrame(rec, columns=["property", "value"])
    out.to_csv(out_csv, index=False)
    return out


# --------------------------------------------------------------------- splits
def group_stratified_split(df: pd.DataFrame, val=0.15, test=0.15, seed=42) -> pd.DataFrame:
    """Stratify by label while keeping every near-duplicate cluster intact."""
    rng = np.random.RandomState(seed)
    cl = df.groupby("cluster").agg(label=("label", "first"), n=("path", "size")).reset_index()
    assign = {}
    for lab in sorted(cl.label.unique()):                     # stratify per class
        sub = cl[cl.label == lab].sample(frac=1.0, random_state=rng.randint(1 << 31))
        cum = np.cumsum(sub.n.values) / sub.n.values.sum()
        for c, u in zip(sub.cluster.values, cum):
            assign[c] = "test" if u <= test else ("val" if u <= test + val else "train")
    out = df.copy()
    out["split"] = out.cluster.map(assign)
    return out


def split_report(df: pd.DataFrame, out_csv: str | Path) -> pd.DataFrame:
    rows = []
    for s in ["train", "val", "test"]:
        d = df[df.split == s]
        rows.append(dict(split=s, images=len(d), clusters=d.cluster.nunique(),
                         edible=int((d.label == "edible").sum()),
                         toxic=int((d.label == "toxic").sum()),
                         toxic_fraction=round(float((d.label == "toxic").mean()), 4)))
    # leakage proof: no cluster may appear in two splits
    leak = df.groupby("cluster").split.nunique()
    rows.append(dict(split="LEAKAGE_CHECK_clusters_in_multiple_splits",
                     images=int((leak > 1).sum()), clusters=0, edible=0, toxic=0, toxic_fraction=0.0))
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    return out


# --------------------------------------------------------------------- torch
class MushroomDataset(Dataset):
    """Reads straight from the manifest so splits/filters stay explicit and auditable."""

    def __init__(self, frame: pd.DataFrame, train: bool, img_size: int = 224):
        self.rows = frame.reset_index(drop=True)
        self.tf = build_transform(train, img_size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        img = ImageOps.exif_transpose(Image.open(r.path)).convert("RGB")
        return self.tf(img), int(r.y)


def build_transform(train: bool, img_size: int = 224):
    """Deliberately minimal preprocessing: geometry normalisation + photometric jitter.

    Train: scale/translation invariance (RandomResizedCrop), reflection symmetry
    (HFlip - mushroom toxicity has no chirality), and mild colour jitter so the
    model cannot key on a single collector's camera white balance.
    Eval: deterministic resize + centre crop only.
    """
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.65, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor(), norm,
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)), transforms.CenterCrop(img_size),
        transforms.ToTensor(), norm,
    ])


def denormalize(t: torch.Tensor) -> np.ndarray:
    """CHW tensor -> HWC float RGB in [0,1] for overlaying CAMs."""
    m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    s = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t.cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


def loaders(df: pd.DataFrame, bs=32, img_size=224, workers=2, representatives_only=True):
    tr = df[df.split == "train"]
    if representatives_only:
        tr = tr[tr.is_representative]                        # drop redundant copies from training
    mk = lambda d, train, sh: DataLoader(
        MushroomDataset(d, train, img_size), batch_size=bs, shuffle=sh,
        num_workers=workers, pin_memory=torch.cuda.is_available(), drop_last=False)
    return (mk(tr, True, True),
            mk(df[df.split == "val"], False, False),
            mk(df[df.split == "test"], False, False))


def class_weights(df: pd.DataFrame) -> torch.Tensor:
    """Inverse-frequency weights; the toxic class must not be under-weighted."""
    tr = df[(df.split == "train") & df.is_representative]
    c = np.bincount(tr.y.values, minlength=2).astype(float)
    return torch.tensor(len(tr) / (2.0 * np.maximum(c, 1)), dtype=torch.float)
