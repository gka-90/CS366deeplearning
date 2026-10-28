"""Visualization utilities: colorized masks, overlays, prediction grids, learning curves.

    # 12+ qualitative comparisons (input | GT | pred | GT overlay | pred overlay)
    python src/viz.py samples --ckpt outputs/checkpoints/<run>/best_weights.pt --n 12

    # learning curves from the CSV logs of one or more runs
    python src/viz.py curves --runs baseline strong_aug strong_aug_dice
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

from datamodule_oxpet import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

PALETTE = np.array([[230, 97, 1], [40, 50, 80], [253, 216, 53]], dtype=np.uint8)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """(H, W) class ids -> (H, W, 3) uint8 RGB."""
    return PALETTE[np.asarray(mask, dtype=np.int64)]


def denormalize(x: torch.Tensor) -> np.ndarray:
    """(3, H, W) normalized tensor -> (H, W, 3) uint8 RGB."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (x.detach().float().cpu() * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def overlay(img: np.ndarray, mask: np.ndarray, alpha: float = 0.45, skip_bg: bool = True) -> np.ndarray:
    """Blend a colorized mask over an RGB image (background left untinted by default)."""
    color = colorize_mask(mask).astype(np.float32)
    out = img.astype(np.float32).copy()
    sel = mask != 1 if skip_bg else np.ones_like(mask, dtype=bool)
    out[sel] = (1 - alpha) * out[sel] + alpha * color[sel]
    return out.astype(np.uint8)


def legend_handles():
    return [Patch(color=PALETTE[i] / 255, label=n) for i, n in enumerate(CLASS_NAMES)]


def save_comparison_grid(images, gts, preds, path, titles=None):
    """One row per sample: input | GT | prediction | GT overlay | prediction overlay."""
    n = len(images)
    cols = ["Input", "GT mask", "Prediction", "GT overlay", "Pred overlay"]
    fig, axes = plt.subplots(n, 5, figsize=(13, 2.6 * n))
    axes = np.atleast_2d(axes)
    for i in range(n):
        panels = [images[i], colorize_mask(gts[i]), colorize_mask(preds[i]),
                  overlay(images[i], gts[i]), overlay(images[i], preds[i])]
        for j, p in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(p)
            ax.set_xticks([]), ax.set_yticks([])
            if i == 0:
                ax.set_title(cols[j], fontsize=11)
        if titles:
            axes[i, 0].set_ylabel(titles[i], fontsize=8)
    fig.legend(handles=legend_handles(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_single_overlay(img, gt, pred, path):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    for ax, p, t in zip(axes, [img, overlay(img, gt), overlay(img, pred)], ["Input", "GT overlay", "Pred overlay"]):
        ax.imshow(p), ax.set_title(t), ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_compact_grid(images, gts, preds, path, titles=None, per_row=2):
    """All samples in one figure: per sample input | GT | prediction | prediction overlay."""
    panels, names = 4, ["Input", "GT", "Pred", "Overlay"]
    n_rows = (len(images) + per_row - 1) // per_row
    fig, axes = plt.subplots(n_rows, panels * per_row, figsize=(1.6 * panels * per_row, 1.65 * n_rows))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    for k in range(len(images)):
        r, c0 = divmod(k, per_row)
        row = [images[k], colorize_mask(gts[k]), colorize_mask(preds[k]), overlay(images[k], preds[k])]
        for j, p in enumerate(row):
            ax = axes[r, c0 * panels + j]
            ax.imshow(p)
            if r == 0:
                ax.set_title(names[j], fontsize=9)
        if titles:
            axes[r, c0 * panels].text(-0.06, 0.5, titles[k], transform=axes[r, c0 * panels].transAxes,
                                      rotation=90, ha="right", va="center", fontsize=6.5)
    fig.legend(handles=legend_handles(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), fontsize=9)
    fig.tight_layout(rect=(0.01, 0, 1, 0.965), w_pad=0.2, h_pad=0.3)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cmd_samples(args):
    from datamodule_oxpet import OxfordPetDataModule
    from model_unetpp import scores_from_confmat
    from train import load_model

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_model(args.ckpt).to(device)
    dm = OxfordPetDataModule(data_dir=args.data_dir, img_size=args.img_size, aug="none")
    dm.setup()
    ds = dm.test_ds if args.split == "test" else dm.val_ds

    rng = np.random.default_rng(args.seed)
    idx = sorted(rng.choice(len(ds), size=args.n, replace=False).tolist())
    images, gts, preds, titles = [], [], [], []
    for i in idx:
        x, y = ds[i]
        probs = model.predict_probs(x[None].to(device), tta=args.tta)
        pred = probs.argmax(1)[0].cpu().numpy()
        cm = torch.bincount(y.flatten() * 3 + torch.from_numpy(pred).flatten(), minlength=9).view(3, 3)
        miou = scores_from_confmat(cm)["miou"].item()
        images.append(denormalize(x)), gts.append(y.numpy()), preds.append(pred)
        titles.append(f"{ds.names[i]}\nmIoU {miou:.3f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.tag}_" if args.tag else ""
    half = (len(idx) + 1) // 2
    save_comparison_grid(images[:half], gts[:half], preds[:half], out_dir / f"{tag}pred_grid_1.png", titles[:half])
    save_comparison_grid(images[half:], gts[half:], preds[half:], out_dir / f"{tag}pred_grid_2.png", titles[half:])
    save_compact_grid(images, gts, preds, out_dir / f"{tag}pred_grid_compact.png", titles)
    for k in range(min(4, len(idx))):
        save_single_overlay(images[k], gts[k], preds[k], out_dir / f"{tag}overlay_{titles[k].split()[0]}.png")
    print(f"Saved {len(idx)} comparisons to {out_dir}")


def _read_metrics(run: str):
    import pandas as pd

    versions = sorted((OUT / "logs" / run).glob("version_*"), key=lambda p: int(p.name.split("_")[1]))
    if not versions:
        raise FileNotFoundError(f"no CSV logs for run '{run}'")
    df = pd.read_csv(versions[-1] / "metrics.csv")
    return df.groupby("epoch").mean(numeric_only=True).reset_index()


def cmd_curves(args):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for run in args.runs:
        df = _read_metrics(run)
        for key, ax, style in [("train/loss_epoch", axes[0], "--"), ("val/loss", axes[0], "-"),
                               ("train/mIoU", axes[1], "--"), ("val/mIoU", axes[1], "-"),
                               ("val/mDice", axes[2], "-")]:
            if key in df:
                d = df[["epoch", key]].dropna()
                ax.plot(d["epoch"], d[key], style, label=f"{run} {key.split('/')[0]}")
    for ax, t in zip(axes, ["Loss", "mIoU", "Validation mDice"]):
        ax.set_title(t), ax.set_xlabel("epoch"), ax.grid(alpha=0.3), ax.legend(fontsize=7)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"Saved {args.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("samples")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--data_dir", default="~/data")
    s.add_argument("--img_size", type=int, default=256)
    s.add_argument("--split", choices=["val", "test"], default="test")
    s.add_argument("--n", type=int, default=12)
    s.add_argument("--tta", action="store_true")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--tag", default="")
    s.add_argument("--out_dir", default=str(OUT / "samples"))

    c = sub.add_parser("curves")
    c.add_argument("--runs", nargs="+", required=True)
    c.add_argument("--out", default=str(OUT / "samples" / "learning_curves.png"))

    a = p.parse_args()
    {"samples": cmd_samples, "curves": cmd_curves}[a.cmd](a)


if __name__ == "__main__":
    main()
