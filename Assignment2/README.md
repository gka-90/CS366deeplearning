# Assignment 2 — UNet++ Semantic Segmentation on Oxford-IIIT Pet (PyTorch Lightning)

3-class trimap segmentation (**pet / background / border**) with UNet++ (ResNet-34 encoder,
ImageNet weights) trained with PyTorch Lightning.

## Repository layout

```
oxpet_download_and_viz_fixed.py  # provided exploration script
src/
  datamodule_oxpet.py   # LightningDataModule: official splits, 80/20 train/val, trimap -> {0,1,2}, transforms
  model_unetpp.py       # LightningModule: UNet++, CE / CE+Dice loss, per-class IoU & Dice, flip-TTA
  train.py              # CLI: `fit` (train + test) and `test` (evaluate a checkpoint)
  viz.py                # CLI: `samples` (prediction grids / overlays) and `curves` (learning curves)
outputs/
  checkpoints/<run>/    # best checkpoint (by val mIoU), best_weights.pt, test_metrics.json
  logs/<run>/           # CSV logs (+ offline W&B runs under logs/wandb)
  samples/              # figures
```

## 1. Setup

```bash
conda create -n oxpet python=3.11 -y && conda activate oxpet
pip install -r requirements.txt
wandb login            
```

Trained on the Hamilton CS HPC cluster (Slurm), one NVIDIA RTX 5070 Ti (16 GB) per run,
torch 2.15 (CUDA 13.2), pytorch-lightning 2.6, segmentation-models-pytorch 0.6.
Each run takes about 35 min of GPU time (≈1.8 GPU-hours for all three, well under the 10 h limit).

On a Slurm cluster, wrap a command like this:

```bash
sbatch --gres=gpu:1 --cpus-per-task=8 --mem=24G --time=04:00:00 \
  --wrap="bash -c 'conda activate oxpet && python src/train.py fit --run_name baseline ...'"
```

## 2. Data exploration

```bash
python oxpet_download_and_viz_fixed.py --root ~/data --split trainval --classes trimap --n 12 --resize 512 \
  --save-dir outputs/samples/exploration
```

This writes `oxpet_viz_grid.png` (image / trimap mask / overlay for 12 samples) and 4 single overlays.

The dataset lives in `~/data/oxford-iiit-pet/` (`train.py` downloads it there if missing).

## 3. Training (baseline + two improvements)

All runs: 384×384 input, batch size 16, 40 epochs, AdamW (lr 3e-4, wd 1e-4), 1-epoch linear warmup
then cosine decay, 16-bit mixed precision (AMP), best checkpoint chosen by **val mIoU**.
After training, each run is evaluated on the official **test** split, with and without flip-TTA.

```bash
# Baseline: resize + horizontal flip, cross-entropy
python src/train.py fit --run_name baseline        --aug light  --loss ce      --epochs 40 --img_size 384 --batch_size 16 --num_workers 8 --wandb

# Improvement 1 (augmentation): random resized crop, rotation, color jitter
python src/train.py fit --run_name strong_aug      --aug strong --loss ce      --epochs 40 --img_size 384 --batch_size 16 --num_workers 8 --wandb

# Improvement 2 (loss): strong augmentation + cross-entropy + Dice loss
python src/train.py fit --run_name strong_aug_dice --aug strong --loss ce_dice --epochs 40 --img_size 384 --batch_size 16 --num_workers 8 --wandb
```

**W&B dashboard:** https://forge.coreweave.com/wandb/gkobea-hamilton-college/oxpet-unetpp

If W&B is not logged in, run with `WANDB_MODE=offline` and upload later with
`wandb sync outputs/logs/wandb/offline-run-*`.

## 4. Evaluation

```bash
# Per-class + mean IoU/Dice on the test split, with and without horizontal-flip TTA
python src/train.py test --ckpt outputs/checkpoints/strong_aug_dice/best_weights.pt --img_size 384 --tta
```

Results are written to `outputs/checkpoints/<run>/test_metrics.json`.

## 5. Figures

```bash
# 12 test images: input | GT | prediction | GT overlay | prediction overlay
python src/viz.py samples --ckpt outputs/checkpoints/strong_aug_dice/best_weights.pt --n 12 --img_size 384 --tta

# Learning curves for all runs
python src/viz.py curves --runs baseline strong_aug strong_aug_dice
```

## Notes

- Trimap values 1/2/3 are mapped to classes 0 (pet), 1 (background), 2 (border).
- Masks are always resized, cropped and rotated with **NEAREST** interpolation; images with bilinear.
- mIoU and Dice are computed from a confusion matrix accumulated over all pixels of the split.
- `.ckpt` files are not committed (>100 MB); `best_weights.pt` (fp16 weights, ~50 MB) is.
  Both can be passed to `--ckpt`.

## References

- Zhou et al., *UNet++: A Nested U-Net Architecture for Medical Image Segmentation*, DLMIA 2018.
- Parkhi et al., *Cats and Dogs*, CVPR 2012 (Oxford-IIIT Pet dataset).
- Iakubovskii, *Segmentation Models PyTorch*, https://github.com/qubvel-org/segmentation_models.pytorch
- PyTorch Lightning documentation, https://lightning.ai/docs/pytorch/stable/
- Milletari et al., *V-Net* (Dice loss), 3DV 2016.
