"""Train / evaluate UNet++ on Oxford-IIIT Pet with PyTorch Lightning.

    python src/train.py fit  --run_name baseline ...
    python src/train.py test --ckpt outputs/checkpoints/baseline/best.ckpt --tta
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from datamodule_oxpet import CLASS_NAMES, OxfordPetDataModule
from model_unetpp import UNetPlusPlusModule

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"


def default_precision() -> str:

    if torch.cuda.is_available() or torch.backends.mps.is_available():
        return "16-mixed"
    return "bf16-mixed"


def add_common(p: argparse.ArgumentParser):
    p.add_argument("--data_dir", default="~/data", help="folder that contains oxford-iiit-pet/")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--precision", default=default_precision())
    p.add_argument("--seed", type=int, default=42)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="train a model")
    add_common(f)
    f.add_argument("--run_name", default="baseline")
    f.add_argument("--encoder", default="resnet34")
    f.add_argument("--aug", choices=["none", "light", "strong"], default="light")
    f.add_argument("--loss", choices=["ce", "ce_dice"], default="ce")
    f.add_argument("--epochs", type=int, default=20)
    f.add_argument("--lr", type=float, default=3e-4)
    f.add_argument("--weight_decay", type=float, default=1e-4)
    f.add_argument("--patience", type=int, default=0, help="early-stopping patience (0 = off)")
    f.add_argument("--max_time", default=None, help='wall-clock cap, e.g. "00:03:00:00"')
    f.add_argument("--wandb", action="store_true", help="also log to Weights & Biases")
    f.add_argument("--wandb_project", default="oxpet-unetpp")
    f.add_argument("--limit_batches", type=float, default=1.0, help="fraction of batches (smoke tests)")
    f.add_argument("--no_test", action="store_true", help="skip test-set evaluation after training")

    t = sub.add_parser("test", help="evaluate a checkpoint on the official test split")
    add_common(t)
    t.add_argument("--ckpt", required=True)
    t.add_argument("--tta", action="store_true", help="also evaluate with horizontal-flip TTA")
    t.add_argument("--split", choices=["val", "test"], default="test")
    t.add_argument("--limit_batches", type=float, default=1.0)
    return parser


def make_datamodule(args, aug="none"):
    return OxfordPetDataModule(
        data_dir=args.data_dir, img_size=args.img_size, batch_size=args.batch_size,
        aug=aug, num_workers=args.num_workers, seed=args.seed,
    )


def evaluate(model, dm, args, tta: bool, split: str = "test") -> dict:
    model.hparams.tta = tta
    trainer = pl.Trainer(accelerator="auto", devices=1, precision=args.precision, logger=False,
                         limit_test_batches=args.limit_batches, enable_model_summary=False)
    if split == "val":
        dm.setup()
        loader = dm.val_dataloader()
        trainer.test(model, dataloaders=loader, verbose=False)
    else:
        trainer.test(model, datamodule=dm, verbose=False)
    s = model.last_test_scores
    return {
        "split": split,
        "tta": tta,
        "mIoU": s["miou"],
        "mDice": s["mdice"],
        "pixel_acc": s["acc"],
        "IoU": dict(zip(CLASS_NAMES, s["iou"])),
        "Dice": dict(zip(CLASS_NAMES, s["dice"])),
    }


def print_scores(r: dict):
    tag = f"{r['split']}{' +TTA' if r['tta'] else ''}"
    print(f"\n[{tag}] mIoU={r['mIoU']:.4f}  mDice={r['mDice']:.4f}  pixel_acc={r['pixel_acc']:.4f}")
    for c in CLASS_NAMES:
        print(f"   {c:<11} IoU={r['IoU'][c]:.4f}  Dice={r['Dice'][c]:.4f}")


def export_weights(ckpt_path: str, out_path: Path):
    """Save a small weights-only file (fp16) next to the full Lightning checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = {k: v.half() if v.is_floating_point() else v for k, v in ckpt["state_dict"].items()}
    torch.save({"state_dict": state, "hyper_parameters": ckpt["hyper_parameters"]}, out_path)


def load_model(path: str) -> UNetPlusPlusModule:
    """Load either a Lightning .ckpt or the exported weights-only .pt."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hp = dict(ckpt["hyper_parameters"])
    hp["encoder_weights"] = None  
    model = UNetPlusPlusModule(**hp)
    model.load_state_dict({k: v.float() for k, v in ckpt["state_dict"].items()})
    return model.eval()


def cmd_fit(args):
    pl.seed_everything(args.seed, workers=True)
    dm = make_datamodule(args, aug=args.aug)
    model = UNetPlusPlusModule(encoder_name=args.encoder, loss=args.loss, lr=args.lr,
                               weight_decay=args.weight_decay)

    ckpt_dir = OUT / "checkpoints" / args.run_name
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best-epoch{epoch:02d}-miou{val/mIoU:.4f}",
        auto_insert_metric_name=False,
        monitor="val/mIoU",
        mode="max",
        save_top_k=1,
        save_last=False,
    )
    callbacks = [ckpt_cb, LearningRateMonitor(logging_interval="step")]
    if args.patience > 0:
        callbacks.append(EarlyStopping(monitor="val/mIoU", mode="max", patience=args.patience))

    loggers = [CSVLogger(save_dir=OUT / "logs", name=args.run_name)]
    if args.wandb:
        loggers.append(WandbLogger(project=args.wandb_project, name=args.run_name,
                                   save_dir=str(OUT / "logs"), log_model=False))
        loggers[-1].log_hyperparams(vars(args))

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        max_time=args.max_time,
        accelerator="auto",
        devices=1,
        precision=args.precision,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=20,
        limit_train_batches=args.limit_batches,
        limit_val_batches=args.limit_batches,
        deterministic=False,
    )
    trainer.fit(model, datamodule=dm)

    best = ckpt_cb.best_model_path
    print(f"\nBest checkpoint: {best}  (val mIoU={float(ckpt_cb.best_model_score):.4f})")
    export_weights(best, ckpt_dir / "best_weights.pt")
    print(f"Weights-only copy: {ckpt_dir / 'best_weights.pt'}")

    if not args.no_test:
        args.limit_batches = args.limit_batches
        results = run_eval(best, args, tta_too=True, split="test")
        if args.wandb:
            import wandb

            for r in results:
                suffix = "_tta" if r["tta"] else ""
                wandb.run.summary.update({f"test{suffix}/mIoU": r["mIoU"], f"test{suffix}/mDice": r["mDice"],
                                          **{f"test{suffix}/IoU_{c}": v for c, v in r["IoU"].items()},
                                          **{f"test{suffix}/Dice_{c}": v for c, v in r["Dice"].items()}})
            wandb.finish()


def run_eval(ckpt: str, args, tta_too: bool, split: str) -> list[dict]:
    model = load_model(ckpt)
    dm = make_datamodule(args, aug="none")
    results = [evaluate(model, dm, args, tta=False, split=split)]
    if tta_too:
        results.append(evaluate(model, dm, args, tta=True, split=split))
    for r in results:
        print_scores(r)

    out = Path(ckpt).with_name(f"{split}_metrics.json")
    with open(out, "w") as f:
        json.dump({"checkpoint": os.path.relpath(ckpt, ROOT), "img_size": args.img_size,
                   "results": results}, f, indent=2)
    print(f"\nSaved metrics -> {out}")
    return results


def cmd_test(args):
    run_eval(args.ckpt, args, tta_too=args.tta, split=args.split)


if __name__ == "__main__":
    a = build_parser().parse_args()
    torch.set_float32_matmul_precision("high")
    {"fit": cmd_fit, "test": cmd_test}[a.cmd](a)
