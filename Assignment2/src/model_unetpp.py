"""LightningModule: UNet++ (segmentation_models_pytorch) for 3-class trimap segmentation."""
from __future__ import annotations

import math

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassConfusionMatrix

from datamodule_oxpet import CLASS_NAMES, NUM_CLASSES


def scores_from_confmat(cm: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-class IoU and Dice from a confusion matrix (rows = target, cols = pred).

    Scores are computed from pixel counts accumulated over the whole split
    (not averaged per image), which is the usual convention for mIoU.
    """
    cm = cm.detach().cpu().double()
    tp = cm.diag()
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    iou = tp / (tp + fp + fn).clamp(min=1)
    dice = 2 * tp / (2 * tp + fp + fn).clamp(min=1)
    acc = tp.sum() / cm.sum().clamp(min=1)
    return {"iou": iou, "dice": dice, "miou": iou.mean(), "mdice": dice.mean(), "acc": acc}


class UNetPlusPlusModule(pl.LightningModule):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        loss: str = "ce",
        dice_weight: float = 1.0,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: float = 1.0,
        tta: bool = False,  
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=NUM_CLASSES,
        )
        self.ce = nn.CrossEntropyLoss()
        self.dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True) if loss == "ce_dice" else None
        if loss not in {"ce", "ce_dice"}:
            raise ValueError(f"unknown loss '{loss}'")

        self.train_cm = MulticlassConfusionMatrix(NUM_CLASSES)
        self.val_cm = MulticlassConfusionMatrix(NUM_CLASSES)
        self.test_cm = MulticlassConfusionMatrix(NUM_CLASSES)
        self.last_test_scores: dict | None = None


    def forward(self, x):
        return self.model(x)

    @torch.no_grad()
    def predict_probs(self, x, tta: bool | None = None):
        """Softmax probabilities, optionally averaged with the h-flipped input."""
        tta = self.hparams.tta if tta is None else tta
        probs = self(x).float().softmax(1)
        if tta:
            probs = 0.5 * (probs + self(x.flip(-1)).float().softmax(1).flip(-1))
        return probs

    def compute_loss(self, logits, y):
        loss = self.ce(logits, y)
        if self.dice is not None:
            loss = loss + self.hparams.dice_weight * self.dice(logits, y)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.compute_loss(logits, y)
        self.train_cm.update(logits.argmax(1), y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=x.size(0))
        return loss

    def on_train_epoch_end(self):
        s = scores_from_confmat(self.train_cm.compute())
        self.log_dict({"train/mIoU": s["miou"].float(), "train/mDice": s["mdice"].float()})
        self.train_cm.reset()

    def _eval_step(self, batch, cm, stage):
        x, y = batch
        logits = self(x)
        self.log(f"{stage}/loss", self.compute_loss(logits, y), on_epoch=True, prog_bar=True,
                 batch_size=x.size(0))
        probs = logits.float().softmax(1)
        if self.hparams.tta:
            probs = 0.5 * (probs + self(x.flip(-1)).float().softmax(1).flip(-1))
        cm.update(probs.argmax(1), y)

    def _eval_epoch_end(self, cm, stage):
        s = scores_from_confmat(cm.compute())
        logs = {f"{stage}/mDice": s["mdice"], f"{stage}/pixel_acc": s["acc"]}
        for i, name in enumerate(CLASS_NAMES):
            logs[f"{stage}/IoU_{name}"] = s["iou"][i]
            logs[f"{stage}/Dice_{name}"] = s["dice"][i]
        self.log_dict({k: v.float() for k, v in logs.items()}, prog_bar=False)
        self.log(f"{stage}/mIoU", s["miou"].float(), prog_bar=True)
        cm.reset()
        return s

    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, self.val_cm, "val")

    def on_validation_epoch_end(self):
        self._eval_epoch_end(self.val_cm, "val")

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, self.test_cm, "test")

    def on_test_epoch_end(self):
        s = self._eval_epoch_end(self.test_cm, "test")
        self.last_test_scores = {k: v.tolist() for k, v in s.items()}

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        total = self.trainer.estimated_stepping_batches
        steps_per_epoch = max(1, total // max(1, self.trainer.max_epochs))
        warmup = int(self.hparams.warmup_epochs * steps_per_epoch)

        def lr_lambda(step):
            if step < warmup:
                return (step + 1) / warmup
            t = (step - warmup) / max(1, total - warmup)
            return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
