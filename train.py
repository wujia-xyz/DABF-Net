"""Train DABF-Net on BUSI under 5-fold cross-validation.

Usage:
    python train.py --variant dabf_s --fold 0 --data-root /path/to/BUSI

Repeat with --fold 0..4 to reproduce the per-fold checkpoints reported in the
paper.

Outputs (under ./outputs/<variant>/fold<k>/):
    checkpoints/best.pth   best-DSC checkpoint
    config.json            resolved run configuration
    training.log           per-epoch metrics
"""
import argparse
import json
import logging
import os
import random
import sys
import time
from itertools import chain, combinations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import BUSIDataset, get_train_transforms, get_val_transforms
from models import build_dabf_net
from utils import DiceLoss, BoundaryLoss, compute_metrics


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("train")
    logger.handlers = []
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(os.path.join(log_dir, "training.log"))
    sh = logging.StreamHandler(sys.stdout)
    for h in (fh, sh):
        h.setFormatter(fmt); logger.addHandler(h)
    return logger


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_model_cfg(yaml_cfg, variant):
    """Merge defaults with the chosen variant block."""
    if variant not in yaml_cfg["variants"]:
        raise ValueError(f"Unknown variant '{variant}'. "
                         f"Available: {list(yaml_cfg['variants'].keys())}")
    cfg = dict(yaml_cfg["defaults"])
    cfg.update(yaml_cfg["variants"][variant])
    cfg.pop("description", None)
    return cfg


def load_split(splits_path, data_root, fold):
    """Resolve {train,val}_files to absolute image and mask paths."""
    with open(splits_path) as f:
        meta = json.load(f)
    fold_info = meta["splits"][fold]
    img_dir = os.path.join(data_root, meta["image_dir"])
    msk_dir = os.path.join(data_root, meta["mask_dir"])
    suf, ext = meta["mask_suffix"], meta["ext"]

    def to_paths(files):
        imgs, masks = [], []
        for fname in files:
            stem = fname[:-len(ext)] if fname.endswith(ext) else os.path.splitext(fname)[0]
            imgs.append(os.path.join(img_dir, fname))
            masks.append(os.path.join(msk_dir, f"{stem}{suf}{ext}"))
        return imgs, masks

    tr_i, tr_m = to_paths(fold_info["train_files"])
    va_i, va_m = to_paths(fold_info["val_files"])
    return {
        "train_imgs": tr_i, "train_masks": tr_m,
        "val_imgs":   va_i, "val_masks":   va_m,
        "fold": fold, "n_total": meta["n_total"],
    }


# ---------------------------------------------------------------------------
# combinatorial multi-scale supervision
# ---------------------------------------------------------------------------
def powerset(it):
    it = list(it)
    return list(chain.from_iterable(combinations(it, r) for r in range(len(it) + 1)))


def build_subsets(n_outs, mode):
    out_idxs = list(range(n_outs))
    if mode == "mutation":
        return [list(s) for s in powerset(out_idxs) if s]
    if mode == "deep_supervision":
        return [[i] for i in out_idxs]
    return [[out_idxs[-1]]]


def compute_multiscale_loss(outputs, masks, subsets, ce_loss, dice_loss,
                            boundary_loss, w_ce, w_dice, w_boundary):
    loss = 0.0
    for subset in subsets:
        agg = None
        for idx in subset:
            agg = outputs[idx] if agg is None else agg + outputs[idx]
        loss = loss + w_ce * ce_loss(agg, masks) + w_dice * dice_loss(agg, masks)
    if boundary_loss is not None and w_boundary > 0:
        loss = loss + w_boundary * boundary_loss(outputs[-1], masks)
    return loss


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["dabf_s", "dabf_l"], default="dabf_s")
    p.add_argument("--fold", type=int, required=True, help="0..4")
    p.add_argument("--data-root", required=True, help="BUSI dataset root containing images/ and seg/")
    p.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "dabf_net.yaml"))
    p.add_argument("--splits", default=os.path.join(PROJECT_ROOT, "splits", "busi_5fold.json"))
    p.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "outputs"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # config
    yaml_cfg = load_yaml(args.config)
    model_cfg = resolve_model_cfg(yaml_cfg, args.variant)
    t_cfg = yaml_cfg["training"]

    # output dirs
    run_dir = os.path.join(args.output_dir, args.variant, f"fold{args.fold}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = setup_logger(run_dir)

    # split
    split = load_split(args.splits, args.data_root, args.fold)
    logger.info(f"variant={args.variant}  fold={args.fold}  "
                f"train={len(split['train_imgs'])}  val={len(split['val_imgs'])}")

    # data
    img_size = t_cfg["image_size"]
    train_ds = BUSIDataset(split["train_imgs"], split["train_masks"],
                           transform=get_train_transforms(img_size), mode="train")
    val_ds   = BUSIDataset(split["val_imgs"],   split["val_masks"],
                           transform=get_val_transforms(img_size), mode="val")
    train_loader = DataLoader(train_ds, batch_size=t_cfg["batch_size"], shuffle=True,
                              num_workers=t_cfg["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=t_cfg["num_workers"], pin_memory=True)

    # model
    model = build_dabf_net(model_cfg).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    logger.info(f"params={n_params/1e6:.2f} M")

    # loss / optim
    pos_w = torch.tensor([t_cfg["pos_weight"]], device=device)
    ce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    dice_loss = DiceLoss()
    boundary_loss = BoundaryLoss() if t_cfg["use_boundary_loss"] else None
    optimizer = optim.AdamW(model.parameters(), lr=t_cfg["lr"],
                            weight_decay=t_cfg["weight_decay"])
    if t_cfg["scheduler"] == "constant":
        scheduler = optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=t_cfg["epochs"] + 1)
    elif t_cfg["scheduler"] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_cfg["epochs"], eta_min=t_cfg["min_lr"])
    else:
        raise ValueError(f"Unknown scheduler: {t_cfg['scheduler']}")

    # snapshot
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"args": vars(args), "model": model_cfg, "training": t_cfg,
                   "n_params": n_params}, f, indent=2)

    # train
    sup_mode = t_cfg["supervision"]
    subset_cache = {}
    best_dice = -1.0

    for epoch in range(t_cfg["epochs"]):
        model.train()
        t0, tot = time.time(), 0.0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            masks  = batch["mask"].to(device, non_blocking=True)
            outs   = model(images)
            if not isinstance(outs, (list, tuple)): outs = [outs]
            k = (sup_mode, len(outs))
            if k not in subset_cache: subset_cache[k] = build_subsets(len(outs), sup_mode)
            loss = compute_multiscale_loss(
                outs, masks, subset_cache[k], ce_loss, dice_loss, boundary_loss,
                t_cfg["supervision_ce_weight"], t_cfg["supervision_dice_weight"],
                t_cfg["boundary_loss_weight"])
            optimizer.zero_grad()
            loss.backward()
            if t_cfg["clip_grad"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg["clip_grad"])
            optimizer.step()
            tot += loss.item()
        train_loss = tot / len(train_loader)

        # validate
        model.eval()
        dice_sum, n_val = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device, non_blocking=True)
                masks  = batch["mask"].to(device, non_blocking=True)
                outs   = model(images)
                if not isinstance(outs, (list, tuple)): outs = [outs]
                m = compute_metrics(outs[-1], masks)
                dice_sum += m["dice"]; n_val += 1
        val_dice = dice_sum / max(n_val, 1)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        dt = time.time() - t0

        is_best = val_dice > best_dice
        logger.info(f"E{epoch:03d}/{t_cfg['epochs']}  lr={lr:.2e}  t={dt:.1f}s  "
                    f"train_loss={train_loss:.4f}  val_dice={val_dice:.4f}"
                    f"{' *' if is_best else ''}")

        if is_best:
            best_dice = val_dice
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "metrics": {"epoch": epoch, "dice": val_dice},
                        "model_cfg": model_cfg},
                       os.path.join(ckpt_dir, "best.pth"))

    logger.info(f"DONE.  best val_dice={best_dice:.4f}")


if __name__ == "__main__":
    main()
