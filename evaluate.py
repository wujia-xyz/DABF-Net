"""Evaluate a trained DABF-Net checkpoint on a BUSI 5-fold validation split.

Usage:
    python evaluate.py --variant dabf_s --fold 0 \
        --checkpoint outputs/dabf_s/fold0/checkpoints/best.pth \
        --data-root /path/to/BUSI
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import BUSIDataset, get_val_transforms
from models import build_dabf_net
from utils import compute_metrics

import yaml


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_model_cfg(yaml_cfg, variant):
    cfg = dict(yaml_cfg["defaults"])
    cfg.update(yaml_cfg["variants"][variant])
    cfg.pop("description", None)
    return cfg


def load_split(splits_path, data_root, fold):
    with open(splits_path) as f:
        meta = json.load(f)
    fold_info = meta["splits"][fold]
    img_dir = os.path.join(data_root, meta["image_dir"])
    msk_dir = os.path.join(data_root, meta["mask_dir"])
    suf, ext = meta["mask_suffix"], meta["ext"]
    val_imgs, val_masks = [], []
    for fname in fold_info["val_files"]:
        stem = fname[:-len(ext)] if fname.endswith(ext) else os.path.splitext(fname)[0]
        val_imgs.append(os.path.join(img_dir, fname))
        val_masks.append(os.path.join(msk_dir, f"{stem}{suf}{ext}"))
    return val_imgs, val_masks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["dabf_s", "dabf_l"], default="dabf_s")
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "dabf_net.yaml"))
    p.add_argument("--splits", default=os.path.join(PROJECT_ROOT, "splits", "busi_5fold.json"))
    p.add_argument("--image-size", type=int, default=256)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yaml_cfg = load_yaml(args.config)
    model_cfg = resolve_model_cfg(yaml_cfg, args.variant)

    val_imgs, val_masks = load_split(args.splits, args.data_root, args.fold)
    val_ds = BUSIDataset(val_imgs, val_masks,
                         transform=get_val_transforms(args.image_size), mode="val")
    loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    model = build_dabf_net(model_cfg).to(device)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"loaded {args.checkpoint}")
    if isinstance(ck, dict) and "metrics" in ck:
        print(f"  recorded train-time metrics: {ck['metrics']}")

    metrics = {"dice": [], "iou": [], "precision": [], "sensitivity": [], "specificity": []}
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks  = batch["mask"].to(device, non_blocking=True)
            outs   = model(images)
            if isinstance(outs, (list, tuple)): outs = outs[-1]
            m = compute_metrics(outs, masks)
            for k in metrics:
                if k in m: metrics[k].append(m[k])

    print(f"\nFold {args.fold} ({args.variant}) — n={len(val_imgs)}")
    print(f"{'metric':<14} {'mean':>10} {'std':>10}")
    for k, vs in metrics.items():
        if vs:
            print(f"{k:<14} {np.mean(vs):>10.4f} {np.std(vs):>10.4f}")


if __name__ == "__main__":
    main()
