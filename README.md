# DABF-Net

Official PyTorch implementation of **DABF-Net: Dynamic Adaptive Basis Fusion Network for Breast Ultrasound Image Segmentation** (Expert Systems with Applications, 2026).

DABF-Net combines a per-pixel content-adaptive convolution (DAB-Block, low-rank basis + top-k sparsification + hybrid static/dynamic branches), a Learnable Gated Attention Guidance (LGAG) module for noise-aware skip-connection fusion, and a combinatorial multi-scale supervision strategy that adds no inference cost.

## Repository contents

```
DABF-Net/
├── models/              DABF-Net implementation (DAB-Block, LGAG, encoder/decoder)
├── data/                BUSI dataset loader + augmentation pipeline
├── utils/               losses (Dice / Boundary / BCE) and segmentation metrics
├── configs/dabf_net.yaml  S and L variant configs + training hyperparameters
├── splits/busi_5fold.json BUSI 5-fold cross-validation split file
├── train.py             5-fold training entry
├── evaluate.py          checkpoint evaluation entry
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/wujia-xyz/DABF-Net.git
cd DABF-Net
pip install -r requirements.txt
```

Tested with PyTorch 2.0 on one NVIDIA RTX 5090.

## Datasets

### Public benchmarks

| Dataset | Source |
|---|---|
| BUSI    | [Al-Dhabyani et al. 2020, *Data in Brief*](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) |
| BrEaST  | [Pawłowska et al. 2024, *Scientific Data*](https://www.nature.com/articles/s41597-024-02985-y) |

Expected layout after preparing BUSI (drop the 133 normal images):

```
BUSI_lesion_dataset/
├── images/
│   ├── benign (1).png
│   ├── ...
└── seg/
    ├── benign (1)_mask.png
    └── ...
```

### Our SoMed-BUS dataset (released)

The 6,572-image in-house cohort (Affiliated Hospital of Southwest Medical University) used for external validation in the paper:

> https://1drv.ms/f/c/93483cb9d8985636/IgCEs4lCjCU7S6VnVFphR09MAd7fSTT5PTLFFGaxAPWiIYE?e=wUNaNn

## Training

5-fold cross-validation on BUSI:

```bash
# DABF-Net-S, all five folds
for f in 0 1 2 3 4; do
    python train.py --variant dabf_s --fold $f --data-root /path/to/BUSI_lesion_dataset
done

# DABF-Net-L
python train.py --variant dabf_l --fold 0 --data-root /path/to/BUSI_lesion_dataset
```

Outputs land under `outputs/<variant>/fold<k>/checkpoints/best.pth`.

Default settings (in `configs/dabf_net.yaml`): 200 epochs, AdamW, constant lr 1e-4, batch 4, 256×256 inputs, combinatorial multi-scale supervision, boundary-weighted BCE+Dice loss.

## Evaluation

```bash
python evaluate.py --variant dabf_s --fold 0 \
    --checkpoint outputs/dabf_s/fold0/checkpoints/best.pth \
    --data-root /path/to/BUSI_lesion_dataset
```

Reports per-fold DSC / Jaccard / Precision / Sensitivity / Specificity (mean ± std across the held-out images of that fold).

## Pretrained checkpoints

We release the **DABF-Net-S** BUSI 5-fold checkpoints (one `.pth` per fold):

> https://1drv.ms/f/c/93483cb9d8985636/IgBw9JytGAoVTrBDFGnstV65ASNZFeEt4R3hFyAjEseJYhU?e=nEpvNl

DABF-Net-L checkpoints are not released; you can reproduce them with `train.py --variant dabf_l`.

## Citation

```bibtex
@article{dabfnet2026,
  title  = {DABF-Net: Dynamic Adaptive Basis Fusion Network for Breast Ultrasound Image Segmentation},
  author = {...},
  journal= {Expert Systems with Applications},
  year   = {2026}
}
```

## License

[MIT](LICENSE)
