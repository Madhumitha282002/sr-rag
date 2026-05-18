"""
Create data/processed/paper_metadata.csv from the downloaded PDFs.
Run: python scripts/build_metadata.py
"""
import csv
from pathlib import Path

METADATA = [
    {
        "file_name":    "srcnn_2014.pdf",
        "title":        "Learning a Deep Convolutional Network for Image Super-Resolution",
        "authors":      "Dong et al.",
        "year":         2014,
        "method":       "SRCNN",
        "venue":        "ECCV",
        "datasets":     "Set5, Set14",
        "key_contribution": "First end-to-end CNN for SR",
        "source_url":   "https://arxiv.org/abs/1501.00092",
    },
    {
        "file_name":    "vdsr_2015.pdf",
        "title":        "Accurate Image Super-Resolution Using Very Deep Convolutional Networks",
        "authors":      "Kim et al.",
        "year":         2015,
        "method":       "VDSR",
        "venue":        "CVPR",
        "datasets":     "Set5, Set14, BSD100",
        "key_contribution": "Very deep residual network for SR, gradient clipping",
        "source_url":   "https://arxiv.org/abs/1511.04587",
    },
    {
        "file_name":    "srgan_2016.pdf",
        "title":        "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network",
        "authors":      "Ledig et al.",
        "year":         2016,
        "method":       "SRGAN / SRResNet",
        "venue":        "CVPR",
        "datasets":     "Set5, BSD100, ImageNet",
        "key_contribution": "GAN + perceptual loss for photo-realistic SR",
        "source_url":   "https://arxiv.org/abs/1609.04802",
    },
    {
        "file_name":    "edsr_2017.pdf",
        "title":        "Enhanced Deep Residual Networks for Single Image Super-Resolution",
        "authors":      "Lim et al.",
        "year":         2017,
        "method":       "EDSR",
        "venue":        "CVPRW",
        "datasets":     "DIV2K, Set5, Set14, BSD100",
        "key_contribution": "Removes batch norm from ResNet, winner of NTIRE 2017",
        "source_url":   "https://arxiv.org/abs/1707.02921",
    },
    {
        "file_name":    "rcan_2018.pdf",
        "title":        "Image Super-Resolution Using Very Deep Residual Channel Attention Networks",
        "authors":      "Zhang et al.",
        "year":         2018,
        "method":       "RCAN",
        "venue":        "ECCV",
        "datasets":     "Set5, Set14, BSD100, Urban100, Manga109",
        "key_contribution": "Channel attention mechanism for SR",
        "source_url":   "https://arxiv.org/abs/1807.02758",
    },
    {
        "file_name":    "esrgan_2018.pdf",
        "title":        "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks",
        "authors":      "Wang et al.",
        "year":         2018,
        "method":       "ESRGAN",
        "venue":        "ECCVW",
        "datasets":     "DIV2K, Flickr2K",
        "key_contribution": "RRDB architecture, relativistic discriminator, improved SRGAN",
        "source_url":   "https://arxiv.org/abs/1809.00219",
    },
    {
        "file_name":    "rdn_2018.pdf",
        "title":        "Residual Dense Network for Image Super-Resolution",
        "authors":      "Zhang et al.",
        "year":         2018,
        "method":       "RDN",
        "venue":        "CVPR",
        "datasets":     "Set5, Set14, BSD100, Urban100, Manga109",
        "key_contribution": "Dense connections across residual blocks for SR",
        "source_url":   "https://arxiv.org/abs/1802.08797",
    },
    {
        "file_name":    "realesrgan_2021.pdf",
        "title":        "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data",
        "authors":      "Wang et al.",
        "year":         2021,
        "method":       "Real-ESRGAN",
        "venue":        "ICCVW",
        "datasets":     "DIV2K, Flickr2K, OST",
        "key_contribution": "Practical blind SR with high-order degradation pipeline",
        "source_url":   "https://arxiv.org/abs/2107.10833",
    },
    {
        "file_name":    "swinir_2021.pdf",
        "title":        "SwinIR: Image Restoration Using Swin Transformer",
        "authors":      "Liang et al.",
        "year":         2021,
        "method":       "SwinIR",
        "venue":        "ICCVW",
        "datasets":     "DIV2K, Flickr2K, Set5, Set14, BSD100, Urban100, Manga109",
        "key_contribution": "First strong transformer backbone for image restoration",
        "source_url":   "https://arxiv.org/abs/2108.10257",
    },
    {
        "file_name":    "hat_2022.pdf",
        "title":        "Activating More Pixels in Image Super-Resolution Transformer",
        "authors":      "Chen et al.",
        "year":         2022,
        "method":       "HAT",
        "venue":        "CVPR",
        "datasets":     "DIV2K, Flickr2K, Set5, Set14, BSD100, Urban100, Manga109",
        "key_contribution": "Hybrid attention transformer, SOTA at release",
        "source_url":   "https://arxiv.org/abs/2205.04437",
    },
]

OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)
csv_path = OUT / "paper_metadata.csv"

fieldnames = ["file_name","title","authors","year","method","venue",
              "datasets","key_contribution","source_url"]

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(METADATA)

print(f"Wrote {len(METADATA)} rows to {csv_path}")
