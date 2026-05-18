"""
Download SR papers from arXiv into data/raw_papers/.
Run: python scripts/download_papers.py
"""
import urllib.request
import time
from pathlib import Path

PAPERS = [
    {"arxiv_id": "1501.00092", "filename": "srcnn_2014.pdf",       "title": "SRCNN",       "year": 2014},
    {"arxiv_id": "1511.04587", "filename": "vdsr_2015.pdf",        "title": "VDSR",        "year": 2015},
    {"arxiv_id": "1609.04802", "filename": "srgan_2016.pdf",       "title": "SRGAN",       "year": 2016},
    {"arxiv_id": "1707.02921", "filename": "edsr_2017.pdf",        "title": "EDSR",        "year": 2017},
    {"arxiv_id": "1807.02758", "filename": "rcan_2018.pdf",        "title": "RCAN",        "year": 2018},
    {"arxiv_id": "1809.00219", "filename": "esrgan_2018.pdf",      "title": "ESRGAN",      "year": 2018},
    {"arxiv_id": "1802.08797", "filename": "rdn_2018.pdf",         "title": "RDN",         "year": 2018},
    {"arxiv_id": "2107.10833", "filename": "realesrgan_2021.pdf",  "title": "RealESRGAN",  "year": 2021},
    {"arxiv_id": "2108.10257", "filename": "swinir_2021.pdf",      "title": "SwinIR",      "year": 2021},
    {"arxiv_id": "2205.04437", "filename": "hat_2022.pdf",         "title": "HAT",         "year": 2022},
]

OUT_DIR = Path("data/raw_papers")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for p in PAPERS:
    dest = OUT_DIR / p["filename"]
    if dest.exists():
        print(f"  already exists: {p['filename']}")
        continue
    url = f"https://arxiv.org/pdf/{p['arxiv_id']}"
    print(f"  downloading {p['title']} ({p['arxiv_id']})...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"    saved to {dest}")
        time.sleep(2)   # be polite to arXiv
    except Exception as e:
        print(f"    FAILED: {e}")

print("\nDone.")
