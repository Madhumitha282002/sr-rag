"""
Confirm every PDF in paper_metadata.csv is present and parseable.
Run: python scripts/verify_corpus.py
"""
import csv
import fitz   # PyMuPDF
from pathlib import Path

RAW_DIR  = Path("data/raw_papers")
META_CSV = Path("data/processed/paper_metadata.csv")

print(f"\n{'='*55}")
print(f"  SR-RAG corpus verification")
print(f"{'='*55}")

passed, failed = [], []

with open(META_CSV, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        pdf_path = RAW_DIR / row["file_name"]

        if not pdf_path.exists():
            print(f"  MISSING  {row['file_name']}")
            failed.append(row["file_name"])
            continue

        try:
            doc  = fitz.open(pdf_path)
            pages = len(doc)
            words = sum(len(p.get_text().split()) for p in doc)
            size_kb = pdf_path.stat().st_size // 1024
            print(f"  OK  {row['method']:<14} {pages:>3} pages  {words:>6} words  {size_kb:>5} KB")
            passed.append(row["file_name"])
        except Exception as e:
            print(f"  ERROR  {row['file_name']}: {e}")
            failed.append(row["file_name"])

print(f"{'='*55}")
print(f"  {len(passed)} passed   {len(failed)} failed")
if failed:
    print(f"\n  Re-run download for failed files:")
    for f in failed:
        print(f"    {f}")
print()
