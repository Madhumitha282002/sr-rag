# Project Goal

## Problem
Researchers reading image super-resolution literature face a fragmented landscape:
dozens of papers with overlapping methods, different datasets, and inconsistent
reporting of metrics. Finding which papers use perceptual loss, or how SwinIR
compares to EDSR, requires manually scanning many PDFs.

## Solution
SR-RAG is a retrieval-augmented generation system that lets users ask natural
language questions across a curated corpus of SR papers and receive grounded
answers with paper/page citations.

## Target capabilities
- Ask "which papers use adversarial loss?" and get cited answers
- Compare methods across papers ("SRGAN vs ESRGAN architecture differences")
- Refuse to answer when the corpus doesn't contain enough evidence

## Success criteria
Recall@5 ≥ 0.7 on a hand-labeled eval set, average latency < 5s,
faithfulness rate > 80% on manual review.
