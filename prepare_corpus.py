#!/usr/bin/env python
"""Prepare corpus for ContextConditionedMLM.

Usage::
    python prepare_corpus.py --dataset wiki40b --config ja --split train \
        --max_docs 100000 --output_dir data --train_ratio 0.99

Output::
    data/train.jsonl
    data/val.jsonl
Each line: {"id": str, "sentences": [str, ...]}
"""
from __future__ import annotations
import argparse, json, os, re, random, itertools
from pathlib import Path
from typing import List
from datasets import load_dataset, IterableDataset
from tqdm import tqdm

JP_SENT_END = re.compile(r"(?<=[。．？！?！])\s*")


def sentence_split(text: str) -> List[str]:
    """Very naive Japanese sentence splitter based on punctuation."""
    sents = [s.strip() for s in JP_SENT_END.split(text) if s.strip()]
    return sents


def iter_documents(dataset: IterableDataset, min_sent: int) -> tuple[str, List[str]]:
    for ex in dataset:
        doc_id = ex.get("id") or ex.get("url") or ex.get("title") or str(random.getrandbits(64))
        text = ex.get("text") or ex.get("content") or ""
        sents = sentence_split(text)
        if len(sents) >= min_sent:
            yield doc_id, sents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="wiki40b", help="HF dataset name")
    parser.add_argument("--config", default="ja", help="HF dataset config")
    parser.add_argument("--split", default="train", help="Split name")
    parser.add_argument("--max_docs", type=int, default=100000, help="Maximum documents to process (0 = all)")
    parser.add_argument("--min_sent", type=int, default=2, help="Minimum sentences per doc to keep")
    parser.add_argument("--train_ratio", type=float, default=0.99, help="Train split ratio; remainder is val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--append", action="store_true", help="append to existing JSONL instead of overwrite")
    args = parser.parse_args()

    random.seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)

    train_path = Path(args.output_dir) / "train.jsonl"
    val_path = Path(args.output_dir) / "val.jsonl"

    mode = "a" if args.append else "w"
    train_f = train_path.open(mode, encoding="utf-8")
    val_f = val_path.open(mode, encoding="utf-8")

    doc_iter = iter_documents(ds, args.min_sent)
    total = args.max_docs if args.max_docs > 0 else None

    for idx, (doc_id, sents) in enumerate(tqdm(itertools.islice(doc_iter, total), total=total, desc="docs")):
        rec = {"id": doc_id, "sentences": sents}
        f = train_f if random.random() < args.train_ratio else val_f
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    train_f.close(); val_f.close()
    print("Saved:", train_path, val_path)


if __name__ == "__main__":
    main()
