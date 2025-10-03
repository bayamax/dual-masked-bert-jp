from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple

import torch


def load_agg(path: Path) -> List[Dict[str, Any]]:
    obj = torch.load(path, map_location="cpu")
    samples = obj.get("samples", [])
    return [normalize_sample(s) for s in samples]


def load_stream(stream_dir: Path, head: int | None = None) -> List[Dict[str, Any]]:
    paths = sorted(glob.glob(str(stream_dir / "sample_*.pt")))
    if head is not None:
        paths = paths[: head]
    out: List[Dict[str, Any]] = []
    for p in paths:
        s = torch.load(p, map_location="cpu")
        out.append(normalize_sample(s))
    return out


def normalize_sample(s: Dict[str, Any]) -> Dict[str, Any]:
    # 期待構造: Hctx [L,d], Hmask [L], e_star [k,d], text, follow_text, meta
    s = dict(s)
    if isinstance(s.get("Hctx"), torch.Tensor):
        s["Hctx"] = s["Hctx"].detach().to(torch.float32).cpu()
    if isinstance(s.get("Hmask"), torch.Tensor):
        s["Hmask"] = s["Hmask"].detach().to(torch.long).cpu()
    if isinstance(s.get("e_star"), torch.Tensor):
        s["e_star"] = s["e_star"].detach().to(torch.float32).cpu()
    return s


def dedup_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[Tuple[str, str]] = set()
    uniq: List[Dict[str, Any]] = []
    for s in samples:
        key = (str(s.get("text", "")), str(s.get("follow_text", "")))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)
    return uniq


def filter_k1(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in samples:
        e = s.get("e_star")
        if not isinstance(e, torch.Tensor):
            continue
        if e.dim() == 2 and e.size(0) == 1:
            out.append(s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", type=Path, nargs="*", default=[])
    ap.add_argument("--stream_dir", type=Path, default=None)
    ap.add_argument("--head", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k1_only", action="store_true", default=False)
    args = ap.parse_args()

    all_samples: List[Dict[str, Any]] = []
    for p in args.agg:
        if p.exists():
            cur = load_agg(p)
            print({"load_agg": str(p), "num": len(cur)}, flush=True)
            all_samples.extend(cur)
    if args.stream_dir is not None and args.stream_dir.exists():
        cur = load_stream(args.stream_dir, head=args.head)
        print({"load_stream": str(args.stream_dir), "num": len(cur)}, flush=True)
        all_samples.extend(cur)

    before = len(all_samples)
    all_samples = dedup_samples(all_samples)
    after_dedup = len(all_samples)
    if args.k1_only:
        all_samples = filter_k1(all_samples)
    after_k1 = len(all_samples)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"samples": all_samples}, args.out)
    print({
        "saved": str(args.out),
        "loaded": before,
        "dedup": after_dedup,
        "k1": after_k1,
    }, flush=True)


if __name__ == "__main__":
    main()


