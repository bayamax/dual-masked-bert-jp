from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in ("float16", "fp16", "half"):
        return torch.float16
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16
    return torch.float32


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a_norm = a / (a.norm(dim=-1, keepdim=True) + eps)
    b_norm = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (a_norm * b_norm).sum(dim=-1)


