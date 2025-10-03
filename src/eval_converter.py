from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .train_converter import build_model
from .utils import cosine_similarity, get_device


def load_dataset(path: Path):
    obj = torch.load(path)
    return obj["X"].to(torch.float32), obj["Y"].to(torch.float32), obj.get("meta", {})


def load_converter(path: Path):
    ckpt = torch.load(path, map_location="cpu")
    meta = ckpt["meta"]
    model = build_model(meta["model_type"], meta["d_model"], meta.get("hidden_dim", 2048))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, meta


@torch.inference_mode()
def evaluate(dataset_path: Path, converter_path: Path, topk: int = 5, query_bs: int = 2048, ref_bs: int = 4096, use_cpu_topk: bool = True) -> None:
    device = get_device()
    X, Y, meta_ds = load_dataset(dataset_path)
    model, meta = load_converter(converter_path)
    model.to(device)

    X = X.to(device)
    Y = Y.to(device)

    # 予測
    Y_hat = torch.empty_like(Y)
    bs = 8192
    for i in tqdm(range(0, X.size(0), bs), desc="predict"):
        Y_hat[i : i + bs] = model(X[i : i + bs])

    # MSE / Cosine
    mse = torch.mean((Y_hat - Y) ** 2).item()
    cos = torch.mean(cosine_similarity(Y_hat, Y)).item()
    # Norm RMSE（方向一致時のスケール差を見る）
    rmse_norm = torch.sqrt(torch.mean((Y_hat.norm(dim=-1) - Y.norm(dim=-1)) ** 2)).item()

    # 近傍検索（CPUストリーミングTop-K。大語彙でのメモリ節約）
    if use_cpu_topk:
        sim_device = torch.device("cpu")
    else:
        sim_device = device

    Y_norm = F.normalize(Y.to(sim_device, dtype=torch.float32), dim=-1)
    Yh_norm = F.normalize(Y_hat.to(sim_device, dtype=torch.float32), dim=-1)

    N = Y_norm.size(0)
    K = min(topk, N)
    all_topk_idx = torch.empty((N, K), dtype=torch.long)

    for i in tqdm(range(0, N, query_bs), desc="topk-query"):
        q = Yh_norm[i : i + query_bs]  # [B, d]
        bsz = q.size(0)
        best_val = None  # [B, K]
        best_idx = None  # [B, K]
        for j in range(0, N, ref_bs):
            ref = Y_norm[j : j + ref_bs]  # [C, d]
            sim = q @ ref.T  # [B, C]
            cur_val, cur_loc = torch.topk(sim, k=min(K, sim.size(1)), dim=1)
            cur_idx = cur_loc + j
            if best_val is None:
                best_val = cur_val
                best_idx = cur_idx
            else:
                cat_val = torch.cat([best_val, cur_val], dim=1)
                cat_idx = torch.cat([best_idx, cur_idx], dim=1)
                sel_val, sel_pos = torch.topk(cat_val, k=K, dim=1)
                best_val = sel_val
                best_idx = torch.gather(cat_idx, 1, sel_pos)
        all_topk_idx[i : i + bsz] = best_idx

    gt = torch.arange(N).unsqueeze(1)
    match_top1 = (all_topk_idx[:, :1] == gt).float().mean().item()
    match_topk = (all_topk_idx == gt).any(dim=1).float().mean().item()

    print({
        "mse": mse,
        "cos": cos,
        "rmse_norm": rmse_norm,
        "top1_acc": match_top1,
        "topk_acc": match_topk,
        "topk": topk,
    })


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", type=Path, required=True)
    p.add_argument("--converter_path", type=Path, required=True)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--query_bs", type=int, default=2048)
    p.add_argument("--ref_bs", type=int, default=4096)
    p.add_argument("--cpu_topk", action="store_true", default=True)
    p.add_argument("--no-cpu_topk", dest="cpu_topk", action="store_false")
    args = p.parse_args()

    evaluate(
        args.dataset_path,
        args.converter_path,
        topk=args.topk,
        query_bs=args.query_bs,
        ref_bs=args.ref_bs,
        use_cpu_topk=args.cpu_topk,
    )


if __name__ == "__main__":
    main()


