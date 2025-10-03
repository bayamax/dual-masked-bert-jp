from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from .utils import get_device, set_seed


class LinearConverter(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MLPConverter(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(model_type: str, d_model: int, hidden_dim: int) -> nn.Module:
    if model_type == "linear":
        return LinearConverter(d_model)
    if model_type == "mlp":
        return MLPConverter(d_model, hidden_dim)
    raise ValueError(f"unknown model_type: {model_type}")


def load_dataset(path: Path) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    obj = torch.load(path)
    X = obj["X"].to(torch.float32)
    Y = obj["Y"].to(torch.float32)
    meta = obj.get("meta", {})
    return X, Y, meta


def train(
    dataset_path: Path,
    model_type: str,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    save_path: Path,
    val_ratio: float = 0.05,
    loss_type: str = "mse",
    norm_weight: float = 0.0,
) -> None:
    set_seed(42)
    device = get_device()

    X, Y, meta = load_dataset(dataset_path)
    d_model = int(meta.get("d_model", X.shape[1]))

    dataset = TensorDataset(X, Y)
    n_val = max(1, int(len(dataset) * val_ratio))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = build_model(model_type, d_model, hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def compute_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if loss_type == "mse":
            return nn.functional.mse_loss(pred, target)
        # cosine: 1 - cos(pred, target)
        pred_n = nn.functional.normalize(pred, dim=-1)
        targ_n = nn.functional.normalize(target, dim=-1)
        cos = (pred_n * targ_n).sum(dim=-1)
        cos_loss = 1.0 - cos
        if loss_type == "cosine":
            return cos_loss.mean()
        if loss_type == "cosine_norm":
            # 方向 + ノルム整合
            pred_norm = pred.norm(dim=-1)
            targ_norm = target.norm(dim=-1)
            norm_l2 = (pred_norm - targ_norm).pow(2)
            return (cos_loss + norm_weight * norm_l2).mean()
        raise ValueError(f"unknown loss_type: {loss_type}")

    best_val = float("inf")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in tqdm(train_loader, desc=f"train epoch {epoch}"):
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = compute_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
        train_loss = total / n_train

        model.eval()
        total_v = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = compute_loss(pred, yb)
                total_v += loss.item() * xb.size(0)
        val_loss = total_v / n_val

        print({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "state_dict": model.state_dict(),
                "meta": {
                    "d_model": d_model,
                    "model_type": model_type,
                    "hidden_dim": hidden_dim,
                    "dataset_meta": meta,
                    "loss_type": loss_type,
                    "norm_weight": norm_weight,
                },
            }, save_path)
            print("saved best to", save_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", type=Path, required=True)
    p.add_argument("--model_type", type=str, default="mlp", choices=["linear", "mlp"])
    p.add_argument("--hidden_dim", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--save_path", type=Path, default=Path("artifacts/converter.pt"))
    p.add_argument("--loss_type", type=str, default="mse", choices=["mse", "cosine", "cosine_norm"])
    p.add_argument("--norm_weight", type=float, default=0.1)
    args = p.parse_args()

    train(
        dataset_path=args.dataset_path,
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        save_path=args.save_path,
        loss_type=args.loss_type,
        norm_weight=args.norm_weight,
    )


if __name__ == "__main__":
    main()


