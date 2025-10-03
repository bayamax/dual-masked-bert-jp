from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

from .hf_model import load_tokenizer_and_model
from .optimize_virtual_token import optimize_virtual


def load_pairs_from_jsonl(path: Path) -> List[Dict[str, str]]:
    pairs: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "text" in obj and "follow_text" in obj:
                pairs.append({"text": obj["text"], "follow_text": obj["follow_text"]})
    return pairs


def default_pairs() -> List[Dict[str, str]]:
    return [
        {
            "text": "大規模言語モデルの推論では、長い履歴を保持するほど計算資源の負担が増す。旧文脈を少数の仮想トークンに圧縮して付与する手法を検証する。",
            "follow_text": "上記の要点を日本語で3点、箇条書きで述べてください。",
        },
        {
            "text": "検索クエリは短く曖昧なことが多い。過去のクリックや関連語を活用して意味を補強する必要があるが、トークン制限が問題となる。",
            "follow_text": "問題点と制約、解決の方向性を2行で述べてください。",
        },
        {
            "text": "音声認識の結果は句読点が失われるため可読性が低い。圧縮と整形を同時に行う軽量な前処理が求められる。",
            "follow_text": "2つのアプローチ案を短く提案してください。",
        },
        {
            "text": "ログ解析では時系列で膨大なイベントを扱う。古いイベントを情報要約して保持できれば、原因分析の文脈を維持できる。",
            "follow_text": "このアイデアの利点を2点説明してください。",
        },
        {
            "text": "長文の小説プロンプトでは前半の設定が後半の展開に影響する。設定を圧縮トークン化して先頭に付与すると、生成の一貫性が増す可能性がある。",
            "follow_text": "想定される失敗例を1つ挙げ、その対策を述べてください。",
        },
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_jsonl", type=Path, default=None)
    p.add_argument("--save_path", type=Path, default=Path("data/e_star_sup.pt"))
    p.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--num_virtual", type=int, default=1)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "auto"])
    p.add_argument("--init_mode", type=str, default="avg_embed", choices=["random", "avg_embed", "prefix_last_embed"])
    p.add_argument("--hidden_weight", type=float, default=0.0)
    p.add_argument("--hidden_layer_index", type=int, default=-2)
    p.add_argument("--hidden_loss_type", type=str, default="cosine", choices=["mse", "cosine", "cosine_norm"])
    p.add_argument("--ctx_len", type=int, default=64)
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--stream_dir", type=Path, default=None)
    args = p.parse_args()

    if args.pairs_jsonl is not None and args.pairs_jsonl.exists():
        pairs = load_pairs_from_jsonl(args.pairs_jsonl)
    else:
        base = default_pairs()
        pairs = []
        for i in range(args.num_samples):
            b = base[i % len(base)].copy()
            # インデックス付与でバリエーション
            b["text"] = f"{b['text']} サンプルID:{i}"
            pairs.append(b)

    # デバイス選択（auto/cpu/mps）。MPSが使える場合はMPSで高速化。
    import torch as _torch
    dev = None if args.device == "auto" else _torch.device(args.device)
    print({"status": "model_loading_start", "device": str(dev)}, flush=True)
    hf = load_tokenizer_and_model(args.model_name, requested_dtype="float32", device=dev)
    print({"status": "model_loading_done", "dtype": str(hf.dtype), "device": str(hf.device)}, flush=True)
    tok = hf.tokenizer

    samples: List[Dict[str, object]] = []
    if args.stream_dir is not None:
        args.stream_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = args.stream_dir / "manifest.jsonl"
        mf = manifest_path.open("a", encoding="utf-8")
    else:
        mf = None
    for i, ex in enumerate(pairs, 1):
        print({"status": "sample_start", "idx": i}, flush=True)
        # 文ごとに e* を個別最適化（教師: prefix+follow）
        obj = optimize_virtual(
            model_name=args.model_name,
            text=ex["text"],
            follow_text=ex["follow_text"],
            num_virtual=args.num_virtual,
            steps=args.steps,
            lr=args.lr,
            temperature=args.temperature,
            lambda_norm=0.05,
            save_path=None,
            device_override=(None if args.device == "auto" else args.device),
            init_mode=args.init_mode,
            hidden_weight=args.hidden_weight,
            hidden_layer_index=args.hidden_layer_index,
            hidden_loss_type=args.hidden_loss_type,
            hf_obj=hf,
            log_every=1,
        )

        # 隠れ表現（教師の文脈部分）を抽出し固定長に整形
        enc = tok(ex["text"], return_tensors="pt", add_special_tokens=True)
        input_ids = enc["input_ids"].to(hf.device)
        attn_mask = enc.get("attention_mask").to(hf.device)
        out = hf.model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
        h_all = out.hidden_states[args.hidden_layer_index][0]  # [L, d]
        L, d = h_all.size(0), h_all.size(1)
        H = torch.zeros((args.ctx_len, d), dtype=h_all.dtype)
        M = torch.zeros((args.ctx_len,), dtype=torch.long)
        if L >= args.ctx_len:
            H[:] = h_all[L - args.ctx_len : L].detach().cpu()
            M[:] = 1
        else:
            H[-L:] = h_all.detach().cpu()
            M[-L:] = 1

        sample = {
            "Hctx": H,
            "Hmask": M,
            "e_star": obj["e_star"],
            "text": ex["text"],
            "follow_text": ex["follow_text"],
            "meta": obj["meta"],
        }
        if args.stream_dir is not None:
            out_path = args.stream_dir / f"sample_{i:06d}.pt"
            torch.save(sample, out_path)
            if mf is not None:
                mf.write(json.dumps({"idx": i, "path": str(out_path)}) + "\n")
                mf.flush()
            print({"status": "sample_done", "idx": i, "path": str(out_path), "ctx_len": int(H.size(0)), "d": int(H.size(1))}, flush=True)
        else:
            samples.append(sample)
            print({"status": "sample_done", "idx": i, "ctx_len": int(H.size(0)), "d": int(H.size(1))}, flush=True)

    if mf is not None:
        mf.close()
    if args.stream_dir is None:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"samples": samples}, args.save_path)
        print("saved dataset:", args.save_path, {"num_samples": len(samples)})
    else:
        print("streamed dataset:", str(args.stream_dir), flush=True)


if __name__ == "__main__":
    main()


