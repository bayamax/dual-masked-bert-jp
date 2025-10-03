from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

from .config import CONFIG
from .hf_model import HFObjects, get_bos_prefixed_input_ids, load_tokenizer_and_model
from .utils import ensure_dir, get_dtype, set_seed


def get_input_embedding_matrix(hf: HFObjects) -> torch.Tensor:
    """入力量子化の影響なく、モデルの入力埋め込み行列を取得。
    LLaMA系は `model.model.embed_tokens.weight` であることが多い。
    transformersの実装差異に備えて `get_input_embeddings()` を優先利用。
    """
    emb = hf.model.get_input_embeddings()
    return emb.weight.detach().to(torch.float32)


@torch.inference_mode()
def extract_pairs(
    model_name: str,
    layer_index: int,
    max_tokens: int | None,
    batch_size: int,
    add_bos: bool,
    out_path: Path,
    dtype_name: str,
) -> Dict[str, torch.Tensor | int | str]:
    set_seed(42)
    hf = load_tokenizer_and_model(model_name=model_name, requested_dtype=dtype_name)

    vocab_size = hf.tokenizer.vocab_size
    indices = torch.arange(vocab_size, dtype=torch.long)
    if max_tokens is not None:
        indices = indices[: max_tokens]

    # 入力埋め込み行列（参照用）
    input_embedding_matrix = get_input_embedding_matrix(hf)  # [V, d]
    d_model = input_embedding_matrix.shape[1]

    # 出力バッファ
    X_hidden = torch.empty((indices.shape[0], d_model), dtype=torch.float32)
    Y_embed = torch.empty((indices.shape[0], d_model), dtype=torch.float32)

    start = 0
    for i in tqdm(range(0, indices.shape[0], batch_size), desc="extract"):
        batch_ids = indices[i : i + batch_size]
        input_ids = get_bos_prefixed_input_ids(hf.tokenizer, batch_ids, add_bos=add_bos)
        input_ids = input_ids.to(hf.device)

        outputs = hf.model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
        )
        # hidden_states: tuple(len=n_layers+1) each: [B, L, d]
        hidden_states = outputs.hidden_states
        h = hidden_states[layer_index]  # 層選択（-1 は最終層の直後）
        # 対象トークン位置: BOSがあれば index=1、なければ index=0
        pos = 1 if (add_bos and hf.tokenizer.bos_token_id is not None) else 0
        token_h = h[:, pos, :].to(torch.float32)  # [B, d]

        # 対応する入力埋め込み
        token_e = input_embedding_matrix[batch_ids.cpu()].to(torch.float32)

        X_hidden[start : start + token_h.shape[0]] = token_h.cpu()
        Y_embed[start : start + token_e.shape[0]] = token_e.cpu()
        start += token_h.shape[0]

    meta = {
        "model_name": model_name,
        "layer_index": layer_index,
        "add_bos": add_bos,
        "dtype": dtype_name,
        "vocab_sampled": int(indices.shape[0]),
        "d_model": int(d_model),
    }

    ensure_dir(out_path.parent)
    torch.save(
        {
            "X": X_hidden,
            "Y": Y_embed,
            "token_ids": indices,
            "meta": meta,
        },
        out_path,
    )
    return meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default=CONFIG.model_name)
    p.add_argument("--layer_index", type=int, default=-1)
    p.add_argument("--max_tokens", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--bos", action="store_true", default=True)
    p.add_argument("--no-bos", dest="bos", action="store_false")
    p.add_argument("--output_path", type=Path, default=Path("data/pairs.pt"))
    p.add_argument("--dtype", type=str, default="float16")
    args = p.parse_args()

    meta = extract_pairs(
        model_name=args.model_name,
        layer_index=args.layer_index,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        add_bos=args.bos,
        out_path=args.output_path,
        dtype_name=args.dtype,
    )
    print("saved:", args.output_path)
    print(meta)


if __name__ == "__main__":
    main()


