from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch

from .hf_model import load_tokenizer_and_model


def load_converter(path: Path):
    ckpt = torch.load(path, map_location="cpu")
    meta = ckpt["meta"]
    from .train_converter import build_model

    model = build_model(meta["model_type"], meta["d_model"], meta.get("hidden_dim", 2048))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, meta


@torch.inference_mode()
def compress_text_to_virtual_token(hf, text: str, converter, pool: str = "last") -> torch.Tensor:
    tok = hf.tokenizer
    enc = tok(text, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(hf.device)
    attn_mask = enc["attention_mask"].to(hf.device)

    out = hf.model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
    # 最終層の隠れ状態
    h = out.hidden_states[-1]  # [1, L, d]
    if pool == "last":
        # マスクの最後の有効位置のベクトルを採用
        lengths = attn_mask.sum(dim=1) - 1  # [1]
        idx = lengths.clamp(min=0).item()
        rep = h[0, idx, :].unsqueeze(0)  # [1, d]
    else:
        mask = attn_mask.bool()
        rep = h[mask].mean(dim=0, keepdim=True)  # [1, d]

    # 変換でトークン埋め込みへ（converterのdtype/deviceに合わせる）
    conv_param = next(converter.parameters())
    virt = converter(rep.to(device=conv_param.device, dtype=conv_param.dtype))  # [1, d]
    return virt


@torch.inference_mode()
def generate_with_virtual_prefix(hf, virtual_token_embed: torch.Tensor, prompt: str, max_new_tokens: int = 128) -> str:
    tok = hf.tokenizer
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(hf.device)
    attn_mask = enc.get("attention_mask")
    if attn_mask is not None:
        attn_mask = attn_mask.to(hf.device)

    # 通常トークンを埋め込みへ
    input_embed_layer = hf.model.get_input_embeddings()
    inp_emb = input_embed_layer(input_ids)
    # dtype/deviceをモデルの入力埋め込みに合わせる
    virt_cast = virtual_token_embed.to(device=inp_emb.device, dtype=inp_emb.dtype)
    # 先頭に仮想トークンを連結
    full_emb = torch.cat([virt_cast.unsqueeze(1), inp_emb], dim=1)
    # attention_mask を明示（仮想トークン分を1で前置）
    if attn_mask is None:
        attn_mask_full = torch.ones(full_emb.size()[:2], dtype=torch.long, device=full_emb.device)
    else:
        virt_mask = torch.ones((attn_mask.size(0), 1), dtype=attn_mask.dtype, device=attn_mask.device)
        attn_mask_full = torch.cat([virt_mask, attn_mask], dim=1)

    gen_ids = hf.model.generate(
        inputs_embeds=full_emb,
        attention_mask=attn_mask_full,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.95,
        temperature=0.8,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    text = tok.decode(gen_ids[0], skip_special_tokens=True)
    return text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--converter_path", type=Path, required=True)
    p.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B")
    p.add_argument("--compress_text", type=str, required=True)
    p.add_argument("--pool", type=str, default="last", choices=["last", "mean"])
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--max_new_tokens", type=int, default=128)
    args = p.parse_args()

    hf = load_tokenizer_and_model(args.model_name)
    converter, _ = load_converter(args.converter_path)
    device = hf.device
    converter.to(device)
    converter.eval()

    virt = compress_text_to_virtual_token(hf, args.compress_text, converter, pool=args.pool)
    out = generate_with_virtual_prefix(hf, virt, args.prompt, max_new_tokens=args.max_new_tokens)
    print(out)


if __name__ == "__main__":
    main()


